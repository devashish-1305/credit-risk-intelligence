"""Ask the Portfolio: question -> LLM-written pandas expression -> allow-list check -> execution -> grounded answer.

Flow (PLAN.md, locked decision 3):
  1. generate_query: the LLM sees only the column schema and allowed values (never rows) and returns
     one pandas expression over a DataFrame called `df`.
  2. validate: AST allow-list. Only `df`, known columns, listed methods and basic operators pass.
  3. The app shows the expression; run_query executes it only after the user clicks "Run query".
  4. answer_question: a second LLM call sees only the question, the result (<= 50 rows) and the
     account count, and writes the answer from that result alone.
"""
import ast
import json
import numbers
import os
import re

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROWS = 50
MAX_EXPR_CHARS = 600
ID_COLS = ["account_id", "customer_id"]  # never sent to the answer-generation call
SCORE_COLS = ["score_lr", "decile_lr", "risk_band"]  # exist only for 2022-23 test accounts

# Method names allowed anywhere in a chain. Anything that runs user code or strings as code
# (apply, pipe, eval, query, map, transform, ...) is absent on purpose.
ALLOWED_METHODS = {
    "groupby", "agg", "aggregate", "mean", "sum", "count", "size", "min", "max", "median", "std",
    "nunique", "quantile", "value_counts", "describe", "sort_values", "sort_index", "head", "tail",
    "nlargest", "nsmallest", "reset_index", "rename", "round", "isin", "between", "isna", "notna",
    "fillna", "abs", "add", "sub", "mul", "div", "to_frame", "idxmax", "idxmin", "unstack",
    "pivot_table", "cumsum",
}
ALLOWED_ATTRS = {"loc", "dt", "year", "month", "quarter"}  # non-call attribute access
AGG_NAMES = {"mean", "sum", "count", "size", "min", "max", "median", "std", "nunique", "first", "last"}
AGG_METHODS = {"agg", "aggregate"}  # keyword names here are named-aggregation aliases
# Keyword arguments allowed per method; any other keyword is rejected.
ALLOWED_KEYWORDS = {
    "sort_values": {"by", "ascending"},
    "sort_index": {"ascending"},
    "value_counts": {"normalize", "ascending", "dropna"},
    "head": {"n"}, "tail": {"n"},
    "nlargest": {"n", "columns"}, "nsmallest": {"n", "columns"},
    "reset_index": {"drop"},
    "round": {"decimals"},
    "pivot_table": {"index", "columns", "values", "aggfunc"},
}
ID_MESSAGE = ("This query uses account or customer identifiers, so the result is shown in the table only "
              "and not sent to the answer model.")
ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Subscript, ast.Slice,
    ast.Constant, ast.List, ast.Tuple, ast.keyword, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.USub, ast.UAdd, ast.BitAnd, ast.BitOr,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
)
_ID_REF = re.compile(r"\b(" + "|".join(ID_COLS) + r")\b")


class AskError(Exception):
    """A user-facing failure: rejected query, missing key, provider error. Never a fabricated answer."""


# --- data -------------------------------------------------------------------

def build_ask_frame(portfolio, scores):
    """Master table (all accounts, with income/age bands) left-joined with LR score, decile and band.
    The score columns are NaN for accounts outside the 2022-23 test period."""
    return portfolio.merge(scores[["account_id", *SCORE_COLS]], on="account_id", how="left")


def categorical_values(df, max_values=40):
    out = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            values = df[col].dropna().astype(str).unique()
            if len(values) <= max_values:
                out[col] = sorted(values)
    return out


def schema_text(df):
    """Column names, types and allowed category values. No rows, no statistics."""
    cats = categorical_values(df)
    lines = []
    for col in df.columns:
        line = f"- {col} ({df[col].dtype})"
        if col in cats:
            line += f": {cats[col]}"
        lines.append(line)
    return "\n".join(lines)


# --- validation -------------------------------------------------------------

def _agg_funcs(node):
    """String constants in an aggregation-function position: one name, or a list/tuple of names."""
    elts = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    if not elts or not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
        raise AskError("Aggregation functions must be given by name, e.g. 'mean' or ['mean', 'count'].")
    for e in elts:
        if e.value not in AGG_NAMES:
            raise AskError(f"'{e.value}' is not an allowed aggregation; use one of {sorted(AGG_NAMES)}.")
    return elts


def _check_call(node, columns):
    """Check a call's keywords and aggregation arguments. Returns (aggregation-name constants, aliases)."""
    method = node.func.attr
    funcs, aliases = [], set()
    if method in AGG_METHODS:
        for arg in node.args:
            funcs += _agg_funcs(arg)
        for kw in node.keywords:  # named aggregation: alias=('column', 'func') or alias='func'
            aliases.add(kw.arg)
            v = kw.value
            if isinstance(v, ast.Tuple) and len(v.elts) == 2:
                col = v.elts[0]
                if not (isinstance(col, ast.Constant) and col.value in columns):
                    raise AskError("A named aggregation must be ('column', 'function') with a known column.")
                funcs += _agg_funcs(v.elts[1])
            else:
                funcs += _agg_funcs(v)
        return funcs, aliases
    for kw in node.keywords:
        if kw.arg not in ALLOWED_KEYWORDS.get(method, set()):
            raise AskError(f"Keyword `{kw.arg}` is not allowed for `{method}`.")
    if method == "pivot_table":
        if node.args:
            raise AskError("pivot_table takes keyword arguments only (index=, columns=, values=, aggfunc=).")
        funcs += [e for kw in node.keywords if kw.arg == "aggfunc" for e in _agg_funcs(kw.value)]
    return funcs, aliases


def validate(expr, df):
    """Raise AskError unless `expr` is a single expression built only from allowed parts."""
    if not expr or len(expr) > MAX_EXPR_CHARS:
        raise AskError(f"The query is empty or longer than {MAX_EXPR_CHARS} characters.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise AskError("The query is not a single pandas expression (assignments and statements are not allowed).")
    columns = set(map(str, df.columns))
    agg_func_nodes, aliases = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise AskError(f"Not allowed in a query: {type(node).__name__}.")
        if isinstance(node, ast.Name) and node.id != "df":
            raise AskError(f"Only `df` may be referenced, not `{node.id}`.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_METHODS | ALLOWED_ATTRS:
                raise AskError(f"Method or attribute `{node.attr}` is not on the allow-list.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in ALLOWED_METHODS:
                raise AskError("Only allow-listed DataFrame methods may be called.")
            if any(kw.arg is None or kw.arg.startswith("_") for kw in node.keywords):
                raise AskError("Keyword unpacking and private keyword names are not allowed.")
            funcs, names = _check_call(node, columns)
            agg_func_nodes.update(map(id, funcs))
            aliases |= names
        if isinstance(node, ast.BinOp) and any(isinstance(s, ast.Constant) and isinstance(s.value, str)
                                               for s in (node.left, node.right)):
            raise AskError("Arithmetic on text constants is not allowed.")
    # Outside aggregation positions a string must be a column, a named-agg alias defined in this
    # expression (a column of its result) or an allowed category value.
    strings = columns | aliases | {v for vals in categorical_values(df).values() for v in vals}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in agg_func_nodes and node.value not in strings):
            raise AskError(f"Unknown column or value: '{node.value}'.")
    return tree


# --- execution --------------------------------------------------------------

def run_query(expr, df):
    """Validate, execute with no builtins and only `df` in scope, and cap the result at MAX_ROWS.
    Returns (result, truncated)."""
    tree = validate(expr, df)
    try:
        result = eval(compile(tree, "<query>", "eval"), {"__builtins__": {}}, {"df": df})
    except Exception as e:  # pandas errors on a valid-looking expression (wrong dtype, bad key, ...)
        raise AskError(f"The query failed to run: {type(e).__name__}: {e}")
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(MAX_ROWS), len(result) > MAX_ROWS
    if isinstance(result, (numbers.Number, np.generic, str, pd.Timestamp)) or result is None:
        return result, False
    raise AskError(f"The query returned a {type(result).__name__}; only a table, a column or a single value is allowed.")


def is_empty(result):
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.empty
    return result is None or (isinstance(result, float) and np.isnan(result))


def clear_stale(state, question):
    """Drop the generated query, result and answer unless they belong to the current question.
    `state` is st.session_state (or any dict)."""
    query = state.get("ask_query")
    if query is not None and query[0] != (question or "").strip():
        state.pop("ask_query", None)
        state.pop("ask_result", None)
    elif query is None:
        state.pop("ask_result", None)


def uses_scores(expr):
    return any(c in expr for c in SCORE_COLS)


def uses_identifiers(expr):
    """True if the expression mentions account_id or customer_id anywhere; its result then stays local."""
    return bool(_ID_REF.search(expr))


def strip_identifiers(result):
    """Copy of the result without account_id / customer_id (columns, index levels or a Series of ids).
    Returns None when nothing but identifiers is left. The original result is left untouched."""
    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return result
    if isinstance(result, pd.Series) and result.name in ID_COLS:
        return None
    names = [n for n in result.index.names if n not in ID_COLS]
    if len(names) < result.index.nlevels:
        result = result.reset_index(level=[n for n in result.index.names if n in ID_COLS], drop=True)
        if not names:
            result = result.reset_index(drop=True)
    if isinstance(result, pd.DataFrame):
        result = result.drop(columns=[c for c in ID_COLS if c in result.columns])
        if result.shape[1] == 0:
            return None
    elif result.index.nlevels == 1 and result.index.isin(ID_COLS).any():  # a row returned as a Series
        result = result[~result.index.isin(ID_COLS)]
        if result.empty:
            return None
    return result


def result_text(result):
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string()
    return str(result)


# --- LLM --------------------------------------------------------------------

QUERY_SYSTEM = """You translate questions about a loan portfolio into ONE pandas expression over a DataFrame named `df`.
One row per loan account. target_default = 1 means the account defaulted. Exposure = loan_amount.
Metric definitions (use exactly these):
- Default rate = defaulted accounts / total accounts, i.e. the mean of target_default.
- Defaulted exposure = sum of loan_amount where target_default = 1.
- Share of defaulted exposure = defaulted exposure / total loan_amount (an exposure-weighted measure, not the default rate).
score_lr, decile_lr and risk_band come from the champion logistic regression and exist only for accounts
originated in 2022-2023 (test accounts); they are NaN elsewhere.

Rules:
- Reply with exactly one JSON object and nothing else (no code fences, no drafts or corrections):
  {{"expression": "<one pandas expression>", "description": "<one plain-English sentence saying what the expression computes>"}}
  The expression has no assignment, imports or lambdas. Example description: "Average default flag for unemployed borrowers".
- Use only df, the listed columns and these methods: {methods}. Also allowed: .loc[...] and .dt.year/.month/.quarter. Access columns as df["col"].
- Keyword arguments allowed (any other keyword is rejected): {keywords}. In agg/aggregate, keywords are
  named-aggregation aliases: alias=("column", "func"). pivot_table takes keyword arguments only.
- Aggregation functions are given by name only: {agg_names}. Dates are not allowed as strings; filter with .dt.year or orig_year.
- To count accounts, use groupby(...).size() per group, or a boolean mask's .sum() for a total (len() is not available).
  Never use the account_id or customer_id columns, to count or otherwise.
- Use only category values exactly as listed. Keep results small (aggregate, or head(n)).
- For "which / highest / lowest / best / worst" questions, return the sorted values, e.g.
  df.groupby("region")["target_default"].mean().sort_values(ascending=False), not idxmax/idxmin, so the answer can quote the figure.
- If the question cannot be answered from these columns, reply with plain text instead of JSON: CANNOT_ANSWER: <short reason>

Columns:
{schema}"""

ANSWER_SYSTEM = """You answer a question about a loan portfolio using ONLY the query result provided.
- Every number you state must appear in the result (you may round it). Do not compute new figures or use outside knowledge.
- If the result is empty or does not answer the question, say so plainly instead of guessing.
- Be concise: 1-2 sentences. Quote the key figure, as a percentage when it is a rate.
- Do not comment on what the result lacks or on the number of accounts unless the question asks for it."""


def get_client():
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise AskError("No ANTHROPIC_API_KEY found in .env, so Ask the Portfolio is unavailable. "
                       "The rest of the dashboard is unaffected.")
    return anthropic.Anthropic()


def model_name():
    return os.getenv("ASK_MODEL", DEFAULT_MODEL)


def _call(client, system, user, max_tokens=4000):
    """One Messages call; provider failures become AskError so the app never shows a made-up answer."""
    try:
        response = client.beta.messages.create(
            model=model_name(), max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise AskError("The API key was rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        raise AskError("The language model is rate-limited right now. Try again in a minute.")
    except anthropic.APIStatusError as e:
        raise AskError(f"The language model returned an error ({e.status_code}). No answer was produced.")
    except anthropic.APIConnectionError:
        raise AskError("Could not reach the language model (network error). No answer was produced.")
    if response.stop_reason == "refusal":
        raise AskError("The language model declined this request. No answer was produced.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise AskError("The language model returned an empty response. No answer was produced.")
    return text


def parse_query_reply(text):
    """Split the model's JSON reply into (expression, description); AskError if malformed or incomplete."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise AskError("The language model's reply was not valid JSON, so no query was generated. Try rephrasing.")
    fields = [data.get(k) if isinstance(data, dict) else None for k in ("expression", "description")]
    if not all(isinstance(f, str) and f.strip() for f in fields):
        raise AskError("The language model's reply was missing the expression or description. Try rephrasing.")
    return fields[0].strip(), fields[1].strip()


def generate_query(question, df, client):
    """Returns (validated expression, description). Only the schema reaches the model, never rows."""
    keywords = "; ".join(f"{m}({', '.join(sorted(k))})" for m, k in ALLOWED_KEYWORDS.items())
    system = QUERY_SYSTEM.format(methods=", ".join(sorted(ALLOWED_METHODS)), keywords=keywords,
                                 agg_names=", ".join(sorted(AGG_NAMES)), schema=schema_text(df))
    text = _call(client, system, f"Question: {question}")
    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", text).strip()
    if text.startswith("CANNOT_ANSWER"):
        raise AskError("The question can't be answered from the available columns: "
                       + text.partition(":")[2].strip())
    expression, description = parse_query_reply(text)
    validate(expression, df)
    return expression, description


def answer_prompt(question, result, n_accounts, truncated=False, scores_used=False):
    parts = [f"Question: {question}", f"Accounts in the selection: {n_accounts}",
             f"Query result{' (first ' + str(MAX_ROWS) + ' rows)' if truncated else ''}:\n{result_text(strip_identifiers(result))}"]
    if scores_used:
        parts.append("Note: score/decile/risk band figures cover 2022-23 test accounts only; say so in the answer.")
    return "\n\n".join(parts)


def answer_question(question, result, n_accounts, client, truncated=False, scores_used=False, ids_used=False):
    if is_empty(result):
        return "The query returned no rows for this selection, so the question can't be answered from the data."
    if ids_used or strip_identifiers(result) is None:
        return ID_MESSAGE
    return _call(client, ANSWER_SYSTEM, answer_prompt(question, result, n_accounts, truncated, scores_used))
