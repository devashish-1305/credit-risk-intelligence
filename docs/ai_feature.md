# Ask the Portfolio

A question box in the dashboard's "Ask the Portfolio" tab. A language model turns a plain-English
question into one pandas expression. The expression is checked, shown to the user and run only after
the user clicks **Run query**. A second model call writes a short answer from the result alone.

Code: `src/ask.py` (logic) and `app/app.py` (tab). Tests: `tests/test_ask.py` (the language model is mocked).

## Flow

| Step | What happens | What the model sees |
|---|---|---|
| 1. Question | The user types a question and clicks **Generate query**. | — |
| 2. Generate | The model returns JSON with an `expression` (one pandas expression over `df`) and a one-sentence plain-English `description`, or `CANNOT_ANSWER: <reason>` as plain text. Malformed JSON or a missing field shows an error and nothing runs. | Column names, data types and allowed category values. No rows and no statistics. |
| 3. Validate | An AST allow-list check (below). Anything that fails is rejected with a reason and never runs. | — |
| 4. Review (human in the loop) | The description ("This query will compute: …") is shown, with the expression in a collapsed "Show query" expander. The description is written by the model; the expression is what runs. Nothing runs until the user clicks **Run query**. | — |
| 5. Execute | The query is validated again and evaluated with empty `__builtins__` and only `df` in scope. The result is capped at 50 rows. | — |
| 6. Answer | The second call writes 1–3 sentences in which every number must come from the result. An empty result returns a fixed "no rows" message without a model call. | The question, the result (≤ 50 rows) and the account count. `account_id` and `customer_id` are removed first (columns or index levels); the table in the app keeps the full result. |

`df` is the master table (14,685 accounts, one row per account, with income and age bands), left-joined
with the champion LR's `score_lr`, `decile_lr` and `risk_band`. These three columns exist only for
the 2022–23 test accounts. When a query uses them, the answer prompt and the tab both state that the
figures cover test accounts only.

The sidebar filters are applied to `df` before generation and execution, and the tab shows
"Answered on N accounts".

## Metric definitions

The query-generation prompt defines these terms, so the model uses them the same way every time:

- **Default rate** = defaulted accounts / total accounts (the mean of `target_default`).
- **Defaulted exposure** = sum of `loan_amount` where `target_default = 1`.
- **Share of defaulted exposure** = defaulted exposure / total `loan_amount`.

## Trigger

The feature is user-initiated only:
- The user types a question, which triggers the query-generation call.
- The user clicks **Run query**, which triggers execution and the answer call.

Nothing runs automatically, on page load or when a filter changes. If the filters change after a query
was generated, **Run query** uses the current filters, and the account count shown reflects them.

## Allow-list (what a query may contain)

- **Names:** only `df`. No other variables or builtins: `open`, `eval`, `exec` and `__import__` all fail.
- **Calls:** only methods on the allow-list. These are the filtering, `groupby`, `agg`, sort, `head`,
  `nlargest`, `value_counts`, `pivot_table` and basic statistics methods (the full list is
  `ALLOWED_METHODS`). Non-call attributes are limited to `loc`, `dt`, `year`, `month` and `quarter`.
  The check applies at every link of a method chain, so `apply`, `pipe`, `eval`, `query`, `map` and
  `transform` are rejected wherever they appear.
- **Strings:** each must be a known column, an allowed category value, an aggregation name, a named-agg
  alias defined in the same expression, or a date (`YYYY-MM[-DD]`). Unknown columns are rejected.
- **Syntax:** comparisons, `&`/`|`/`~`, and `+ - * / // %` on numbers. No assignments, lambdas,
  comprehensions, dicts, dunder or private attributes, or `**kwargs`. Expressions are limited to 600 characters.
- **Result:** must be a DataFrame, a Series or a scalar. Anything else, such as a GroupBy object or a
  bound method, is rejected.

## Example: false positive (wrong query, plausible wrong answer)

> **Question:** "What share of our exposure defaulted?"
> **Generated:** `df["target_default"].mean()`
> **Answer:** "About 16.1% of the portfolio defaulted."

The query is valid, runs and returns a believable number, but it answers a different question. (This example predates the definitions above; with them in the prompt the model should now write the exposure-weighted query, but prompt guidance only reduces wrong queries and does not eliminate them. The example stays because the allow-list cannot detect this class of error.) It gives
the share of *accounts* that defaulted (16.1%), not the share of *exposure* (defaulted loan amount ÷ total
loan amount = 21.1%). The allow-list cannot catch this, because the query is safe; it is just the wrong
calculation. **Mitigation:** the query is shown before it runs, and a reviewer who knows the definitions
should spot that no `loan_amount` appears. Similar risks are ambiguous segment words ("high income"
mapped to `120k+` when the user meant `80-120k` and above) and account-weighted versus
exposure-weighted rates.

## Example: false negative (valid question blocked)

> **Question:** "What is the default rate by origination quarter?"
> **Generated:** `df.groupby(df["origination_date"].dt.to_period("Q"))["target_default"].mean()`
> **Result:** blocked, with "Only allow-listed DataFrame methods may be called."

`to_period` is harmless but not on the allow-list, so a reasonable question is refused. Renaming columns
with a dict (`rename(columns={...})`) is blocked for the same reason, because dict literals are not
allowed. **Mitigation:** the user can rephrase (for example, "by origination year" works through
`orig_year`), or a reviewer can add the method to `ALLOWED_METHODS` after checking that it cannot run
code.

## Human-in-the-loop step

The user is the reviewer. Between **Generate query** and **Run query**, the model's description is on
screen and the exact expression is one click away ("Show query"). The description can be wrong, so
the expression is what to check:
- which columns and filters it uses,
- whether a "rate" is account-weighted or exposure-weighted,
- whether it touches the test-only score columns.

The result table is shown under the answer, so every number in the answer can be traced back to it.
Answers are for exploration; any figure used in a decision should be re-derived from the query.

## Failure handling

The following each show a clear message and produce no answer:
- missing `ANTHROPIC_API_KEY`,
- rejected key, rate limit, provider error or network error,
- a refusal or empty model reply,
- a query that fails validation, fails at run time or returns the wrong type.

The other tabs keep working. `CANNOT_ANSWER` from the model (for example "What's the weather in Paris?")
is shown as "can't be answered from the available columns".

## Configuration

- **Model:** `ASK_MODEL` environment variable, default `claude-sonnet-5-5`. Calls use low effort and the
  provider's server-side refusal fallback (`fallbacks="default"`).
- **Key:** `ANTHROPIC_API_KEY` in `.env`.

## Known limits

- **Correctness is not verified.** The allow-list makes queries safe, not right (see the false positive above).
- **The answer model can still misstate a number.** It is instructed to use only the result, but its
  output is not checked number by number. The result table is shown so the user can check it.
- **Results are capped at 50 rows.** The answer is told when the result was truncated, but questions
  needing the full list ("all accounts where...") get the first 50 only.
- **Single expression only.** No multi-step analysis, joins to other tables, charts or `pd.*` functions
  such as `pd.crosstab` (use `pivot_table` instead).
- **No conversation memory.** Each question is independent.
- **Score columns cover 2022–23 test accounts only.** The LR score is an uncalibrated ranking score
  (L3 in `docs/limitations.md`).
- **Only `account_id` and `customer_id` are removed from the answer request.** Other row-level values
  (loan amounts, dates, a column renamed with `rename`) in a result are still sent, and an identifier
  hidden under a different alias would not be recognised. The data is synthetic; a real deployment would
  need a data-handling review. A result made only of identifiers gets a fixed message instead of a written answer.
- **Allowed category values are sent to the provider** as part of the schema.
