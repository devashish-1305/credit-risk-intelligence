from types import SimpleNamespace

import json

import anthropic
import httpx2
import pandas as pd
import pytest

from src import ask
from src.dashboard import apply_filters


@pytest.fixture
def df():
    n = 120
    return pd.DataFrame({
        "account_id": [f"ACC{i:06d}" for i in range(n)],
        "customer_id": [f"CUST{i:06d}" for i in range(n)],
        "region": ["North", "South", "East"] * (n // 3),
        "loan_type": ["Auto Loan", "Mortgage"] * (n // 2),
        "employment_type": ["Salaried"] * n,
        "income_band": ["<50k", "50-80k", "80-120k", "120k+"] * (n // 4),
        "loan_amount": [1000.0 * (i + 1) for i in range(n)],
        "target_default": [1 if i % 5 == 0 else 0 for i in range(n)],
        "risk_band": ["Low", "High", None] * (n // 3),
    })


def reply(expr="df['loan_amount'].sum()", description="Total loan amount"):
    return json.dumps({"expression": expr, "description": description})


class FakeClient:
    """Mimics client.beta.messages.create; records every call."""
    def __init__(self, reply="ok", error=None, stop_reason="end_turn"):
        self.calls, self.reply, self.error, self.stop_reason = [], reply, error, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=self.reply)])


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "exec('x=1')",
    "eval('1+1')",
    "df.__class__",
    "df.groupby('region').__dict__",
    "df['__class__']",
    "x = 1",
    "(x := df)",
    "df['not_a_column'].mean()",
    "df[df['region'] == 'Atlantis']",
    "lambda: df",
    "[c for c in df]",
])
def test_allow_list_rejects_unsafe_or_unknown(expr, df):
    with pytest.raises(ask.AskError):
        ask.validate(expr, df)


@pytest.mark.parametrize("expr", [
    "df.pipe(len)",
    "df.apply(len)",
    "df.eval('loan_amount * 2')",
    "df.query('loan_amount > 0')",
    "df.groupby('region')['loan_amount'].apply(sum)",
    "df.groupby('region').agg('mean').transform('sum')",
    "df['loan_amount'].map(str)",
])
def test_method_chain_bypass_is_rejected(expr, df):
    with pytest.raises(ask.AskError):
        ask.validate(expr, df)


@pytest.mark.parametrize("expr", [
    "df.groupby('region')['loan_amount'].agg('transform')",                   # non-allow-listed name in .agg
    "df.groupby('region')['loan_amount'].agg(['mean', 'pipe'])",
    "df.groupby('region').agg(n=('loan_amount', 'transform'))",               # ... in a named-agg tuple
    "df.pivot_table(index='region', values='loan_amount', aggfunc='cumsum')",  # ... in pivot_table aggfunc
    "df.groupby('region')['loan_amount'].agg('loan_amount')",                 # column name as agg function
    "df.groupby('region').agg(n=('loan_amount', 'region'))",
    "df.pivot_table(index='region', values='loan_amount', aggfunc=['mean', 'loan_type'])",
    "df.sort_values('loan_amount', ignore_index=True)",                       # unknown keyword arguments
    "df.groupby('region', as_index=False)['loan_amount'].sum()",
    "df['loan_amount'].sum().round(decimals=2, out=1)",
    "df.sort_values('mean')",                                                 # agg name outside agg context
    "df.sort_values('loan_amount', ascending='ascending')",                   # keyword name used as a string
])
def test_string_and_keyword_policy_rejects(expr, df):
    with pytest.raises(ask.AskError):
        ask.validate(expr, df)


@pytest.mark.parametrize("expr", [
    "[0] * 999999999",                     # list repetition (memory exhaustion)
    "999999999 * [0]",
    "(0,) * 999999999",                    # tuple repetition
    "'a' * 999999999",                     # string repetition
    "df['region'].isin(['North'] * 999999999)",
])
def test_list_tuple_and_string_arithmetic_is_rejected(expr, df):
    with pytest.raises(ask.AskError, match="Arithmetic on lists, tuples or text constants"):
        ask.validate(expr, df)  # validator only; never executed


@pytest.mark.parametrize("expr", [
    "df.groupby('region')['loan_amount'].agg(['mean', 'count'])",
    "df.pivot_table(index='region', columns='loan_type', values='target_default', aggfunc='mean')",
    "df.groupby('region')['loan_amount'].sum().nlargest(n=2).round(decimals=1)",
    "df['region'].value_counts(normalize=True).head(n=3)",
])
def test_string_and_keyword_policy_accepts(expr, df):
    ask.validate(expr, df)


@pytest.mark.parametrize("expr", [
    "df['customer_id'].max()",
    "df.groupby('account_id')['loan_amount'].sum().idxmax()",          # idxmax returning an account ID
    "df.groupby('account_id')['loan_amount'].sum().nlargest(5)",       # result indexed by account_id
])
def test_identifier_queries_never_reach_the_answer_model(expr, df):
    result, truncated = ask.run_query(expr, df)  # accepted by the validator, so safe to run
    client = FakeClient()
    answer = ask.answer_question("q", result, len(df), client, truncated, ask.uses_scores(expr),
                                 ask.uses_identifiers(expr))
    assert answer == ask.ID_MESSAGE and client.calls == []


def test_row_series_identifiers_are_not_sent(df):
    expr = "df.loc[df['loan_amount'].idxmax()]"  # a whole row as a Series; no ID column named
    result, truncated = ask.run_query(expr, df)
    assert not ask.uses_identifiers(expr) and result["account_id"] == "ACC000119"  # UI keeps the full row
    client = FakeClient(reply="The largest loan is 120,000.")
    ask.answer_question("Largest loan?", result, len(df), client, truncated, ask.uses_scores(expr),
                        ask.uses_identifiers(expr))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "ACC0" not in prompt and "CUST0" not in prompt and "account_id" not in prompt
    assert "120000" in prompt


def test_query_prompt_lists_keywords_and_counting_rule(df):
    client = FakeClient(reply=reply())
    ask.generate_query("q", df, client)
    system = client.calls[0]["system"]
    assert "sort_values(ascending, by)" in system and "pivot_table(aggfunc, columns, index, values)" in system
    assert "groupby(...).size()" in system and "Never use the account_id or customer_id columns" in system


def test_valid_expression_runs_on_fixture(df):
    expr = "df.groupby('region')['target_default'].mean().sort_values(ascending=False)"
    result, truncated = ask.run_query(expr, df)
    assert isinstance(result, pd.Series) and not truncated
    assert result.to_dict() == df.groupby("region")["target_default"].mean().to_dict()
    named, _ = ask.run_query("df.groupby('region').agg(n=('account_id', 'size')).sort_values('n')", df)
    assert named["n"].sum() == len(df)
    scalar, _ = ask.run_query("df.loc[df['loan_type'] == 'Mortgage', 'loan_amount'].sum()", df)
    assert scalar == df.loc[df["loan_type"] == "Mortgage", "loan_amount"].sum()


def test_non_table_results_are_rejected(df):
    with pytest.raises(ask.AskError, match="only a table"):
        ask.run_query("df.groupby('region')", df)
    with pytest.raises(ask.AskError, match="only a table"):
        ask.run_query("df['loan_amount'].mean", df)


def test_results_capped_at_50_rows(df):
    result, truncated = ask.run_query("df[['account_id', 'loan_amount']]", df)
    assert len(result) == ask.MAX_ROWS and truncated


def test_filters_reduce_the_data_queried(df):
    view = apply_filters(df, {"region": ["North"]})
    result, _ = ask.run_query("df['account_id'].count()", view)
    assert result == len(view) == 40 < len(df)


def test_empty_result_is_handled_without_llm(df):
    result, _ = ask.run_query("df[df['loan_amount'] < 0]", df)
    client = FakeClient()
    answer = ask.answer_question("Any negative loans?", result, len(df), client)
    assert "no rows" in answer and client.calls == []


def test_answer_prompt_contains_only_the_result(df):
    result, _ = ask.run_query("df.groupby('region')['target_default'].mean()", df)
    client = FakeClient(reply="North has the highest default rate.")
    ask.answer_question("Default rate by region?", result, len(df), client)
    prompt = client.calls[0]["messages"][0]["content"]
    assert ask.result_text(result) in prompt and f"Accounts in the selection: {len(df)}" in prompt
    assert "ACC000007" not in prompt and "loan_amount" not in prompt  # no raw rows / other columns


def test_query_prompt_has_schema_but_no_rows(df, monkeypatch):
    client = FakeClient(reply="```json\n" + reply() + "\n```")
    assert ask.generate_query("Total exposure?", df, client) == ("df['loan_amount'].sum()", "Total loan amount")
    sent = client.calls[0]["system"] + client.calls[0]["messages"][0]["content"]
    assert "loan_amount" in sent and "ACC000007" not in sent and "120000" not in sent


def test_missing_key_or_provider_error_gives_clear_message(df, monkeypatch):
    monkeypatch.setattr(ask, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ask.AskError, match="No ANTHROPIC_API_KEY"):
        ask.get_client()
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    with pytest.raises(ask.AskError, match="network error"):
        ask.generate_query("Total exposure?", df, FakeClient(error=err))
    with pytest.raises(ask.AskError, match="declined"):
        ask.answer_question("q", pd.Series([1]), 1, FakeClient(stop_reason="refusal"))
    with pytest.raises(ask.AskError, match="allow-list"):  # LLM output that fails validation never runs
        ask.generate_query("q", df, FakeClient(reply=reply("df.pipe(len)")))


def test_metric_definitions_are_in_the_query_prompt(df):
    client = FakeClient(reply=reply())
    ask.generate_query("Share of defaulted exposure?", df, client)
    system = client.calls[0]["system"]
    assert "Default rate = defaulted accounts / total accounts" in system
    assert "Defaulted exposure = sum of loan_amount where target_default = 1" in system
    assert "Share of defaulted exposure = defaulted exposure / total loan_amount" in system


def test_exposure_share_differs_from_default_rate_on_fixture():
    # 1 of 4 accounts defaulted (25%), but it is the largest loan: 700 of 1000 exposure (70%).
    small = pd.DataFrame({"account_id": list("abcd"), "loan_amount": [700.0, 100.0, 100.0, 100.0],
                          "target_default": [1, 0, 0, 0], "region": ["North"] * 4})
    rate, _ = ask.run_query("df['target_default'].mean()", small)
    share, _ = ask.run_query("df.loc[df['target_default'] == 1, 'loan_amount'].sum() / df['loan_amount'].sum()", small)
    assert rate == pytest.approx(0.25) and share == pytest.approx(0.70)
    assert rate != pytest.approx(share)


def test_identifiers_are_stripped_from_the_answer_prompt_but_not_the_result():
    res = pd.DataFrame({"account_id": ["ACC900001", "ACC900002"], "customer_id": ["CUST900001", "CUST900002"],
                        "loan_amount": [111.0, 222.0]})
    client = FakeClient(reply="Two loans.")
    ask.answer_question("Show two loans", res, 2, client)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "ACC9" not in prompt and "CUST9" not in prompt and "account_id" not in prompt
    assert "111" in prompt and "222" in prompt
    assert list(res.columns) == ["account_id", "customer_id", "loan_amount"]  # UI keeps the full result
    # identifiers as index levels or as a lone column are removed too
    indexed = res.set_index("account_id")["loan_amount"]
    assert "ACC9" not in ask.answer_prompt("q", indexed, 2)
    assert ask.strip_identifiers(res["account_id"]) is None
    only_ids = FakeClient()
    assert "identifiers" in ask.answer_question("q", res[["account_id"]], 2, only_ids) and only_ids.calls == []


def test_default_model_is_sonnet_and_configurable(monkeypatch):
    monkeypatch.delenv("ASK_MODEL", raising=False)
    assert ask.model_name() == "claude-sonnet-5-5"
    monkeypatch.setenv("ASK_MODEL", "claude-opus-5-5")
    assert ask.model_name() == "claude-opus-5-5"


def test_stale_query_and_result_cleared_when_question_changes():
    state = {"ask_query": ("Default rate by region", "df['x']"), "ask_result": ("r", False, "a", 5, False)}
    ask.clear_stale(state, "Default rate by region")
    assert "ask_query" in state and "ask_result" in state          # same question: kept
    ask.clear_stale(state, "  Default rate by region  ")
    assert "ask_query" in state                                    # whitespace is ignored
    ask.clear_stale(state, "Default rate by loan type")
    assert "ask_query" not in state and "ask_result" not in state  # changed: both cleared
    state = {"ask_query": ("q", "df['x']"), "ask_result": ("r", False, "a", 5, False)}
    ask.clear_stale(state, "")
    assert state == {}                                             # cleared question: everything cleared
    ask.clear_stale({"ask_result": 1}, "q")                        # orphan result without a query is dropped


def test_prompts_carry_ranking_and_answer_style_instructions(df):
    client = FakeClient(reply=reply())
    ask.generate_query("Which region is highest?", df, client)
    system = client.calls[0]["system"]
    assert "sort_values(ascending=False)" in system and "not idxmax/idxmin" in system
    assert "concise" in ask.ANSWER_SYSTEM and "percentage" in ask.ANSWER_SYSTEM
    assert "Do not comment on what the result lacks" in ask.ANSWER_SYSTEM


def test_query_reply_json_parsing(df):
    assert ask.parse_query_reply(reply("df['x']", " Some text ")) == ("df['x']", "Some text")
    for bad in ["df['loan_amount'].sum()", "{not json", "[1, 2]", "null",
                json.dumps({"expression": "df['loan_amount'].sum()"}),
                json.dumps({"description": "Total"}),
                json.dumps({"expression": "", "description": "Total"}),
                json.dumps({"expression": "df['loan_amount'].sum()", "description": 5})]:
        with pytest.raises(ask.AskError):
            ask.parse_query_reply(bad)
        with pytest.raises(ask.AskError):  # nothing is validated or run on a bad reply
            ask.generate_query("q", df, FakeClient(reply=bad))


def test_expression_in_json_is_still_validated(df):
    for expr in ["df.apply(len)", "__import__('os')", "df['nope'].sum()", "x = 1"]:
        with pytest.raises(ask.AskError):
            ask.generate_query("q", df, FakeClient(reply=reply(expr, "Looks harmless")))


def test_cannot_answer_stays_plain_text(df):
    with pytest.raises(ask.AskError, match="can't be answered"):
        ask.generate_query("Weather?", df, FakeClient(reply="CANNOT_ANSWER: no weather data"))
