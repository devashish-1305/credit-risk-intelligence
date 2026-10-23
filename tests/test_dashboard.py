import pytest

from src import dashboard as d
from src.model import BAND_LABELS


@pytest.fixture(scope="module")
def data():
    portfolio = d.load_portfolio()
    scores = d.load_test_scores()
    return portfolio, d.scored_test(portfolio, scores), scores, d.load_lr_drivers(), d.load_model_summary()


def test_loaders_return_expected_shapes(data):
    portfolio, test, scores, drivers, summary = data
    assert len(portfolio) == 14685 and {"income_band", "age_band"} <= set(portfolio.columns)
    assert len(scores) == summary["split"]["n_test"] == len(test)
    assert len(drivers) == 5 * len(scores)
    assert len(d.shap_top10(summary)) == 10


def test_filters_reduce_rows_correctly(data):
    portfolio, test, *_ = data
    assert len(d.apply_filters(portfolio, {})) == len(portfolio)
    assert len(d.apply_filters(portfolio, {"region": []})) == len(portfolio)
    region = portfolio["region"].iloc[0]
    one = d.apply_filters(portfolio, {"region": [region]})
    assert len(one) == (portfolio["region"] == region).sum() < len(portfolio)
    both = d.apply_filters(portfolio, {"region": [region], "loan_type": [portfolio["loan_type"].iloc[0]]})
    assert 0 < len(both) <= len(one) and (both["region"] == region).all()
    assert d.apply_filters(test, {"region": [region]})["region"].eq(region).all()


def test_kpis_and_model_tables_follow_the_selection(data):
    portfolio, test, *_ = data
    k = d.kpis(portfolio)
    assert k["accounts"] == 14685 and 0.15 < k["default_rate"] < 0.17
    table = d.model_metrics_table(test)
    assert list(table["Role"]) == ["Champion", "Challenger", "Sensitivity variant"]
    assert table["AUC"].between(0.5, 1).all()
    assert d.lr_decile_table(test)["n"].sum() == len(test)
    assert d.risk_band_table(test)["n"].sum() == len(test)
    assert d.model_metrics_table(test.head(50))["AUC"].isna().all()


def test_account_lookup_returns_band_and_five_contributions(data):
    _, test, scores, drivers, _ = data
    res = d.account_lookup(scores["account_id"].iloc[0], scores, drivers)
    assert res["risk_band"] in BAND_LABELS and 1 <= res["decile"] <= 10
    assert len(res["drivers"]) == 5 and res["drivers"]["contribution"].notna().all()
    assert "score" not in res and "probability" not in res
    with pytest.raises(KeyError):
        d.account_lookup("ACC000000", scores, drivers)


def test_driver_labels_are_plain_english(data):
    _, _, scores, drivers, _ = data
    assert d.driver_label("loan_amount", "306310.51") == "Loan amount: 306,311"
    assert d.driver_label("collateral_flag", "1") == "Loan is secured by collateral: yes"
    assert d.driver_label("years_employed", "missing") == "Years employed: not recorded"
    used = set(drivers["feature"])
    assert used <= set(d.FEATURE_LABELS), used - set(d.FEATURE_LABELS)
