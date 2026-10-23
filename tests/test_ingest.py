import pandas as pd
import pytest

from src.ingest import clean_all, load_raw

RAW_COUNTS = {
    "customers": 10000,
    "loan_accounts": 14685,
    "bureau_credit": 22979,
    "repayment_history": 176220,
    "collections_events": 8635,
}


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def clean():
    return clean_all()


def test_row_counts_unchanged(clean):
    assert {name: len(df) for name, df in clean.items()} == RAW_COUNTS


def test_gender_standardized(clean):
    assert set(clean["customers"]["gender"]) == {"Male", "Female"}


@pytest.mark.parametrize(
    "table,col",
    [
        ("customers", "years_employed"),
        ("customers", "num_dependents"),
        ("customers", "years_at_residence"),
        ("loan_accounts", "interest_rate_pct"),
        ("bureau_credit", "days_past_due"),
    ],
)
def test_imputed_no_nulls_and_flags_match_raw(raw, clean, table, col):
    assert clean[table][col].notna().all()
    assert clean[table][f"{col}_missing"].sum() == raw[table][col].isna().sum()


def test_key_integrity(clean):
    c, l, b, r, e = (clean[t] for t in RAW_COUNTS)
    assert c["customer_id"].is_unique and l["account_id"].is_unique
    assert b["bureau_id"].is_unique and e["collection_event_id"].is_unique
    assert not r.duplicated(["account_id", "payment_month"]).any()
    assert l["customer_id"].isin(c["customer_id"]).all()
    assert b["customer_id"].isin(c["customer_id"]).all()
    assert r["account_id"].isin(l["account_id"]).all()
    assert e["account_id"].isin(l["account_id"]).all()


def test_years_employed_capped(raw, clean):
    c = clean["customers"]
    assert (c["years_employed"] <= c["age"] - 14).all()
    raw_c = raw["customers"]
    assert c["years_employed_capped"].sum() == (raw_c["years_employed"] > raw_c["age"] - 14).sum()
