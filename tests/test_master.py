import pandas as pd
import pytest

from src.ingest import clean_all
from src.master import build_master


@pytest.fixture(scope="module")
def clean():
    return clean_all()


@pytest.fixture(scope="module")
def master(clean):
    return build_master(clean)


def test_one_row_per_account_and_target(master, clean):
    assert len(master) == len(clean["loan_accounts"]) == 14685
    assert master["account_id"].is_unique
    assert master["target_default"].mean() == pytest.approx(0.161, abs=0.001)


def test_thin_file_flag_preserved(master, clean):
    no_bureau = set(clean["customers"]["customer_id"]) - set(clean["bureau_credit"]["customer_id"])
    thin = master[master["bur_missing"] == 1]
    assert set(thin["customer_id"]) == no_bureau and len(no_bureau) == 1008
    assert (thin[["bur_n_lines", "bur_total_outstanding"]] == 0).all().all()
    assert thin["bur_max_dpd"].isna().all()


def test_bureau_dpd_uses_raw_values(master, clean):
    per_customer = master.drop_duplicates("customer_id")
    assert per_customer["bur_dpd_missing_cnt"].sum() == 689
    assert per_customer["bur_max_dpd"].isna().sum() > per_customer["bur_missing"].sum()


def test_pit_ignores_lines_opened_after_origination(master, clean):
    shifted = dict(clean)
    shifted["bureau_credit"] = clean["bureau_credit"].assign(date_opened=pd.Timestamp("2030-01-01"))
    m = build_master(shifted)
    assert (m["bur_pit_n_lines"] == 0).all() and (m["bur_pit_missing"] == 1).all()
    pd.testing.assert_series_equal(m["bur_n_lines"], master["bur_n_lines"])


def test_repayment_features_use_months_1_to_3_only(master, clean):
    r = clean["repayment_history"].copy()
    late = r["payment_month"] > 3
    r.loc[late, ["days_past_due", "emi_paid", "payment_status"]] = [89, 0.0, "Missed"]
    m = build_master({**clean, "repayment_history": r})
    rep_cols = [c for c in master.columns if c.startswith("rep_m3_")]
    pd.testing.assert_frame_equal(m[rep_cols], master[rep_cols])
