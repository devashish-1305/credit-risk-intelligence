import pandas as pd
import pytest

from src.analytics import (
    add_bands,
    build_profiles,
    cross_tab_default_rate,
    load_master,
    profile_repayment_signal,
)


@pytest.fixture(scope="module")
def master():
    return load_master()


@pytest.fixture(scope="module")
def profiles(master):
    return build_profiles(master)


def test_segment_counts_sum_to_total_accounts(profiles):
    for name, p in profiles.items():
        assert p["n_accounts"].sum() == 14685, name


def test_exposure_shares_sum_to_one(profiles):
    for name, p in profiles.items():
        assert p["pct_exposure"].sum() == pytest.approx(1.0, abs=1e-6), name
        assert p["pct_accounts"].sum() == pytest.approx(1.0, abs=1e-6), name


def test_defaulted_exposure_shares_sum_to_one(profiles, master):
    for name, p in profiles.items():
        assert p["pct_defaulted_exposure"].sum() == pytest.approx(1.0, abs=1e-6), name
    signal = profile_repayment_signal(master)
    assert signal["pct_defaulted_exposure"].sum() == pytest.approx(1.0, abs=1e-6)


def test_relative_risk_weighted_average_is_one(profiles):
    for name, p in profiles.items():
        weighted = (p["relative_risk"] * p["n_accounts"]).sum() / p["n_accounts"].sum()
        assert weighted == pytest.approx(1.0, abs=1e-6), name


def test_bands_cover_every_account_with_no_gaps(master):
    df = add_bands(master)
    assert df["income_band"].notna().all()
    assert df["age_band"].notna().all()


def test_cross_tab_margins_match_profile(master):
    df = add_bands(master)
    rate, count = cross_tab_default_rate(df, "income_band", "employment_type")
    assert count.loc["All", "All"] == 14685
    assert rate.loc["All", "All"] == pytest.approx(df["target_default"].mean(), abs=1e-6)
