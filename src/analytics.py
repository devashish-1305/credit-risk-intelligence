"""Portfolio profiling and cross-tabs on the master table.

Exposure = loan_amount (the original, fixed amount at origination). This is
chosen over outstanding balance because outstanding is only known for the
months 1-3 window the model uses (A3 in docs/assumptions.md), while every
account has a loan_amount from day one.

Collections features (coll_*) are used here for descriptive analytics only,
never as model input (A2 / PLAN.md locked decisions - leakage).
"""
import pandas as pd

from src.ingest import PROCESSED_DIR

INCOME_BINS = [0, 50_000, 80_000, 120_000, float("inf")]
INCOME_LABELS = ["<50k", "50-80k", "80-120k", "120k+"]
AGE_BINS = [0, 30, 40, 50, 60, float("inf")]
AGE_LABELS = ["<30", "30-40", "40-50", "50-60", "60+"]


def load_master(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(processed_dir / "master.parquet")


def add_bands(master):
    df = master.copy()
    df["income_band"] = pd.cut(df["annual_income"], INCOME_BINS, labels=INCOME_LABELS, right=False)
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def profile_by(df, by):
    """Segment profile: accounts, exposure, default rate, relative risk.

    relative_risk = segment default rate / portfolio default rate. The
    exposure-weighted... no, the *account*-count-weighted average of
    relative_risk across all segments of a column is exactly 1.0, since it
    reduces to (sum of segment defaults) / (n_accounts * portfolio_rate).
    """
    overall_rate = df["target_default"].mean()
    total_accounts = len(df)
    total_exposure = df["loan_amount"].sum()
    total_defaulted_exposure = df.loc[df["target_default"] == 1, "loan_amount"].sum()

    defaulted_exposure = (
        df[df["target_default"] == 1].groupby(by, observed=True)["loan_amount"].sum()
    )

    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "n_accounts": g.size(),
        "exposure": g["loan_amount"].sum(),
        "default_rate": g["target_default"].mean(),
        "defaulted_exposure": defaulted_exposure,
    })
    out["defaulted_exposure"] = out["defaulted_exposure"].fillna(0.0)
    out["pct_accounts"] = out["n_accounts"] / total_accounts
    out["pct_exposure"] = out["exposure"] / total_exposure
    out["relative_risk"] = out["default_rate"] / overall_rate
    # Share of the portfolio's total *defaulted* exposure that sits in this
    # segment - i.e. where loss is concentrated, not just where risk is relatively higher.
    out["pct_defaulted_exposure"] = out["defaulted_exposure"] / total_defaulted_exposure
    return out[["n_accounts", "pct_accounts", "exposure", "pct_exposure",
                "default_rate", "relative_risk", "pct_defaulted_exposure"]
               ].sort_values("n_accounts", ascending=False)


def cross_tab_default_rate(df, row, col):
    """Default rate for each (row, col) combination, plus the account count."""
    rate = pd.crosstab(df[row], df[col], values=df["target_default"], aggfunc="mean", margins=True, margins_name="All")
    count = pd.crosstab(df[row], df[col], margins=True, margins_name="All")
    return rate, count


def collections_descriptive(df):
    """Descriptive-only: collections activity by loan_type. Not a model feature."""
    g = df.groupby("loan_type", observed=True)
    return pd.DataFrame({
        "n_accounts": g.size(),
        "pct_with_collections": g["coll_any"].mean(),
        "default_rate": g["target_default"].mean(),
        "default_rate_given_collections": df[df["coll_any"] == 1].groupby("loan_type", observed=True)["target_default"].mean(),
        "default_rate_no_collections": df[df["coll_any"] == 0].groupby("loan_type", observed=True)["target_default"].mean(),
    })


def profile_repayment_signal(master):
    """Early-warning signal at month 3: rep_m3_n_missed bucketed 0 / 1 / 2+.

    This is an association found in the data, not a causal claim - a missed
    payment in months 1-3 and eventual default could share a common cause
    (e.g. income shock) rather than one causing the other.
    """
    df = master.copy()
    df["missed_bucket"] = pd.cut(
        df["rep_m3_n_missed"], [-0.5, 0.5, 1.5, float("inf")], labels=["0", "1", "2+"]
    )
    return profile_by(df, "missed_bucket")


def low_income_employment_detail(master):
    """Account count + default rate for income_band x employment_type cells,
    flagging cells with fewer than `min_n` accounts as low-confidence."""
    df = add_bands(master)
    min_n = 30
    g = df.groupby(["income_band", "employment_type"], observed=True)
    out = g["target_default"].agg(n_accounts="size", default_rate="mean").reset_index()
    out["small_sample"] = out["n_accounts"] < min_n
    return out


def build_profiles(master):
    df = add_bands(master)
    return {
        "income": profile_by(df, "income_band"),
        "employment": profile_by(df, "employment_type"),
        "region": profile_by(df, "region"),
        "age": profile_by(df, "age_band"),
        "loan_type": profile_by(df, "loan_type"),
    }


if __name__ == "__main__":
    master = load_master()
    for name, p in build_profiles(master).items():
        print(f"\n=== {name} ===")
        print(p.round(3).to_string())

    df = add_bands(master)
    rate, count = cross_tab_default_rate(df, "income_band", "employment_type")
    print("\n=== cross-tab: income_band x employment_type (default rate) ===")
    print(rate.round(3).to_string())
    rate2, count2 = cross_tab_default_rate(df, "region", "loan_type")
    print("\n=== cross-tab: region x loan_type (default rate) ===")
    print(rate2.round(3).to_string())

    print("\n=== collections (descriptive only) ===")
    print(collections_descriptive(df).round(3).to_string())

    print("\n=== month-3 early-warning signal (rep_m3_n_missed bucket) ===")
    print(profile_repayment_signal(master).round(3).to_string())

    print("\n=== low income x employment detail (flagging n<30) ===")
    print(low_income_employment_detail(master).round(3).to_string())
