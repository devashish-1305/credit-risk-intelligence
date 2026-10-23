"""Load the 5 raw tables, fix data-quality issues, write cleaned parquet.

Grain is unchanged here: no rows are dropped and no tables are joined (that is
the master-table step). Imputed values are for analytics only; the model
pipeline refits imputation on train data, using the `_missing` flags kept here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Last date covered by the loan and collections tables (max date is 2023-12-30).
DATA_END = pd.Timestamp("2023-12-31")

TABLES = ["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"]
DATE_COLS = {
    "customers": ["customer_since"],
    "loan_accounts": ["origination_date"],
    "bureau_credit": ["date_opened", "date_closed"],
    "collections_events": ["event_date"],
}
GENDER_MAP = {"M": "Male", "F": "Female"}
SECURED_TYPES = {"Auto Loan", "Mortgage"}
MIN_WORKING_AGE = 14


def load_raw(raw_dir=RAW_DIR):
    """Read the 5 CSVs with date columns parsed."""
    raw_dir = Path(raw_dir)
    return {
        name: pd.read_csv(raw_dir / f"{name}.csv", parse_dates=DATE_COLS.get(name, []))
        for name in TABLES
    }


def _strip_strings(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df


def _flag_missing(df, cols):
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df


def clean_customers(customers):
    df = _strip_strings(customers)
    df["gender"] = df["gender"].replace(GENDER_MAP)

    df = _flag_missing(df, ["years_employed", "num_dependents", "years_at_residence"])

    # Cap observed years_employed at age - 14; values above it are impossible.
    max_years = df["age"] - MIN_WORKING_AGE
    over = df["years_employed"] > max_years
    df["years_employed_capped"] = over.astype("int8")
    df.loc[over, "years_employed"] = max_years[over]

    df["years_employed"] = df["years_employed"].fillna(
        df.groupby("employment_type")["years_employed"].transform("median")
    )
    df["years_employed"] = df["years_employed"].clip(upper=max_years)
    df["num_dependents"] = df["num_dependents"].fillna(df["num_dependents"].mode()[0])
    df["years_at_residence"] = df["years_at_residence"].fillna(df["years_at_residence"].median())
    return df


def clean_loans(loans):
    df = _strip_strings(loans)
    secured = df["loan_type"].isin(SECURED_TYPES).astype(int)
    if not (secured == df["collateral_flag"]).all():
        raise ValueError("collateral_flag inconsistent with loan_type")

    df = _flag_missing(df, ["interest_rate_pct"])
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(
        df.groupby("loan_type")["interest_rate_pct"].transform("median")
    )
    return df


def clean_bureau(bureau):
    df = _strip_strings(bureau)
    df = _flag_missing(df, ["days_past_due"])
    df["days_past_due"] = df["days_past_due"].fillna(
        df.groupby("account_status")["days_past_due"].transform("median")
    )
    # date_closed stays null for Active lines (null means "still open").
    df["future_dated"] = (
        (df["date_opened"] > DATA_END) | (df["date_closed"] > DATA_END)
    ).astype("int8")
    return df


def clean_repayment(repayment):
    df = _strip_strings(repayment)
    if (df["emi_paid"] > df["emi_due"] + 0.01).any():
        raise ValueError("emi_paid exceeds emi_due")
    return df


def clean_collections(collections, loans):
    df = _strip_strings(collections)
    orig = df["account_id"].map(loans.set_index("account_id")["origination_date"])
    df["pre_origination"] = (df["event_date"] < orig).astype("int8")
    return df


def clean_all(raw_dir=RAW_DIR, out_dir=None):
    """Clean every table; write parquet to out_dir if given."""
    raw = load_raw(raw_dir)
    loans = clean_loans(raw["loan_accounts"])
    clean = {
        "customers": clean_customers(raw["customers"]),
        "loan_accounts": loans,
        "bureau_credit": clean_bureau(raw["bureau_credit"]),
        "repayment_history": clean_repayment(raw["repayment_history"]),
        "collections_events": clean_collections(raw["collections_events"], loans),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in clean.items():
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return clean


if __name__ == "__main__":
    for name, df in clean_all(out_dir=PROCESSED_DIR).items():
        print(f"{name}: {len(df)} rows -> {PROCESSED_DIR / (name + '.parquet')}")
