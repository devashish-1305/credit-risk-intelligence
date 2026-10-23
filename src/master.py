"""Build the master table: 1 row per loan account.

loan_accounts (base) <- customers on customer_id
                     <- bureau aggregates: all lines (bur_*, per customer)
                        and point-in-time lines (bur_pit_*, per account)
                     <- repayment months 1-3 aggregates (rep_m3_*)
                     <- collections aggregates (coll_*, analytics only)
"""
import numpy as np
import pandas as pd

from src.ingest import PROCESSED_DIR, TABLES

REPAY_WINDOW = 3  # model is scored at month 3 after origination
COLLECTIONS_COLS = ["coll_n_events", "coll_any", "coll_amount_recovered", "coll_n_pre_origination"]
BUREAU_COUNT_COLS = ["n_lines", "n_active", "n_closed", "n_written_off",
                     "any_written_off", "total_sanctioned", "total_outstanding",
                     "written_off_outstanding", "dpd_missing_cnt"]


def load_clean(processed_dir=PROCESSED_DIR):
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in TABLES}


def _ratio(num, den):
    return (num / den.replace(0, np.nan)).astype(float)


def agg_bureau(lines, key, prefix):
    """Aggregate bureau lines to `key`. DPD uses raw values (ingest imputation undone)."""
    df = lines.assign(
        raw_dpd=lines["days_past_due"].mask(lines["days_past_due_missing"] == 1),
        is_active=lines["account_status"].eq("Active"),
        is_closed=lines["account_status"].eq("Closed"),
        is_wo=lines["account_status"].eq("Written-Off"),
    )
    df["active_sanctioned"] = df["sanctioned_amount"].where(df["is_active"], 0)
    df["active_outstanding"] = df["outstanding_amount"].where(df["is_active"], 0)
    df["wo_outstanding"] = df["outstanding_amount"].where(df["is_wo"], 0)
    g = df.groupby(key)
    out = pd.DataFrame({
        "n_lines": g.size(),
        "n_active": g["is_active"].sum(),
        "n_closed": g["is_closed"].sum(),
        "n_written_off": g["is_wo"].sum(),
        "total_sanctioned": g["sanctioned_amount"].sum(),
        "total_outstanding": g["outstanding_amount"].sum(),
        "written_off_outstanding": g["wo_outstanding"].sum(),
        "utilization_active": _ratio(g["active_outstanding"].sum(), g["active_sanctioned"].sum()),
        "max_dpd": g["raw_dpd"].max(),
        "mean_dpd": g["raw_dpd"].mean(),
        "dpd_missing_cnt": g["days_past_due_missing"].sum(),
    })
    out["any_written_off"] = (out["n_written_off"] > 0).astype("int8")
    return out.add_prefix(prefix)


def _fill_thin_file(df, prefix):
    """No lines -> counts/amounts are truly 0; ratios and DPD stay NaN."""
    missing = df[f"{prefix}n_lines"].isna()
    df[f"{prefix}missing"] = missing.astype("int8")
    cols = [prefix + c for c in BUREAU_COUNT_COLS]
    df[cols] = df[cols].fillna(0)
    df[f"{prefix}any_written_off"] = df[f"{prefix}any_written_off"].astype("int8")
    return df


def agg_repayment(repayment, loans, window=REPAY_WINDOW):
    r = repayment[repayment["payment_month"] <= window].assign(
        is_missed=lambda d: d["payment_status"].eq("Missed"),
        is_late=lambda d: d["payment_status"].eq("Late"),
    )
    g = r.groupby("account_id")
    last = r.sort_values("payment_month").groupby("account_id")["outstanding_balance"].last()
    out = pd.DataFrame({
        "n_missed": g["is_missed"].sum(),
        "n_late": g["is_late"].sum(),
        "max_dpd": g["days_past_due"].max(),
        "mean_dpd": g["days_past_due"].mean(),
        "pay_ratio": g["emi_paid"].sum() / g["emi_due"].sum(),
        "balance_ratio": last / loans.set_index("account_id")["loan_amount"],
    })
    out["any_dpd30"] = (out["max_dpd"] >= 30).astype("int8")
    return out.add_prefix(f"rep_m{window}_")


def agg_collections(collections):
    g = collections.groupby("account_id")
    return pd.DataFrame({
        "coll_n_events": g.size(),
        "coll_amount_recovered": g["amount_recovered"].sum(),
        "coll_n_pre_origination": g["pre_origination"].sum(),
    })


def build_master(clean):
    loans, cust = clean["loan_accounts"], clean["customers"]
    bureau = clean["bureau_credit"]

    m = loans.merge(cust, on="customer_id", how="left", validate="many_to_one")
    m["loan_to_income"] = m["loan_amount"] / m["annual_income"]
    m["orig_year"] = m["origination_date"].dt.year
    m["orig_before_customer_since"] = (m["origination_date"] < m["customer_since"]).astype("int8")

    # Bureau, all lines (snapshot assumption A2): customer level.
    m = m.merge(agg_bureau(bureau, "customer_id", "bur_"), left_on="customer_id",
                right_index=True, how="left")
    m = _fill_thin_file(m, "bur_")

    # Bureau, point-in-time variant: lines opened on/before this account's origination.
    pairs = bureau.merge(loans[["account_id", "customer_id", "origination_date"]], on="customer_id")
    pit = pairs[pairs["date_opened"] <= pairs["origination_date"]]
    m = m.merge(agg_bureau(pit, "account_id", "bur_pit_"), left_on="account_id",
                right_index=True, how="left")
    m = _fill_thin_file(m, "bur_pit_")

    m = m.merge(agg_repayment(clean["repayment_history"], loans), left_on="account_id",
                right_index=True, how="left")

    m = m.merge(agg_collections(clean["collections_events"]), left_on="account_id",
                right_index=True, how="left")
    m[["coll_n_events", "coll_amount_recovered", "coll_n_pre_origination"]] = (
        m[["coll_n_events", "coll_amount_recovered", "coll_n_pre_origination"]].fillna(0)
    )
    m["coll_any"] = (m["coll_n_events"] > 0).astype("int8")
    return m


if __name__ == "__main__":
    master = build_master(load_clean())
    master.to_parquet(PROCESSED_DIR / "master.parquet", index=False)
    print(f"master: {master.shape} -> {PROCESSED_DIR / 'master.parquet'}")
