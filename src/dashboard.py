"""Loading, filtering and lookup functions behind the Streamlit app (app/app.py).

Reads the artifacts written by src/model.py; nothing is retrained here.
LR is the champion model and LightGBM the challenger (PLAN.md, locked decisions).
"""
import json

import numpy as np
import pandas as pd

from src.analytics import add_bands, profile_by
from src.ingest import PROCESSED_DIR
from src.model import BAND_LABELS, decile_table, metrics

FILTER_COLS = {"region": "Region", "loan_type": "Loan type",
               "employment_type": "Employment type", "income_band": "Income band"}
MODEL_ROLES = {
    "lr": ("Logistic regression", "Champion", "score_lr"),
    "lgbm": ("LightGBM", "Challenger", "score_lgbm"),
    "lgbm_pit": ("LightGBM, opened-before-origination bureau", "Sensitivity variant", "score_lgbm_pit"),
}
MIN_METRIC_ROWS = 200  # AUC/KS on fewer filtered test accounts is mostly noise

FEATURE_LABELS = {
    "loan_amount": "Loan amount", "interest_rate_pct": "Interest rate (%)",
    "interest_rate_pct_missing": "Interest rate not recorded", "tenure_months": "Loan term (months)",
    "collateral_flag": "Loan is secured by collateral", "loan_to_income": "Loan size relative to income",
    "loan_type": "Loan type", "loan_purpose": "Loan purpose", "disbursal_channel": "Disbursal channel",
    "age": "Customer age", "annual_income": "Annual income", "years_employed": "Years employed",
    "years_employed_missing": "Years employed not recorded", "years_employed_capped": "Years employed adjusted for age",
    "num_dependents": "Number of dependents", "num_dependents_missing": "Dependents not recorded",
    "years_at_residence": "Years at current residence", "years_at_residence_missing": "Residence years not recorded",
    "education_level": "Education level", "employment_type": "Employment type",
    "occupation_sector": "Occupation sector", "region": "Region", "city_tier": "City tier",
    "housing_type": "Housing type",
    "bur_n_lines": "Credit lines on bureau", "bur_n_active": "Active credit lines",
    "bur_n_closed": "Closed credit lines", "bur_n_written_off": "Written-off credit lines",
    "bur_total_sanctioned": "Total credit sanctioned elsewhere",
    "bur_total_outstanding": "Total outstanding elsewhere",
    "bur_written_off_outstanding": "Outstanding on written-off lines",
    "bur_utilization_active": "Utilisation of active credit lines",
    "bur_max_dpd": "Worst bureau days past due", "bur_mean_dpd": "Average bureau days past due",
    "bur_dpd_missing_cnt": "Bureau lines with unknown days past due",
    "bur_any_written_off": "Has a written-off credit line", "bur_missing": "No bureau record (thin file)",
    "rep_m3_n_missed": "Missed payments in months 1-3", "rep_m3_n_late": "Late payments in months 1-3",
    "rep_m3_max_dpd": "Worst days past due in months 1-3", "rep_m3_mean_dpd": "Average days past due in months 1-3",
    "rep_m3_pay_ratio": "Share of due amount paid in months 1-3",
    "rep_m3_balance_ratio": "Balance remaining after month 3 (ratio)",
    "rep_m3_any_dpd30": "30+ days past due in months 1-3",
}
_MONEY = {"loan_amount", "annual_income", "bur_total_sanctioned", "bur_total_outstanding",
          "bur_written_off_outstanding"}
_FLAGS = {c for c in FEATURE_LABELS if c.endswith("_missing") or c.startswith(("bur_any", "rep_m3_any"))
          or c in {"collateral_flag", "years_employed_capped", "bur_missing"}}


# --- loaders ----------------------------------------------------------------

def load_portfolio(processed_dir=PROCESSED_DIR):
    """Master table (one row per account) with income and age bands."""
    return add_bands(pd.read_parquet(processed_dir / "master.parquet"))


def load_test_scores(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(processed_dir / "test_scores.parquet")


def load_lr_drivers(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(processed_dir / "test_lr_drivers.parquet")


def load_model_summary(processed_dir=PROCESSED_DIR):
    return json.loads((processed_dir / "model_metrics.json").read_text())


# --- filters ----------------------------------------------------------------

def filter_options(df):
    return {col: sorted(df[col].dropna().astype(str).unique()) for col in FILTER_COLS}


def apply_filters(df, selections):
    """Keep rows whose value is in the selected list for each column.
    A missing or empty selection means "no restriction" for that column."""
    mask = pd.Series(True, index=df.index)
    for col, chosen in selections.items():
        if chosen:
            mask &= df[col].astype(str).isin(chosen)
    return df[mask]


def scored_test(portfolio, scores):
    """Test-period accounts with their scores and the segment columns the filters use."""
    return scores.merge(portfolio[["account_id", *FILTER_COLS, "loan_amount", "annual_income"]],
                        on="account_id", how="inner")


# --- portfolio tab ----------------------------------------------------------

def kpis(df):
    defaulted = df["target_default"] == 1
    exposure = df["loan_amount"].sum()
    return {
        "accounts": len(df),
        "exposure": float(exposure),
        "default_rate": float(df["target_default"].mean()) if len(df) else float("nan"),
        "defaulted_exposure_share": float(df.loc[defaulted, "loan_amount"].sum() / exposure) if exposure else float("nan"),
    }


def segment_profile(df, by):
    if df.empty:
        return pd.DataFrame()
    return profile_by(df, by).reset_index()


def income_employment_grid(df):
    """Default rate and account count per income band x employment type."""
    g = df.groupby(["income_band", "employment_type"], observed=True)["target_default"]
    out = g.agg(n="size", default_rate="mean").reset_index()
    return out


def default_rate_by_year(df):
    return (df.groupby("orig_year")
              .agg(accounts=("account_id", "size"), exposure=("loan_amount", "sum"),
                   default_rate=("target_default", "mean"))
              .reset_index())


# --- model tab --------------------------------------------------------------

def model_metrics_table(scored):
    """AUC / KS / Gini of each model on the given test accounts, with champion/challenger labels.
    Metrics are NaN when the selection is too small or has a single outcome class."""
    ok = len(scored) >= MIN_METRIC_ROWS and scored["target_default"].nunique() == 2
    rows = []
    for key, (name, role, col) in MODEL_ROLES.items():
        m = metrics(scored["target_default"], scored[col]) if ok else {"auc": np.nan, "ks": np.nan, "gini": np.nan}
        rows.append({"Model": name, "Role": role, "AUC": m["auc"], "KS": m["ks"], "Gini": m["gini"]})
    return pd.DataFrame(rows)


def lr_decile_table(scored):
    """Decile 1 = riskiest 10% of the selected accounts by LR score (re-cut within the selection)."""
    if len(scored) < 10:
        return pd.DataFrame()
    table, _ = decile_table(scored["target_default"], scored["score_lr"])
    return table.reset_index()[["decile", "n", "defaults", "default_rate", "lift"]]


def risk_band_table(scored):
    """Observed default rate per LR risk band (bands fixed from training scores)."""
    g = scored.groupby("risk_band", observed=False)["target_default"]
    out = pd.DataFrame({"n": g.size(), "default_rate": g.mean()}).reindex(BAND_LABELS)
    out["n"] = out["n"].fillna(0).astype(int)
    return out.reset_index().rename(columns={"index": "risk_band"})


def shap_top10(summary):
    """LightGBM mean |SHAP| for the full test set, labelled in plain English (largest first)."""
    s = pd.Series(summary["shap_top10"]).sort_values(ascending=False)
    return pd.DataFrame({"feature": [FEATURE_LABELS.get(f, f) for f in s.index], "mean_abs_shap": s.to_numpy()})


# --- account tab ------------------------------------------------------------

def format_value(feature, raw):
    if raw in (None, "missing") or (isinstance(raw, float) and np.isnan(raw)):
        return "not recorded"
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if feature in _FLAGS:
        return "yes" if x else "no"
    if feature in _MONEY:
        return f"{x:,.0f}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def driver_label(feature, raw):
    return f"{FEATURE_LABELS.get(feature, feature)}: {format_value(feature, raw)}"


def account_lookup(account_id, scores, lr_drivers):
    """Risk band, decile and the 5 largest signed LR contributions for one test account.
    The raw probability is deliberately not returned."""
    row = scores.loc[scores["account_id"] == account_id]
    if row.empty:
        raise KeyError(f"{account_id} is not a test-period account")
    row = row.iloc[0]
    d = lr_drivers[lr_drivers["account_id"] == account_id].sort_values("rank")
    drivers = pd.DataFrame({
        "label": [driver_label(f, v) for f, v in zip(d["feature"], d["feature_value"])],
        "contribution": d["contribution"].to_numpy(),
        "direction": d["direction"].to_numpy(),
    })
    return {"account_id": account_id, "risk_band": str(row["risk_band"]),
            "decile": int(row["decile_lr"]), "drivers": drivers}
