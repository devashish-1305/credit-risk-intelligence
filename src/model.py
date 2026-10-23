"""Month-3 early-warning default model.

Scored 3 months after origination: uses loan + customer attributes, bureau
aggregates and repayment months 1-3 (rep_m3_*). Not an application scorecard.

Models (same train/test accounts for all three):
  lr        - logistic regression baseline, bur_* features
  lgbm      - LightGBM primary, bur_* features (bureau as snapshot, A2)
  lgbm_pit  - LightGBM sensitivity, bur_pit_* instead of bur_*
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ingest import PROCESSED_DIR
from src.master import COLLECTIONS_COLS

TRAIN_MAX_YEAR = 2021
VALID_YEAR = 2021  # in-train holdout used only to pick the LightGBM tree count
TARGET = "target_default"
SEED = 42

# Ingest imputed these for analytics; reset to NaN so imputation is fit on train only.
INGEST_IMPUTED = ["years_employed", "num_dependents", "years_at_residence", "interest_rate_pct"]

LOAN_NUM = ["loan_amount", "interest_rate_pct", "interest_rate_pct_missing", "tenure_months",
            "collateral_flag", "loan_to_income"]
LOAN_CAT = ["loan_type", "loan_purpose", "disbursal_channel"]
CUST_NUM = ["age", "annual_income", "years_employed", "years_employed_missing",
            "years_employed_capped", "num_dependents", "num_dependents_missing",
            "years_at_residence", "years_at_residence_missing"]
# gender and marital_status are deliberately excluded (protected characteristics
# under fair-lending rules); kept for analytics and fairness review only.
CUST_CAT = ["education_level", "employment_type", "occupation_sector", "region",
            "city_tier", "housing_type"]
BUREAU_STEMS = ["n_lines", "n_active", "n_closed", "n_written_off", "total_sanctioned",
                "total_outstanding", "written_off_outstanding", "utilization_active",
                "max_dpd", "mean_dpd", "dpd_missing_cnt", "any_written_off", "missing"]
REPAY = ["rep_m3_n_missed", "rep_m3_n_late", "rep_m3_max_dpd", "rep_m3_mean_dpd",
         "rep_m3_pay_ratio", "rep_m3_balance_ratio", "rep_m3_any_dpd30"]
CATEGORICAL = LOAN_CAT + CUST_CAT

# Never model inputs: IDs, dates, target, collections, data-quality-only flags.
EXCLUDED = (["account_id", "customer_id", "origination_date", "customer_since", "orig_year",
             "orig_before_customer_since", "gender", "marital_status", TARGET] + COLLECTIONS_COLS)


def feature_list(bureau_prefix="bur_"):
    return LOAN_NUM + LOAN_CAT + CUST_NUM + CUST_CAT + [bureau_prefix + s for s in BUREAU_STEMS] + REPAY


def prepare(master):
    df = master.copy()
    for col in INGEST_IMPUTED:
        df.loc[df[f"{col}_missing"] == 1, col] = np.nan
    return df


def time_split(df):
    is_train = df["origination_date"].dt.year <= TRAIN_MAX_YEAR
    return df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)


def customer_overlap(train, test):
    shared = set(train["customer_id"]) & set(test["customer_id"])
    return {
        "customers_in_both": len(shared),
        "test_accounts_with_customer_in_train": int(test["customer_id"].isin(shared).sum()),
        "test_accounts": len(test),
    }


# --- models -----------------------------------------------------------------

def make_lr(features):
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def to_lgbm_frame(X, categories):
    X = X.copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


def make_lgbm(n_estimators=500):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
    )


def fit_lgbm(train, features, categories):
    """Pick the tree count by early stopping on the latest training year, then
    refit on all training years. The test set is never used for this choice."""
    is_val = train["origination_date"].dt.year == VALID_YEAR
    fit, val = train[~is_val], train[is_val]
    probe = make_lgbm().fit(
        to_lgbm_frame(fit[features], categories), fit[TARGET],
        eval_set=[(to_lgbm_frame(val[features], categories), val[TARGET])], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best = int(probe.best_iteration_)
    model = make_lgbm(best).fit(to_lgbm_frame(train[features], categories), train[TARGET])
    return model, best


# --- metrics ----------------------------------------------------------------

def metrics(y, score):
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr))}


def decile_table(y, score):
    """Decile 1 = highest scores (riskiest). Rank first so ties split evenly."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    df["decile"] = pd.qcut(df["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    g = df.groupby("decile", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "min_score": g["score"].min(),
        "max_score": g["score"].max(),
        "defaults": g["y"].sum(),
        "default_rate": g["y"].mean(),
    })
    out["cum_default_capture"] = out["defaults"].cumsum() / out["defaults"].sum()
    out["lift"] = out["default_rate"] / df["y"].mean()
    return out, df["decile"].astype(int).to_numpy()


def psi(expected, actual, bins=10):
    """Population Stability Index, bins = deciles of the expected (train) distribution."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value):
    # Industry heuristics, not universal standards.
    if value < 0.10:
        return "stable"
    return "investigate" if value <= 0.25 else "material drift review"


# --- SHAP -------------------------------------------------------------------

def shap_values(model, X):
    """TreeSHAP in log-odds space; returns (values, base_value)."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap returns [class0, class1]
        values = values[1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    return np.asarray(values), base


def top_drivers(values, X, account_ids, k=5):
    """Long table: per account, the k features with the largest |SHAP|."""
    order = np.argsort(-np.abs(values), axis=1)[:, :k]
    rows = []
    for i, idx in enumerate(order):
        for rank, j in enumerate(idx, start=1):
            rows.append({
                "account_id": account_ids[i], "rank": rank, "feature": X.columns[j],
                "feature_value": str(X.iat[i, j]), "shap_value": float(values[i, j]),
            })
    return pd.DataFrame(rows)


# --- pipeline ---------------------------------------------------------------

def run(master):
    df = prepare(master)
    train, test = time_split(df)
    y_train, y_test = train[TARGET], test[TARGET]
    categories = {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL}

    out = {"train": train, "test": test, "overlap": customer_overlap(train, test),
           "features": {}, "models": {}, "test_scores": {}, "train_scores": {}, "n_trees": {}}

    feats = feature_list("bur_")
    lr = make_lr(feats).fit(train[feats], y_train)
    out["features"]["lr"], out["models"]["lr"] = feats, lr
    out["train_scores"]["lr"] = lr.predict_proba(train[feats])[:, 1]
    out["test_scores"]["lr"] = lr.predict_proba(test[feats])[:, 1]

    for name, prefix in [("lgbm", "bur_"), ("lgbm_pit", "bur_pit_")]:
        feats = feature_list(prefix)
        X_tr, X_te = to_lgbm_frame(train[feats], categories), to_lgbm_frame(test[feats], categories)
        model, out["n_trees"][name] = fit_lgbm(train, feats, categories)
        out["features"][name], out["models"][name] = feats, model
        out["train_scores"][name] = model.predict_proba(X_tr)[:, 1]
        out["test_scores"][name] = model.predict_proba(X_te)[:, 1]
        if name == "lgbm":
            out["X_test_lgbm"] = X_te

    out["metrics"] = {m: metrics(y_test, s) for m, s in out["test_scores"].items()}
    out["deciles"], out["test_decile"] = decile_table(y_test, out["test_scores"]["lgbm"])
    out["psi"] = {m: psi(out["train_scores"][m], out["test_scores"][m]) for m in out["test_scores"]}

    values, base = shap_values(out["models"]["lgbm"], out["X_test_lgbm"])
    out["shap_values"], out["shap_base"] = values, base
    out["shap_importance"] = (
        pd.Series(np.abs(values).mean(axis=0), index=out["X_test_lgbm"].columns)
        .sort_values(ascending=False)
    )
    out["drivers"] = top_drivers(values, out["X_test_lgbm"], test["account_id"].to_numpy())
    return out


def save_artifacts(out, out_dir=PROCESSED_DIR):
    test = out["test"]
    scores = pd.DataFrame({
        "account_id": test["account_id"],
        "origination_date": test["origination_date"],
        "target_default": test[TARGET],
        "score_lgbm": out["test_scores"]["lgbm"],
        "score_lr": out["test_scores"]["lr"],
        "score_lgbm_pit": out["test_scores"]["lgbm_pit"],
        "decile_lgbm": out["test_decile"],
    })
    scores.to_parquet(out_dir / "test_scores.parquet", index=False)
    out["drivers"].to_parquet(out_dir / "test_shap_drivers.parquet", index=False)
    summary = {
        "split": {"train_max_year": TRAIN_MAX_YEAR, "n_train": len(out["train"]), "n_test": len(test),
                  "train_default_rate": float(out["train"][TARGET].mean()),
                  "test_default_rate": float(test[TARGET].mean())},
        "customer_overlap": out["overlap"],
        "metrics": out["metrics"],
        "psi_score": {m: {"psi": v, "label": psi_label(v)} for m, v in out["psi"].items()},
        "n_features": {m: len(f) for m, f in out["features"].items()},
        "lgbm_n_trees": out["n_trees"],
        "shap_top10": out["shap_importance"].head(10).to_dict(),
        "shap_base_value": out["shap_base"],
        "deciles_lgbm": out["deciles"].reset_index().to_dict(orient="records"),
    }
    (out_dir / "model_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    master = pd.read_parquet(PROCESSED_DIR / "master.parquet")
    summary = save_artifacts(run(master))
    print(json.dumps({k: summary[k] for k in ["split", "customer_overlap", "metrics", "psi_score",
                                              "lgbm_n_trees"]}, indent=2))
