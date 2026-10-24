"""Validation figures that support docs/model_report.md and docs/limitations.md.

Reads master.parquet and test_scores.parquet (written by src/model.py) and writes
data/processed/validation.json:
  - bootstrap: paired bootstrap 95% CIs for test-AUC differences
      lr - lgbm          (champion vs challenger)
      lgbm - lgbm_pit    (all bureau lines vs opened-before-origination variant)
  - grouped_retrain: models refit without the train rows of customers who also appear in test
      (limitation L1), scored on the same test set
  - fairness: top-30% flag rate, TPR, FPR and AUC by age band and gender, per model
      (plus the share of each group in the LR Very High band)

Diagnostics only; nothing here changes the models.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ingest import PROCESSED_DIR
from src.model import CATEGORICAL, TARGET, feature_list, fit_lgbm, make_lr, prepare, time_split, to_lgbm_frame

SEED = 42
N_BOOT = 2000
FLAG_SHARE = 0.30  # "flagged" = top 30% of test scores
AGE_BINS = [-np.inf, 25, 35, 45, 55, np.inf]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-55", "56+"]
PAIRS = {"lr_minus_lgbm": ("score_lr", "score_lgbm"), "lgbm_minus_lgbm_pit": ("score_lgbm", "score_lgbm_pit")}


# --- paired bootstrap -------------------------------------------------------

def paired_bootstrap_auc_diff(y, score_a, score_b, n_boot=N_BOOT, seed=SEED):
    """AUC(a) - AUC(b) on the full sample plus a 95% percentile CI from paired resamples
    (the same rows for both models in each resample)."""
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": len(diffs), "seed": seed}


# --- grouped retrain (L1) ---------------------------------------------------

def grouped_retrain(master):
    """Refit LR and LightGBM without train rows of customers who also have a test loan."""
    df = prepare(master)
    train, test = time_split(df)
    shared = set(train["customer_id"]) & set(test["customer_id"])
    in_overlap = test["customer_id"].isin(shared).to_numpy()
    y = test[TARGET].to_numpy()

    def aucs(fit_rows):
        cats = {c: sorted(fit_rows[c].dropna().unique()) for c in CATEGORICAL}
        feats = feature_list("bur_")
        lr = make_lr(feats).fit(fit_rows[feats], fit_rows[TARGET])
        lgbm, n_trees = fit_lgbm(fit_rows, feats, cats)
        scores = {"lr": lr.predict_proba(test[feats])[:, 1],
                  "lgbm": lgbm.predict_proba(to_lgbm_frame(test[feats], cats))[:, 1]}
        return {m: {"auc_all": float(roc_auc_score(y, s)),
                    "auc_overlap": float(roc_auc_score(y[in_overlap], s[in_overlap])),
                    "auc_new": float(roc_auc_score(y[~in_overlap], s[~in_overlap]))}
                for m, s in scores.items()} | {"lgbm_n_trees": n_trees}

    grouped = train[~train["customer_id"].isin(shared)].reset_index(drop=True)
    return {"n_train_full": len(train), "n_train_grouped": len(grouped),
            "n_train_dropped": len(train) - len(grouped), "n_test": len(test),
            "n_test_overlap": int(in_overlap.sum()),
            "full": aucs(train), "grouped": aucs(grouped)}


# --- fairness ---------------------------------------------------------------

def fairness_table(df, score_col, flag_share=FLAG_SHARE):
    """Per group: n, default rate, flag rate (top `flag_share` of scores), TPR, FPR, AUC."""
    flagged = df[score_col] >= df[score_col].quantile(1 - flag_share)
    df = df.assign(flag=flagged, age_band=pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS))
    rows = []
    for col in ["age_band", "gender"]:
        for group, g in df.groupby(col, observed=True):
            y, f = g[TARGET], g["flag"]
            rows.append({
                "attribute": col, "group": str(group), "n": len(g), "default_rate": float(y.mean()),
                "flag_rate": float(f.mean()),
                "tpr": float(f[y == 1].mean()), "fpr": float(f[y == 0].mean()),
                "auc": float(roc_auc_score(y, g[score_col])) if y.nunique() == 2 else None,
            })
            if "risk_band" in g:  # LR champion band (fixed training thresholds)
                rows[-1]["share_very_high_band"] = float((g["risk_band"] == "Very High").mean())
    return rows


# --- pipeline ---------------------------------------------------------------

def run(master, scores):
    test = scores.merge(master[["account_id", "age", "gender"]], on="account_id", how="left", validate="one_to_one")
    y = test["target_default"]
    return {
        "bootstrap": {k: paired_bootstrap_auc_diff(y, test[a], test[b]) for k, (a, b) in PAIRS.items()},
        "grouped_retrain": grouped_retrain(master),
        "fairness": {"flag_share": FLAG_SHARE,
                     "lr": fairness_table(test, "score_lr"), "lgbm": fairness_table(test.drop(columns="risk_band"), "score_lgbm")},
    }


def save(out, out_dir=PROCESSED_DIR):
    (out_dir / "validation.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    out = save(run(pd.read_parquet(PROCESSED_DIR / "master.parquet"),
                   pd.read_parquet(PROCESSED_DIR / "test_scores.parquet")))
    print(json.dumps(out["bootstrap"], indent=2))
