import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.ingest import PROCESSED_DIR
from src.master import COLLECTIONS_COLS
from src.model import (BAND_QUANTILES, CATEGORICAL, EXCLUDED, TARGET, TRAIN_MAX_YEAR, VALID_YEAR,
                       band_table, prepare, run)

MODELS = ["lr", "lgbm", "lgbm_pit"]
ROW_KEY = ["loan_amount", "annual_income", "age", "loan_to_income"]  # numeric, in every feature set


@pytest.fixture(scope="module")
def out():
    return run(pd.read_parquet(PROCESSED_DIR / "master.parquet"))


def test_no_collections_target_or_id_fields_in_X(out):
    for name, feats in out["features"].items():
        assert not set(feats) & set(EXCLUDED), name
        assert not [f for f in feats if f.startswith("coll_") or f in COLLECTIONS_COLS], name
        assert TARGET not in feats
    assert not [f for f in out["features"]["lgbm_pit"] if f.startswith("bur_") and not f.startswith("bur_pit_")]


def test_time_split_has_no_overlap_in_dates(out):
    assert out["train"]["origination_date"].max() < out["test"]["origination_date"].min()


def test_gini_equals_two_auc_minus_one(out):
    for name, m in out["metrics"].items():
        assert m["gini"] == pytest.approx(2 * m["auc"] - 1, abs=1e-12), name


def test_deciles_cover_every_test_row_once(out):
    deciles = out["test_decile"]
    assert len(deciles) == len(out["test"]) and set(deciles) == set(range(1, 11))
    assert out["deciles"]["n"].sum() == len(out["test"])


def test_lr_imputer_and_scaler_fit_on_train_only(out):
    num_step = out["models"]["lr"].named_steps["pre"].named_transformers_["num"]
    num_cols = [f for f in out["features"]["lr"] if f not in CATEGORICAL]
    expected = out["train"][num_cols].median().to_numpy()
    np.testing.assert_allclose(num_step.named_steps["impute"].statistics_, expected)
    assert num_step.named_steps["scale"].n_samples_seen_ == len(out["train"])


def test_all_models_score_same_test_accounts(out):
    n = len(out["test"])
    assert all(len(out["test_scores"][m]) == n for m in MODELS)
    assert all(np.isfinite(out["test_scores"][m]).all() for m in MODELS)
    assert out["drivers"]["account_id"].nunique() == n


def test_shap_explanations_add_up_to_model_predictions(out):
    raw = out["models"]["lgbm"].predict_proba(out["X_test_lgbm"], raw_score=True)
    recon = out["shap_base"] + out["shap_values"].sum(axis=1)
    np.testing.assert_allclose(recon, raw, atol=1e-4)


def test_lr_contributions_add_up_to_log_odds(out):
    lr, feats = out["models"]["lr"], out["features"]["lr"]
    log_odds = lr.decision_function(out["test"][feats])
    np.testing.assert_allclose(out["lr_contrib"].sum(axis=1) + out["lr_base"], log_odds, atol=1e-8)
    d = out["lr_drivers"]
    assert d["account_id"].nunique() == len(out["test"]) and (d.groupby("account_id").size() == 5).all()
    assert ((d["contribution"] > 0) == (d["direction"] == "raises risk")).all()


def test_risk_bands_learned_on_train_lr_scores(out):
    np.testing.assert_allclose(out["band_thresholds"], np.quantile(out["train_scores"]["lr"], BAND_QUANTILES))
    assert len(out["test_band"]) == len(out["test"]) and not pd.isna(out["test_band"]).any()
    assert band_table(out["train"][TARGET], out["train_band"])["default_rate"].is_monotonic_increasing


def _row_keys(X):
    return set(pd.util.hash_pandas_object(X[ROW_KEY], index=False))


def test_lgbm_early_stopping_and_refit_see_only_train_years(monkeypatch):
    master = pd.read_parquet(PROCESSED_DIR / "master.parquet")
    df = prepare(master)
    year = df["origination_date"].dt.year
    train_keys, test_keys = _row_keys(df[year <= TRAIN_MAX_YEAR]), _row_keys(df[year > TRAIN_MAX_YEAR])
    val_keys = _row_keys(df[year == VALID_YEAR])
    assert not train_keys & test_keys  # the key tells the periods apart

    seen, fit = [], lgb.LGBMClassifier.fit

    def spy(self, X, y, **kwargs):
        seen.append(("fit", _row_keys(X)))
        seen.extend(("eval", _row_keys(Xe)) for Xe, _ in kwargs.get("eval_set") or [])
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy)
    run(master)
    assert {kind for kind, _ in seen} == {"fit", "eval"}
    for kind, keys in seen:
        assert keys <= train_keys and not keys & test_keys
        if kind == "eval":
            assert keys <= val_keys
