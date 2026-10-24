import numpy as np
import pandas as pd
import pytest

from src import run_all
from src.validation import AGE_LABELS, fairness_table, paired_bootstrap_auc_diff


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    a = y * 0.6 + rng.normal(0, 0.5, 600)   # stronger model
    b = y * 0.3 + rng.normal(0, 0.5, 600)   # weaker model
    return y, a, b


def test_bootstrap_ci_exists_and_brackets_the_point_estimate(sample):
    out = paired_bootstrap_auc_diff(*sample, n_boot=300)
    assert out["ci_low"] < out["diff"] < out["ci_high"]
    assert out["ci_low"] > 0 and out["n_boot"] == 300  # a clearly stronger model gives a CI above zero


def test_bootstrap_is_reproducible_with_the_seed(sample):
    first = paired_bootstrap_auc_diff(*sample, n_boot=200, seed=7)
    assert paired_bootstrap_auc_diff(*sample, n_boot=200, seed=7) == first
    assert paired_bootstrap_auc_diff(*sample, n_boot=200, seed=8) != first


def test_fairness_table_covers_every_test_row(sample):
    y, a, _ = sample
    df = pd.DataFrame({"target_default": y, "score_lr": a,
                       "age": np.resize([22, 30, 40, 50, 60], len(y)), "gender": np.resize(["Female", "Male"], len(y))})
    rows = fairness_table(df, "score_lr")
    ages = [r for r in rows if r["attribute"] == "age_band"]
    assert [r["group"] for r in ages] == AGE_LABELS and sum(r["n"] for r in ages) == len(df)
    overall_flag = sum(r["flag_rate"] * r["n"] for r in ages) / len(df)
    assert overall_flag == pytest.approx(0.30, abs=0.01)


def test_run_all_reports_missing_raw_files(tmp_path):
    (tmp_path / "customers.csv").write_text("x\n")
    assert run_all.missing_raw(tmp_path) == [f"{n}.csv" for n in run_all.RAW_FILES[1:]]
