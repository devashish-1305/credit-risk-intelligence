# Model Report — Month-3 Early-Warning Default Model

Source: `src/model.py` (`python -m src.model`). Artifacts in `data/processed/`:
- `model_metrics.json`: all figures below
- `test_scores.parquet`: per-account scores for all 3 models, plus the LightGBM decile
- `test_shap_drivers.parquet`: the top-5 SHAP drivers per test account for the primary LightGBM

Every metric here was re-computed from those saved files.

## What the model is (and is not)
- **Scored at month 3 after origination.** It uses loan and customer attributes, bureau aggregates, and repayment behaviour in months 1–3 (`rep_m3_*`).
- **It is an early-warning / servicing-prioritisation model, not an application scorecard.** It cannot be used to approve or decline loans, because the `rep_m3_*` features don't exist at application time.
- **Target:** `target_default`, a final outcome observed after month 3. The data has no default date; see assumption A4.

## Setup
| | Train | Test |
|---|---|---|
| Origination years | 2018–2021 | 2022–2023 |
| Accounts | 9,764 | 4,921 |
| Default rate | 16.2% | 15.9% |

- **Test population:** identical for all three models. All 4,921 accounts from 2022–23 are scored, with no exclusions.
- **Inputs:** 44 per model:
  - loan: 6 numeric + 3 categorical
  - customer: 9 numeric + 6 categorical
  - bureau: 13
  - repayment months 1–3: 7
- **Excluded from the inputs:**
  - IDs, dates and `orig_year`
  - `orig_before_customer_since`, a data-quality flag
  - all `coll_*` collections fields (these are leakage)
  - `gender` and `marital_status`. These are protected characteristics under fair-lending rules; they are kept only for fairness review. Age is kept, which matches common practice in statistically derived models. It should get a compliance review before any real use.
- **Imputation:**
  - The ingest fills for `years_employed`, `num_dependents`, `years_at_residence` and `interest_rate_pct` are reset to NaN using their `_missing` flags.
  - Logistic regression uses a median imputer and scaler fit on train only, plus one-hot categoricals.
  - LightGBM takes NaN natively, using the missing flags and raw bureau days past due.
- **LightGBM settings:**
  - Defaults: learning rate 0.05, 31 leaves, at least 50 rows per leaf, 0.8 row and column subsampling.
  - The **tree count is chosen by early stopping on the 2021 vintage inside the training data**, then the model is refit on all of 2018–21. The test set is never used for this choice. The result is 39 trees for the primary model and 27 for the point-in-time model.

## Results (test, 2022–23)
| Model | Bureau features | AUC | Gini | KS | Score PSI (train → test) |
|---|---|---|---|---|---|
| Logistic regression (baseline) | `bur_*` (all lines) | **0.676** | **0.351** | **0.277** | 0.005 (stable) |
| LightGBM (primary) | `bur_*` (all lines) | 0.673 | 0.345 | 0.262 | 0.004 (stable) |
| LightGBM (sensitivity) | `bur_pit_*` (point-in-time) | 0.657 | 0.315 | 0.234 | 0.003 (stable) |

**Interpretation:**
- **Discrimination is moderate.** An AUC around 0.67 is typical of a behavioural model on weak-to-moderate signals. These numbers are reported as a diagnostic, not checked against a pass mark.
- **LightGBM does not beat logistic regression; the two are within 0.003 AUC.** The signal here is mostly additive and monotonic: income, employment tenure, bureau days past due, and early missed payments. In production, the simpler and more explainable model would be the default choice. LightGBM stays as the primary model per the plan because its per-account SHAP drivers feed the dashboard.
- **LightGBM's first run overfit badly.** With 300 trees, the training AUC was 0.999 and the test AUC 0.633. This was diagnosed on the 2021 in-train holdout and fixed only by early-stopping the tree count. No other tuning was done.
- **Point-in-time vs snapshot bureau (A2):** the point-in-time model scores 0.016 AUC lower.
  - This does **not** show that the snapshot model leaks. Filtering also removes real data: 23.7% of test accounts have no point-in-time bureau lines, vs 10.2% with no bureau lines at all.
  - The status and balance fields remain snapshot values even in the point-in-time variant.
  - The honest reading is that snapshot bureau adds at most around 0.016 AUC. Some or all of that may come from information that wouldn't be available at month 3.

## Decile table (LightGBM primary, test)
Decile 1 is the highest score.

| Decile | n | Score range | Defaults | Default rate | Cumulative default capture | Lift |
|---|---|---|---|---|---|---|
| 1 | 493 | 0.263–0.579 | 185 | 37.5% | 23.7% | 2.37 |
| 2 | 492 | 0.208–0.263 | 117 | 23.8% | 38.7% | 1.50 |
| 3 | 492 | 0.178–0.208 | 99 | 20.1% | 51.4% | 1.27 |
| 4 | 492 | 0.158–0.178 | 75 | 15.2% | 61.0% | 0.96 |
| 5 | 492 | 0.142–0.158 | 70 | 14.2% | 70.0% | 0.90 |
| 6 | 492 | 0.128–0.142 | 61 | 12.4% | 77.8% | 0.78 |
| 7 | 492 | 0.116–0.128 | 54 | 11.0% | 84.7% | 0.69 |
| 8 | 492 | 0.105–0.116 | 43 | 8.7% | 90.3% | 0.55 |
| 9 | 492 | 0.094–0.105 | 43 | 8.7% | 95.8% | 0.55 |
| 10 | 492 | 0.065–0.094 | 33 | 6.7% | 100.0% | 0.42 |

- The default rate falls with each decile; deciles 8 and 9 are tied at 8.7%.
- **The top 30% of accounts by score holds 51% of the defaults.** That makes it a practical first cut for month-3 servicing outreach.

## SHAP — top 10 features
Mean |SHAP| in log-odds, LightGBM primary, test set:

| # | Feature | Mean \|SHAP\| | Group |
|---|---|---|---|
| 1 | annual_income | 0.142 | customer |
| 2 | years_employed | 0.113 | customer |
| 3 | bur_mean_dpd | 0.109 | bureau |
| 4 | age | 0.102 | customer |
| 5 | loan_to_income | 0.096 | loan |
| 6 | rep_m3_max_dpd | 0.057 | repayment months 1–3 |
| 7 | interest_rate_pct | 0.054 | loan |
| 8 | bur_max_dpd | 0.041 | bureau |
| 9 | bur_utilization_active | 0.033 | bureau |
| 10 | occupation_sector | 0.025 | customer |

- **The per-account drivers match the model's own predictions.** The base value plus each account's SHAP values reproduces the model's log-odds output (checked in `tests/test_model.py`).
- **Format:** the top-5 drivers per account, ranked by |SHAP|, are saved in long format: `account_id`, `rank`, `feature`, `feature_value`, `shap_value`.
- Employment type isn't in the top 10 even though Unemployed shows a 2.2× relative risk in the analytics. Its effect largely overlaps with income and `years_employed`; Unemployed customers have a median tenure of 0.

## Caveats
1. **Data timing.**
   - `rep_m3_*` is known at month 3, the scoring point.
   - Bureau features are a snapshot that may include information from after origination (A2).
   - Repayment months for loans originated in 2023 run past the 2023-12-31 data end. This is a synthetic artifact; see the data-quality notes.
2. **Customer overlap (limitation L1).**
   - 1,881 customers have loans in both train and test, so 2,070 of the 4,921 test accounts (42%) belong to a customer the model saw in training, with identical `bur_*` values.
   - The worry was that this inflates AUC. The data shows the opposite: AUC on those accounts is **0.656**, vs **0.684** for customers new to the test set (LightGBM primary). Logistic regression and the point-in-time model show the same pattern.
   - The overlap doesn't appear to inflate test AUC here, but the limitation stays on record, and a grouped split remains deferred.
3. **No loss-severity view.** The model ranks default probability only. The data has no recovery or LGD field, which matters most for collateralised Mortgage and Auto loans.
4. **Calibration not assessed.** Scores are used for ranking (deciles, AUC). Using them as probabilities would need a calibration check first.

## Monitoring plan
- **Score drift (PSI):**
  - Each month, compute the score PSI against the training baseline.
  - Compute a characteristic stability index (CSI) for the top-10 SHAP features and for every bureau and `rep_m3_*` input.
  - Thresholds are common heuristics, not universal standards. Below 0.10 is generally stable; 0.10–0.25 should be investigated by checking input pipelines and the segment mix; above 0.25 triggers a material-drift review with the model owner.
  - The current train → test score PSI is 0.004–0.005 for logistic regression and LightGBM, so drift is low.
- **Outcome monitoring:** quarterly, once outcomes mature, track AUC and KS on newly matured vintages, plus the observed default rate by score decile.
- **Retraining cadence:** an annual scheduled refit on a rolling window of the most recent vintages. Retrain early if PSI or CSI goes above 0.25, or if AUC falls more than about 0.03 below the 0.67 benchmark on a matured vintage.
- **Before deployment:** each retrain repeats the time-based holdout and the point-in-time comparison, and gets a fairness and compliance review of the inputs.
