# Model Report — Month-3 Early-Warning Default Model

Source: `src/model.py` (`python -m src.model`). Artifacts in `data/processed/`:
- `model_metrics.json`: all pipeline figures below
- `test_scores.parquet`: per-account scores for all 3 models, LR and LightGBM deciles, LR risk band
- `test_lr_drivers.parquet`: the top-5 signed LR contributions per test account (champion)
- `test_shap_drivers.parquet`: the top-5 SHAP drivers per test account (LightGBM challenger)

Every metric here was re-computed from those saved files. The bootstrap CIs, the grouped retrain and the fairness table come from `src/validation.py` (`python -m src.validation`, fixed seed 42), which writes `data/processed/validation.json`. Moving them into the pipeline changed two figures; both are marked *(corrected)* below.

## What the model is (and is not)
- **Scored at month 3 after origination.** It uses loan and customer attributes, bureau aggregates, and repayment behaviour in months 1–3 (`rep_m3_*`).
- **It is an early-warning / servicing-prioritisation model, not an application scorecard.** It cannot be used to approve or decline loans, because the `rep_m3_*` features don't exist at application time.
- **Prediction task the data supports:** the *eventual* `target_default` flag, given information available at month 3. The data has no default date and no outcome horizon, so it cannot establish that the features precede the default (assumption A4).
- **Bureau features are snapshot values** (A2). In 62.5% of customer–account pairs the bureau line was opened after the loan's origination, so the reported AUCs are optimistic for a real month-3 decision.

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
  - `gender` and `marital_status`. Age is kept. See the fairness section.
- **Imputation:**
  - The ingest fills for `years_employed`, `num_dependents`, `years_at_residence` and `interest_rate_pct` are reset to NaN using their `_missing` flags.
  - Logistic regression uses a median imputer and scaler fit on train only, plus one-hot categoricals.
  - LightGBM takes NaN natively, using the missing flags and raw bureau days past due.
- **LightGBM settings:**
  - Defaults: learning rate 0.05, 31 leaves, at least 50 rows per leaf, 0.8 row and column subsampling.
  - The **tree count is chosen by early stopping on the 2021 vintage inside the training data**, then the model is refit on all of 2018–21. The test set is never used for this choice (tested: every LightGBM fit and eval set contains only rows from ≤ 2021, and eval rows only from 2021). The result is 39 trees for the challenger and 27 for the sensitivity variant.

## Results (test, 2022–23)
| Role | Model | Bureau features | AUC | Gini | KS | Brier |
|---|---|---|---|---|---|---|
| **Champion** | Logistic regression | `bur_*` (snapshot, all lines) | **0.675** | **0.351** | **0.277** | 0.125 |
| Challenger | LightGBM | `bur_*` (snapshot, all lines) | 0.673 | 0.345 | 0.262 | 0.125 |
| Sensitivity | LightGBM, opened-before-origination variant | `bur_pit_*` | 0.657 | 0.315 | 0.234 | 0.127 |

**Champion / challenger:**
- **The LR vs LightGBM difference is inconclusive.** The paired bootstrap 95% CI for AUC(LR) − AUC(LightGBM) is **−0.008 to +0.014** (`validation.py`, 2,000 resamples, seed 42; *corrected* from +0.015 in the earlier one-off check). Neither model is shown to be better.
- **LR is the champion:** it performs on par with LightGBM and is simpler to govern, with fixed coefficients, monotone effects and an exact per-account decomposition. **LightGBM is the challenger**, kept for comparison and for its SHAP explanations.
- **Discrimination is moderate.** An AUC around 0.67 is typical of a behavioural model on weak-to-moderate signals. These numbers are a diagnostic, not checked against a pass mark.
- **LightGBM's first run overfit badly.** With 300 trees, the training AUC was 0.999 and the test AUC 0.633. This was diagnosed on the 2021 in-train holdout and fixed only by early-stopping the tree count. No other tuning was done. Even at 39 trees, the in-sample train AUC is 0.81 against 0.67 on test.

**Opened-before-origination sensitivity variant (A2):**
- This variant keeps only bureau lines with `date_opened ≤ origination_date`. It is **not** point-in-time: `account_status`, `outstanding_amount` and DPD remain snapshot values. For example, 2,453 kept lines are marked Closed but closed after origination, and 1,923 are Written-Off with no write-off date.
- It scores 0.015 AUC below the challenger (bootstrap 95% CI 0.005 to 0.026, `validation.py`; *corrected* from 0.016, which was a difference of rounded AUCs: 0.6726 − 0.6573 = 0.0152).
- **This is a sensitivity result, not a lower bound.** The filter removes real history (23.7% of test accounts have no kept lines, vs 10.2% with no bureau lines at all), and the kept lines still carry snapshot information. Neither AUC is the true month-3 figure.

## Decile table (LR champion, test)
Decile 1 is the highest score.

| Decile | n | Score range | Mean score | Defaults | Default rate | Cumulative default capture | Lift |
|---|---|---|---|---|---|---|---|
| 1 | 493 | 0.284–0.911 | 0.377 | 175 | 35.5% | 22.4% | 2.24 |
| 2 | 492 | 0.223–0.284 | 0.250 | 130 | 26.4% | 39.1% | 1.67 |
| 3 | 492 | 0.185–0.222 | 0.202 | 100 | 20.3% | 51.9% | 1.28 |
| 4 | 492 | 0.160–0.185 | 0.172 | 85 | 17.3% | 62.8% | 1.09 |
| 5 | 492 | 0.141–0.160 | 0.150 | 66 | 13.4% | 71.3% | 0.85 |
| 6 | 492 | 0.125–0.141 | 0.133 | 55 | 11.2% | 78.3% | 0.71 |
| 7 | 492 | 0.107–0.125 | 0.116 | 53 | 10.8% | 85.1% | 0.68 |
| 8 | 492 | 0.090–0.107 | 0.098 | 47 | 9.6% | 91.2% | 0.60 |
| 9 | 492 | 0.071–0.089 | 0.081 | 31 | 6.3% | 95.1% | 0.40 |
| 10 | 492 | 0.018–0.071 | 0.056 | 38 | 7.7% | 100.0% | 0.49 |

- The default rate falls through decile 9; decile 10 (7.7%) is slightly above decile 9 (6.3%).
- **The top 30% of accounts by score holds 52% of the defaults** (LightGBM: 51%). That makes it a practical first cut for month-3 servicing outreach.
- The LightGBM challenger's decile table is in `model_metrics.json` (`deciles_lgbm`).

## Risk bands (LR champion)
The dashboard will show **risk bands, not probabilities**. The cut points are the 40th, 70th and 90th percentiles of the **training-period** LR scores (0.123, 0.184, 0.281). They were fixed on train and applied unchanged to the test accounts.

| Band | Train share | Train default rate | Test share | Test default rate | Share of defaulted exposure (test) |
|---|---|---|---|---|---|
| Low | 40.0% | 9.1% | 39.4% | 8.6% | 11.3% |
| Medium | 30.0% | 14.2% | 30.4% | 14.0% | 16.6% |
| High | 20.0% | 22.8% | 19.7% | 23.4% | 28.6% |
| Very High | 10.0% | 37.6% | 10.4% | 34.7% | 43.4% |

High + Very High: 30.1% of test accounts, 51.9% of defaults, 72.1% of defaulted exposure (sum of loan_amount for defaults). Part of the dollar concentration is a loan-size effect: the top loan_amount quartile both scores higher (mean LR score 0.193 vs about 0.15 for Q1–Q3) and defaults more (19.3% vs 13.3–15.7%).

## Calibration
**No formal calibration was done** (no Platt or isotonic step, and no calibration test). Scores are used for ranking and banding. Diagnostic only, on test:

- **Brier score:** LR 0.1249, LightGBM 0.1254.
- **Mean predicted vs observed default rate:** LR 16.3% vs 15.9%; LightGBM 16.3% vs 15.9%.

| Decile (1 = highest) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| LR predicted | 37.7% | 25.0% | 20.2% | 17.2% | 15.0% | 13.3% | 11.6% | 9.8% | 8.1% | 5.6% |
| LR observed | 35.5% | 26.4% | 20.3% | 17.3% | 13.4% | 11.2% | 10.8% | 9.6% | 6.3% | 7.7% |
| LightGBM predicted | 33.2% | 23.2% | 19.2% | 16.8% | 15.0% | 13.5% | 12.2% | 11.1% | 9.9% | 8.5% |
| LightGBM observed | 37.5% | 23.8% | 20.1% | 15.2% | 14.2% | 12.4% | 11.0% | 8.7% | 8.7% | 6.7% |

- LR tracks the observed rates within about 2 points in every decile.
- LightGBM is compressed: it under-predicts the top decile (33% vs 38%) and over-predicts the bottom (8.5% vs 6.7%).
- Before any score is shown or used as a probability, a calibration check on a matured out-of-time vintage is needed.

## Explanations
**LR champion: signed contributions.**
- Each input's contribution is its coefficient × (scaled value − training mean), in log-odds. One-hot categories are summed back to the source feature.
- The contributions plus a base of −1.777 equal the LR log-odds exactly (tested). The base is the intercept plus the training-mean offset.
- Contributions are measured against the average training account. The intercept alone is not used as the base, because the uncentred one-hot terms would then carry every category's full coefficient.
- **Format:** the top-5 contributions per account, by |contribution|, are saved in long format: `account_id`, `rank`, `feature`, `feature_value`, `contribution`, `direction` (raises / lowers risk). Missing inputs show as `missing`; LR used the train median for them.
- **Most common #1 driver:** `annual_income` (1,122 accounts), `bur_mean_dpd` (970), `loan_amount` (650), `age` (558), `years_employed` (392).
- **Caveat:** LR is linear in raw amounts, so extreme values give large contributions that can partly offset each other. One example is a large mortgage, with `loan_amount` +1.45 and `tenure_months` −0.30.

**LightGBM challenger: SHAP.** Mean |SHAP| in log-odds, test set:

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

- The base value plus each account's SHAP values reproduces the model's log-odds output (tested).
- Employment type isn't in the top 10 even though Unemployed shows a 2.2× relative risk in the analytics. Its effect largely overlaps with income and `years_employed`; Unemployed customers have a median tenure of 0.

## Fairness (open)
- **Excluding `gender` and `marital_status` from the inputs does not establish fairness.** Other inputs can act as proxies, and outcomes can still differ across groups.
- **No claim of regulatory compliance is made.** These are diagnostic figures only.

LightGBM challenger (`validation.py`; the LR table is in `validation.json`). "Flagged" means scored in the top 30% of test scores. The oldest band is ages 56 and over; it was labelled "55+" before, which overlapped with 46–55 (same 271 accounts, label corrected):

| Group | n | Default rate | Flag rate | True-positive rate | False-positive rate | AUC |
|---|---|---|---|---|---|---|
| Age ≤25 | 634 | 23.0% | 56.3% | 71.2% | 51.8% | 0.644 |
| Age 26–35 | 1,369 | 18.2% | 41.3% | 61.4% | 36.8% | 0.680 |
| Age 36–45 | 1,747 | 13.9% | 25.6% | 46.3% | 22.3% | 0.663 |
| Age 46–55 | 900 | 12.9% | 10.1% | 25.0% | 7.9% | 0.645 |
| Age 56+ | 271 | 10.0% | 6.3% | 11.1% | 5.7% | **0.557** |
| Female | 2,135 | 15.8% | 29.2% | 50.1% | 25.3% | 0.664 |
| Male | 2,786 | 15.9% | 30.7% | 52.4% | 26.5% | 0.679 |

- **Age:** the flag rate is 56% for ≤25 against 6% for 56+, and the false-positive rate 52% against 6%. Part of the gap reflects real differences in default rate (23% against 10%). Ranking is weak for 56+ (AUC 0.557, n=271).
- **The LR champion shows the same pattern:** flag rate 58% against 5%, and 56+ AUC 0.582. In risk bands, 24% of ≤25 accounts land in Very High against 1.5% of 56+.
- **Gender:** flag and error rates are close (false-positive rate 25% Female, 27% Male).
- `age` is the #4 SHAP feature and ranks 12th of 85 LR coefficients by size.

**Open questions:**
1. Is age allowed as an input under the target jurisdiction's credit rules, and is the "statistically derived model" exemption documented?
2. Should a model without age be compared on AUC and on disparity across age bands?
3. Is `years_employed` capped at age − 14 acting as an age proxy? Its correlation with age is only 0.08, but the capped flag applies to 736 accounts.
4. Can the model be relied on for customers 56+, given AUC 0.557 and n=271?
5. Which fairness metric governs: flag rate, false-positive rate, or calibration within each group?
6. Whether to use the protected attributes in the fairness review only, which they currently are.

## Caveats
1. **Data timing.**
   - `rep_m3_*` is known at month 3, the scoring point.
   - Bureau features are snapshot values that may include information from after origination, and after month 3 (A2).
   - The target has no event date or horizon (A4).
   - Repayment months for loans originated in 2023 run past the 2023-12-31 data end. This is a synthetic artifact; see the data-quality notes.
2. **Customer overlap (limitation L1). Effect on test AUC is inconclusive.**
   - 1,881 customers have loans in both train and test, so 2,070 of the 4,921 test accounts (42%) belong to a customer the model saw in training, with identical `bur_*` values.
   - On those accounts, AUC is 0.656 vs 0.684 for customers new to the test set (LightGBM). These are different populations, so this comparison cannot show whether the overlap inflates AUC.
   - **Grouped retrain (`validation.py`):** we dropped the 2,255 train rows of shared customers (9,764 → 7,509) and scored the same test set. Test AUC moved from 0.675 to 0.672 (LR) and from 0.673 to 0.658 (LightGBM). On the overlap accounts, it moved from 0.661 to 0.663 (LR) and from 0.656 to 0.650 (LightGBM).
   - **Why this is inconclusive:** the retrain also loses 23% of the training data, so a drop can't be separated from the smaller train set.
   - Default rate is 23.3% when a customer's other loan defaulted vs 14.6% otherwise, so customer-level signal does carry across loans.
3. **No loss-severity view.** The model ranks default probability only. The data has no recovery or LGD field, which matters most for collateralised Mortgage and Auto loans.
4. **Calibration not formally assessed.** See the Calibration section.

## Monitoring plan
- **Score drift (PSI):** each month, compute the score PSI against an **out-of-sample baseline**.
  - Out-of-sample baseline (≤2020 model scoring the 2021 vintage, then the 2022–23 test accounts): **LR 0.006, LightGBM 0.011, sensitivity variant 0.008**. All stable.
  - In-sample (train scores vs test, kept for reference; train scores are in-sample, so LightGBM's are overfit): LR 0.005, LightGBM 0.004, sensitivity variant 0.003.
  - LightGBM out-of-sample PSI baseline is approximate — category lists come from ≤2021 rows and tree count was early-stopped on 2021; LR champion baseline is fully ≤2020. Diagnostic only.
- **Input drift (CSI)** for the top-10 SHAP features, train vs test: 0.0001–0.004, all stable. Highest: `bur_utilization_active` 0.004 and `occupation_sector` 0.004.
  - Planned: extend CSI to every bureau and `rep_m3_*` input.
- **Thresholds:** these are common heuristics, not universal standards. Below 0.10 is generally stable; 0.10–0.25 should be investigated by checking input pipelines and the segment mix; above 0.25 triggers a material-drift review with the model owner.
- **Outcome monitoring:** quarterly, once outcomes mature, track AUC and KS on newly matured vintages, the observed default rate by score decile and risk band, and predicted vs observed rates (calibration).
- **Retraining cadence:** an annual scheduled refit on a rolling window of the most recent vintages. Retrain early if PSI or CSI goes above 0.25, or if AUC falls more than about 0.03 below the 0.67 benchmark on a matured vintage.
- **Before deployment:** each retrain repeats the time-based holdout, the champion/challenger comparison and the opened-before-origination sensitivity check, and gets a fairness review of the inputs.
