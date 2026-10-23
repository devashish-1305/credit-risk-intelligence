# Plan

Scope, stack and workflow are set in CLAUDE.md. This file records the
sub-problem breakdown and the locked design decisions. It is changed only if a
test or the data contradicts it.

## Data profile (raw)
- **Row counts:** customers 10000, loan_accounts 14685, bureau_credit 22979,
  repayment_history 176220, collections_events 8635.
- **Keys:** every PK is unique and no FK points to a missing row.
  - Every customer has 1–3 loans.
  - Every account has exactly 12 repayment months.
  - 1008 customers have no bureau record (thin file).
- **Nulls:**
  - customers: `years_employed`, `num_dependents` and `years_at_residence` have 500 each.
  - loans: `interest_rate_pct` has 294.
  - bureau: `days_past_due` has 689, and `date_closed` is null for every Active line.
- **Labels:**
  - Gender uses "M"/"F" in 600 rows.
  - `disbursal_channel` uses `Partner/DSA` and `action_type` uses `SMS/Email Notice`, which differ from the spelling in the data dictionary.
- **Other data quality issues:**
  - Bureau dates extend to 2025 (opened) and 2029 (closed), past the loan data end of 2023-12.
  - 50% of loans were originated before `customer_since`.
  - 3875 collections events are dated before origination.
  - 510 customers have `years_employed > age − 14`.
- **Default rate:** 16.1% overall, flat by origination year at 15–18%.

Details are in `docs/dq_notes.md`.

## Sub-problems (each needs green pytest before moving on)
| # | Sub-problem → output | Tests |
|---|---|---|
| 1 | **Load + clean** (`src/ingest.py`): typed dates, standardized labels, imputation with `_missing` flags, cap/flag anomalies → `data/processed/*.parquet`, `docs/dq_notes.md` | row counts unchanged; gender ∈ {Male, Female}; imputed cols have no nulls and the flag sums match the raw null counts; PK unique / FK integrity; years_employed capped |
| 2 | **Master table** (1 row per account): customer attrs + bureau aggregates (all lines `bur_*` + point-in-time variant `bur_pit_*`) + repayment months 1–3 aggregates + collections flag/count (analytics only) → `data/processed/master.parquet` | 14685 rows, unique account_id; explicit `bur_missing` thin-file flag preserved through aggregation/imputation; target rate ≈ 0.161; PIT aggregates use no line with date_opened > origination; repayment features use only months ≤ 3 |
| 3 | **Analytics** (`src/analytics.py`): profiles for income band, employment, region, age band, loan type; 2+ cross-tabs; 5 insights (exposure share, defaulted-exposure share, relative default rate, action) → `docs/insights.md`. Notebook deferred — `analytics.py` + the dashboard (sub-problem 5) already cover the same ground. | exposure shares sum to 1; defaulted-exposure shares sum to 1; segment counts sum to 14685; relative default rate of the full portfolio = 1.0 |
| 4 | **Model** (`src/model.py`): time split on origination (train ≤ 2021, test 2022–23), logistic regression + LightGBM (primary = all bureau; variant = PIT bureau, same test set), AUC/KS/Gini, decile table, SHAP, PSI train vs test | no collections features in X; max train date < min test date; Gini = 2·AUC − 1; deciles cover all test rows; imputers fit on train only. Metrics are computed and reported as a diagnostic, not as an AUC pass/fail threshold. Unexpectedly poor performance gets investigated, and the model is never tuned to pass a threshold. |
| 5 | **Dashboard** (`app/app.py`): portfolio view, account view with score + top SHAP drivers, filters | data loaders return the expected shapes; filter function reduces rows correctly; account lookup returns a score in [0, 1] |
| 6 | **Ask the Portfolio** (`src/ask.py`): LLM → pandas expression → AST allow-list validation → safe execution → answer from the result only; query shown; human clicks "Run Query" | allow-list rejects `__import__`/`open`/dunders; a valid query executes on a fixture; the answer prompt contains only the result (LLM mocked) |
| 7 | **Docs**: brief, architecture, assumptions, limitations, monitoring note, 10-slide deck for a Chief Risk Officer | files exist; no banned wording in docs/ |

## Locked decisions
1. **Bureau data:**
   - The primary model uses all bureau lines, treated as a synthetic snapshot. This is a documented assumption.
   - Variant B keeps only lines with `date_opened ≤ origination_date`.
   - Both AUCs are compared on the same test population and the limitations are documented.
   - The comparison does not prove leakage, because filtering also removes bureau data.
2. **Repayment window: months 1–3 only.** This is an early-warning model scored at month 3 after origination, not an application scorecard. The prediction timing is stated in the docs and on the dashboard's account view.
3. **Ask the Portfolio:**
   - The LLM generates a pandas expression, which is validated against an AST allow-list. There is no unrestricted eval and no arbitrary code.
   - It runs only after the user clicks "Run Query".
   - The query is shown to the user, and the answer is grounded only in the execution result.
4. **Imputation scope:** the ingest imputation is for analytics only. Sub-problem 4 refits all imputation on the train data only, inside the model pipeline. Ingest keeps the `_missing` flags, so raw missingness can be recovered.
   - **Bureau DPD:** the master table aggregates the *raw* bureau DPD, with the ingest imputation undone, and keeps a `bur_*_dpd_missing_cnt` count.
     - LightGBM uses these as-is, with NaN plus the missing flags.
     - Logistic regression uses an imputer fit on the train data only.
     - The other customer columns imputed in ingest are reset to NaN where their `_missing` flag = 1 before model fitting, and the same rule applies to them.
5. **Storage:** processed tables are written as parquet (pyarrow).
6. **No collections features in the model** (leakage).
7. **Settled after review of sub-problem 1:**
   - The repayment window stays at months 1–3. The diagnostic is in assumptions A4.
   - Labels stay as spelled in the CSV.
