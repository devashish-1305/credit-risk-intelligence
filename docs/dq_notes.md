# Data Quality Notes

Produced by `src/ingest.py`, which writes `data/processed/*.parquet`. Run it with `python -m src.ingest`.

**Principles:**
- No rows are dropped.
- Every imputed column keeps a `<col>_missing` flag.
- The imputed values here are for **analytics only**. The model pipeline refits imputation on train data only.

## Keys and join logic
| Table | Grain | Primary key | Foreign key → parent | Raw integrity |
|---|---|---|---|---|
| customers | customer | customer_id | — | unique |
| loan_accounts | account | account_id | customer_id → customers | unique; 0 orphans; every customer has 1–3 loans |
| bureau_credit | external credit line | bureau_id | customer_id → customers | unique; 0 orphans; 1008 customers have no lines (thin file) |
| repayment_history | account × month | (account_id, payment_month) | account_id → loan_accounts | unique; 0 orphans; every account has exactly months 1–12 |
| collections_events | collections action | collection_event_id | account_id → loan_accounts | unique; 0 orphans; 4278 accounts have events |

The master table (sub-problem 2) has one row per account.
- **Base table:** `loan_accounts`, left-joined to `customers` on customer_id.
- **Bureau:** aggregated to customer level, then left-joined on customer_id. A customer with no bureau lines keeps an explicit thin-file flag.
- **Repayment:** months 1–3 are aggregated to account level and left-joined on account_id.
- **Collections:** aggregated to account level for analytics only, and never used as model input.

## Issues and handling
| # | Issue | Count | Handling | Rationale |
|---|---|---|---|---|
| 1 | `gender` coded "M"/"F" | 358 / 242 | Mapped to Male/Female | Same category, inconsistent coding |
| 2 | `years_employed` null | 500 | `_missing` flag; median within `employment_type` (Unemployed 0.0, others 4.0–4.5) | Tenure differs by employment type |
| 3 | `years_employed > age − 14` | 510 | Capped at `age − 14`, `years_employed_capped` flag | Impossible value; capping keeps the row |
| 4 | `num_dependents` null | 500 | `_missing` flag; mode (1) | Discrete count |
| 5 | `years_at_residence` null | 500 | `_missing` flag; median (2.8) | Skewed continuous variable |
| 6 | `interest_rate_pct` null | 294 | `_missing` flag; median within `loan_type` | Rates differ strongly by product (Mortgage around 6.5%, Card around 19.5%). Missing rows default at 17.3% vs 16.1%, so the flag is kept. |
| 7 | Bureau `days_past_due` null | 689 | `_missing` flag; median within `account_status` (Active 20, Closed 0, Written-Off 22) | DPD depends on status |
| 8 | Bureau `date_closed` null | 11243 | Left null | Exactly the Active lines; null means "open" |
| 9 | Bureau lines dated after the data end (2023-12-31): opened up to 2025-12, closed up to 2029-12 | 8892 lines | `future_dated` flag | Point-in-time risk. In 62.5% of customer–loan pairs, the line opened after the loan's origination. See the assumptions log (bureau treated as a snapshot, plus an opened-before-origination sensitivity variant). |
| 10 | Loan `origination_date` < customer `customer_since` | 7293 loans | Logged; `customer_since` is excluded from features and flagged at loan level in sub-problem 2 | Relationship start date is unreliable |
| 11 | Collections `event_date` < account origination | 3875 events | `pre_origination` flag; rows kept | Synthetic artifact. Collections are used only for analytics and never in the model. |
| 12 | Repayment months 1–12 run past the data end for loans originated in 2023 | — | Logged only | Synthetic artifact. The model uses months 1–3 only. |
| 13 | Label spellings differ from the data dictionary: `Partner/DSA` (dictionary says Partner-DSA), `SMS/Email Notice` (dictionary says SMS-Email) | all rows of those labels | Kept as in the CSV | The CSV is the source of truth; values are internally consistent |

## Consistency checks that passed (no action needed)
- `collateral_flag` = 1 exactly for Auto Loan and Mortgage. `clean_loans` asserts this.
- Repayment: `emi_paid ≤ emi_due`.
  - Missed ⇒ `emi_paid` = 0 and DPD ≥ 30.
  - Late ⇒ DPD 1–28.
  - On-Time ⇒ DPD 0.
- Bureau:
  - `outstanding ≤ sanctioned`.
  - `date_closed ≥ date_opened`.
  - `date_closed` is null if and only if status is Active.
- No duplicate rows in any table. Every date column parses.
- Default rate is 16.1% and flat by origination year (15–18%), so a time-based split is not distorted by target drift.
