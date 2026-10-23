# Assumptions Log

| # | Assumption | Why | Impact if wrong |
|---|---|---|---|
| A1 | The data ends on **2023-12-31**. This is the last loan origination and collections date. | Defines "future" for the bureau and repayment checks | The `future_dated` flag would move |
| A2 | **Bureau is a synthetic snapshot.** The primary model uses all bureau lines. A variant uses only lines with `date_opened ≤ origination_date`. Both are compared on the same test set. | Bureau dates run past the data end, and `account_status` is a current snapshot, not a status as of origination | The primary model may overstate bureau signal. The variant comparison cannot prove leakage, because filtering also removes data.<br>Scale of the removal:<br>- 47% of accounts (6927) have no point-in-time bureau lines, vs 10% (1462) with no lines at all.<br>- Lines per account fall from 2.30 to 0.86 on average.<br>- Status and outstanding balance stay snapshot values even in the variant. |
| A3 | The model is an **early-warning model scored at month 3 after origination**, not an application scorecard. It uses repayment months 1–3 only. | Repayment features would not exist at application time. Months 1–3 limit the overlap with the default outcome. | It cannot be used for origination decisions |
| A4 | `target_default` is a final outcome observed after month 3. No default date is given. | The dictionary gives no default date | If defaults occur within months 1–3, the early-warning framing weakens. **Diagnostic (repayment table):**<br>- Max DPD is 89, so no account ever reaches 90+ DPD. The target is not 90+ DPD in this table.<br>- 17.3% of defaulters have a first Missed payment in months 1–3 (vs 12.1% of non-defaulters), and 35.9% in months 4–12.<br>- 46.7% of defaulters never miss a payment in 12 months.<br>- Conclusion: months 1–3 do not contain most of the defaults. The window is kept, and the true default date remains unverifiable. |
| A5 | Ingest imputation (medians or mode by group) is for analytics only. The model refits imputation on train data. | Avoids test-set information leaking into training | — |
| A6 | `customer_since` is unreliable. 50% of loans predate it, so it is not used as a feature. | Data contradiction | A real relationship-tenure feature is lost |
| A7 | `gender` and `marital_status` are excluded from model inputs (protected characteristics); `age` is kept, as is common in statistically derived models, pending compliance review. | Fair-lending practice | A compliance review may require removing or constraining `age` |

## What a real bank's IT/data team would need to provide
- Bureau pulls **as of the decision date**: point-in-time archives, not the current snapshot.
- A default definition and default date, such as 90+ DPD or a write-off, with its observation window.
- A data-end and refresh schedule for each source system, plus the key mappings across systems.
