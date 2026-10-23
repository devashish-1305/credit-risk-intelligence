# Portfolio Insights

Source: `src/analytics.py` (`python -m src.analytics`), run on `data/processed/master.parquet`
(14,685 accounts). Exposure = `loan_amount` (the original amount at origination).
Relative risk = segment default rate ÷ portfolio default rate (16.1%).
Share of defaulted exposure = segment's defaulted `loan_amount` ÷ total defaulted
`loan_amount` across the portfolio — this is where loss dollars are concentrated,
which is a different question from where the default *rate* is highest.

Collections figures are descriptive only — collections data is never used as a model
feature (it is post-origination outcome information; see PLAN.md locked decisions).

## 1. Unemployed borrowers are a small slice of volume but carry outsized risk
- **Segment:** `employment_type` = Unemployed
- **% of accounts:** 4.9% (718)
- **% of exposure:** 4.9% ($29.6M)
- **Default rate:** 35.0%
- **Relative risk:** 2.17× portfolio
- **Share of defaulted exposure:** 10.8%
- **Action:** Tighten underwriting (lower approval rate or require collateral/co-signer) for Unemployed applicants, or price risk-adjusted rates. Low exposure share (4.9%) vs. a disproportionate 10.8% share of defaulted dollars makes this a contained, high-leverage fix.

## 2. Mortgage is a small share of accounts but over half of total exposure, at the highest default rate — and the clear concentration of loss dollars
- **Segment:** `loan_type` = Mortgage
- **% of accounts:** 9.8% (1,432)
- **% of exposure:** 52.3% ($314.5M)
- **Default rate:** 24.7%
- **Relative risk:** 1.53× portfolio
- **Share of defaulted exposure:** 65.0% — i.e., roughly two-thirds of all defaulted dollars in the portfolio sit in Mortgage, far more concentrated than its 52.3% exposure share alone suggests.
- **Important caveat (severity, not just default rate):** Mortgage is a collateralized loan type (`collateral_flag` = 1). A defaulted mortgage typically still recovers value through the collateral, so loss-given-default (LGD) is likely lower than for unsecured products — this data has no recovery/LGD field, so the comparison above is in default *rate*, not realized *loss*. A high default rate does not automatically mean a high loss rate.
- **Action:** Given both the caveat above and the scale of exposure involved, the right next step is a **Mortgage deep dive** — vintage analysis (does risk cluster in specific origination periods?), LTV/collateral-value data if obtainable, and borrower profile within Mortgage — before any underwriting policy change. This segment is too large and too structurally different (secured, long tenor) to act on from the default-rate number alone.

## 3. Low-income borrowers (<50k) default at nearly 3.6x the rate of top earners
- **Segment:** `income_band` = <50k
- **% of accounts:** 12.5% (1,840)
- **% of exposure:** 12.7% ($76.6M)
- **Default rate:** 27.9% (vs. 7.8% for 120k+)
- **Relative risk:** 1.74× portfolio
- **Share of defaulted exposure:** 21.3%
- **Action:** Introduce income-tiered pricing or lower approval caps for <50k applicants. The effect compounds with employment type — see the detail below, both cells are large enough to trust:
  - <50k × Contract: **261 accounts**, default rate **29.5%**
  - <50k × Unemployed: **458 accounts**, default rate **40.2%**
  - Neither cell is flagged as a small sample (the threshold used is n ≥ 30); the smallest cell anywhere in the income×employment grid is 120k+ × Unemployed at n=1, which *is* flagged low-confidence and excluded from any claim.

## 4. Young borrowers (under 30) default well above the portfolio average
- **Segment:** `age_band` = <30
- **% of accounts:** 22.1% (3,250)
- **% of exposure:** 23.7% ($142.7M)
- **Default rate:** 21.6%
- **Relative risk:** 1.34× portfolio
- **Share of defaulted exposure:** 30.7%
- **Action:** This segment is large (nearly a quarter of the book), so broad-brush tightening would cut volume substantially. Favor a targeted scorecard adjustment (e.g., weight bureau/employment tenure more heavily for this age band) over an across-the-board policy change.

## 5. Month-3 missed payments are an early-warning association, not a cause of default
- **Segment:** `rep_m3_n_missed` (number of Missed payments in months 1–3), bucketed 0 / 1 / 2+
- **% of accounts:** 0 missed = 87.0% (12,780); 1 missed = 12.2% (1,788); 2+ missed = 0.8% (117)
- **% of exposure:** 0 = 84.6%; 1 = 14.2%; 2+ = 1.2%
- **Default rate:** 0 = 15.3%; 1 = 20.7%; 2+ = 33.3%
- **Relative risk:** 0 = 0.95×; 1 = 1.29×; 2+ = 2.07× portfolio
- **Share of defaulted exposure:** 0 = 80.3%; 1 = 17.3%; 2+ = 2.4%
- **Framing — association, not causation:** a missed payment in months 1–3 and eventual default likely share a common underlying cause (e.g., an income shock or employment change), rather than the miss itself causing the default. This is the basis for the early-warning model in PLAN.md (scored at month 3), and the same reasoning rules it out as a basis for individual account-level intervention decisions without further review.
- **Action:** Route accounts with ≥1 missed payment by month 3 to earlier servicing outreach — the 2+ group is only 0.8% of accounts but has more than double the portfolio's default rate and a 2.4% share of defaulted exposure, a disproportionate concentration of loss for such a small group. Outreach timing: month 3, consistent with when this feature is observed.

## Footnote: Mortgage + collections activity (descriptive only)
Within Mortgage specifically, 35.8% of accounts (513 of 1,432) have at least one
collections event. Comparing default rates:
- **Vs. Mortgage accounts without collections:** 29.2% (with collections) vs. 22.1%
  (without collections) — a 7.1pp gap within the same loan type.
- **Vs. the overall portfolio baseline (16.1%):** Mortgage-with-collections accounts
  default at 1.82× the full-portfolio rate.
This is a lagging indicator (collections activity follows delinquency) and is never
used as a model feature. Its business use is operational, not predictive: it supports
routing Mortgage accounts with any collections event to intensified servicing.

## Supporting cross-tabs
**Income band × employment type (default rate):**
| | Business Owner | Contract | Salaried | Self-Employed | Unemployed | All |
|---|---|---|---|---|---|---|
| <50k | 20.5% | 29.5% | 22.2% | 23.8% | 40.2% | 27.9% |
| 50-80k | 18.4% | 16.8% | 17.3% | 18.1% | 26.4% | 17.9% |
| 80-120k | 11.1% | 13.1% | 12.4% | 11.6% | 21.9% | 12.2% |
| 120k+ | 8.7% | 7.4% | 7.5% | 7.9% | 0.0%* | 7.8% |
| **All** | **13.6%** | **17.3%** | **15.0%** | **15.3%** | **35.0%** | **16.1%** |

\* 120k+ × Unemployed has n=1; flagged low-confidence (small_sample=True in
`low_income_employment_detail()`), not a reliable 0%.

**Region × loan type (default rate):** flat across region for most loan types; Mortgage
default rate ranges from 21.3% (Central) to 27.3% (South) — a secondary signal worth
monitoring but not strong enough for a standalone action beyond #2 above.
