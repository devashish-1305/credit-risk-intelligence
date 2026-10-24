# Product Brief: Credit Risk Intelligence Platform

## Problem
A retail lending book is reviewed mostly through static reports. Risk teams struggle to see which
segments carry the exposure and the defaulted exposure, and which new loans are heading for trouble
early enough to act. Ad-hoc questions ("what share of exposure defaulted for unemployed borrowers?")
wait in an analyst queue. This project gives one place for portfolio visibility and early warning on a
synthetic retail book (10,000 customers, 14,685 loan accounts).

**Nature of this work:** a retrospective evaluation on synthetic data. It is not a production lending-decision system, and its scores must not be used to approve, price or decline credit.

## Target users
- **Chief Risk Officer:** portfolio-level exposure, concentration and early-warning view.
- **Portfolio risk managers:** segment profiles, cross-tabs, score and risk-band monitoring.
- **Collections / servicing leads:** which accounts to contact first, and the reason codes behind a flag.

## What it does
- **Portfolio analytics:** default rate, exposure share and defaulted-exposure share by income, employment,
  region, age and loan type, with cross-tabs and five insights ([insights.md](insights.md)).
- **Month-3 early-warning score:** a logistic regression champion (LightGBM challenger) scored three
  months after origination, shown as four risk bands (Low, Medium, High, Very High) with the top-5
  reason codes per account. Test AUC 0.675, KS 0.277, Gini 0.351 ([model_report.md](model_report.md)).
- **Ask the Portfolio:** a language model writes one pandas query, an allow-list checks it, a person
  reviews it and clicks **Run query**, and the answer uses only the result ([ai_feature.md](ai_feature.md)).

## Scope
| In | Out |
|---|---|
| Portfolio analytics and dashboard | Application scoring (the model is scored at month 3, not at origination) |
| Month-3 default early-warning score, risk bands, reason codes | LGD, recovery or loss forecasting |
| Model validation, PSI/CSI monitoring plan | Automated credit decisions |
| Grounded, human-approved natural-language queries | Production deployment, authentication |

## Success measures
- **Defaults captured in the top risk bands:** on the 2022–23 test set, High and Very High hold 30.1% of
  accounts and 51.9% of defaults (405 of 780). Test default rates run from 8.6% (Low) to 34.7% (Very High).
- **Stability:** score PSI below 0.10 and input CSI below 0.10 on each monthly check (currently 0.006 for the
  champion's score PSI).
- **Analyst time saved:** share of ad-hoc portfolio questions answered by Ask the Portfolio without an
  analyst, and the share of generated queries a reviewer accepts. Not measured here; it needs real users.

## Key risks and limitations
Full list in [limitations.md](limitations.md).
- **Optimistic performance:** bureau features are snapshots and the default flag has no event date or
  horizon, so AUCs overstate a real month-3 decision (L4).
- **Customer overlap across the time split:** 42% of test accounts belong to a customer seen in training;
  the effect on AUC is inconclusive (L1).
- **No calibration or LGD data:** scores are rankings, not probabilities or loss estimates (L3).
- **Query correctness:** the allow-list guarantees a query is safe, not that it answers the question asked,
  so the human review step is essential ([ai_feature.md](ai_feature.md)).
