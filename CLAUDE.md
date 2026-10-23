# Credit Risk Intelligence Platform
End-to-end credit risk analytics + AI on a synthetic banking portfolio
(5 tables, 10k customers). Data: data/raw/. Schema: docs/DATA_DICTIONARY.md.

## 0. Preflight (run first, report before planning)
- 5 CSVs in data/raw/ with row counts: customers 10000, loan_accounts 14685,
  bureau_credit 22979, repayment_history 176220, collections_events 8635
- Packages import: pandas, sklearn, lightgbm, shap, streamlit, plotly, pytest, anthropic, dotenv
- .env has ANTHROPIC_API_KEY (check presence only, never print it)
- Report anything missing. Do not proceed until resolved.

## 1. Deliverables
1. Ingestion: join all 5 tables (document keys/join logic), handle missing values,
   inconsistent labels, other DQ issues → clean master table (1 row per account) + docs/dq_notes.md
2. Analytics: profile by income, employment, region, age, loan type; 2+ cross-tabs;
   5 insights with business impact (segment share of exposure, relative default rate, action)
3. Model: engineered features (bureau + repayment aggregates). Logistic Regression baseline + LightGBM.
   AUC, KS, Gini, decile table, SHAP. Monitoring note: PSI/CSI, retraining cadence.
4. Streamlit dashboard: portfolio view (segments, exposure, trends), account view
   (score + key drivers), at least one working filter/drill-down
5. AI feature embedded in dashboard: "Ask the Portfolio" — LLM writes pandas query,
   it executes, answer uses only the result, query shown. Document trigger,
   false positive/negative examples, human-in-the-loop step.
6. docs/: product brief (1 page: problem, target user, scope in/out), architecture diagram
   (ingestion → analytics → model → dashboard + AI, tools per stage), assumptions log
   (incl. what a real bank's IT/data team must provide), limitations, 10-slide deck
   for a Chief Risk Officer with risk/compliance angle

## 2. Locked decisions (do not redesign)
- Stack: pandas, scikit-learn, LightGBM, SHAP, Streamlit, plotly, pytest, Anthropic API
- Time-based train/test split on origination_date
- No collections features in the model (leakage); repayment features from early months only
- Out of scope: deep learning, heavy tuning, auth, cloud deployment

## 3. Workflow
One sub-problem at a time. After each:
1. Test: 3–5 pytest asserts in tests/; don't proceed until green.
2. Complexity gate: propose simple vs richer option, I choose. Serves a deliverable?
   Explainable in 2 lines? Realistic for a bank? >45 min → simplify, log in docs/limitations.md.
3. Neutral language: personal portfolio project. No company names or hiring-process words anywhere.
- Log data fixes → docs/dq_notes.md, assumptions → docs/assumptions.md
- PLAN.md changes only if a test or the data contradicts it
- Commits: short messages, never add Co-Authored-By lines. Commit only when I ask.
