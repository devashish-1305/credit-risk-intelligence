# Credit Risk Intelligence Platform

A personal portfolio project: end-to-end credit risk analytics on a synthetic retail banking portfolio
(10,000 customers, 14,685 loan accounts, 5 source tables). The pipeline cleans and joins the raw
tables into one row per account, profiles default risk and exposure by segment, and trains a
**month-3 early-warning default model**. The model is a logistic regression champion, with LightGBM as
the challenger and an opened-before-origination bureau variant as a sensitivity check. A Streamlit
dashboard shows the portfolio, the model and individual accounts. It also includes "Ask the
Portfolio": a language model writes a pandas query, a person reviews it, and the answer uses only the
query's result. The model is scored three months after origination; it is not an application scorecard.

## Key results

Test set: 4,921 accounts originated 2022–23. The models were trained on 2018–21 originations.

| Model | Role | AUC | KS | Gini |
|---|---|---|---|---|
| Logistic regression | Champion | 0.675 | 0.277 | 0.351 |
| LightGBM | Challenger | 0.673 | 0.262 | 0.345 |
| LightGBM, opened-before-origination bureau | Sensitivity variant | 0.657 | 0.234 | 0.315 |
| Logistic regression, opened-before-origination bureau | Sensitivity variant | 0.671 | 0.267 | 0.341 |

The LR row is a separate in-memory check: LR primary − LR variant AUC = +0.005 (bootstrap 95% CI −0.002 to
+0.012, inconclusive). The kept bureau lines still carry snapshot status, balance and DPD, so this does not
show the champion is leakage-free.

- **Champion vs challenger:** LR − LightGBM AUC = +0.003, bootstrap 95% CI −0.008 to +0.014. Neither
  model is shown to be better.
- **Champion risk bands:** test default rates are Low 8.6%, Medium 14.0%, High 23.4% and Very High
  34.7%. The bands are set from training-score quantiles at 40/30/20/10% of accounts.

**Top insights** (details in [docs/insights.md](docs/insights.md)):
1. Unemployed borrowers are 4.9% of exposure but 10.8% of defaulted exposure, with a 35.0% default rate.
2. Mortgage is 9.8% of accounts, 52.3% of exposure and 65.0% of defaulted exposure.
3. Borrowers earning under 50k default at 27.9%, against 7.8% for 120k+.
4. Borrowers under 30 default at 21.6%, 1.34× the portfolio rate, across nearly a quarter of the book.
5. Two or more missed payments by month 3 go with a 33.3% default rate. This is an early-warning
   association, not a cause.

## Setup

These steps were tested on macOS with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp          # macOS only: LightGBM needs OpenMP
```

### Data

The raw data is **not included in the repository**. Put the 5 CSVs in `data/raw/`:

```
data/raw/customers.csv
data/raw/loan_accounts.csv
data/raw/bureau_credit.csv
data/raw/repayment_history.csv
data/raw/collections_events.csv
```

The schema is in [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md). Everything derived from the CSVs is
written to `data/processed/`.

### Optional: Ask the Portfolio

Ask the Portfolio needs an Anthropic API key. Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key
# ASK_MODEL=claude-sonnet-5-5   # optional override
```

Without a key, the tab shows a message and the rest of the dashboard works normally.

## Run

```bash
python -m src.run_all          # ingest → master → analytics → model → validation (about 15 s)
pytest -q                      # tests (the language model is mocked; no API key needed)
streamlit run app/app.py       # dashboard
```

`run_all` stops with a clear message if any raw CSV is missing. Each stage can also be run on its own,
for example `python -m src.model` or `python -m src.validation`.

## Repository structure

```
app/app.py             Streamlit dashboard (Portfolio, Model, Account, Ask the Portfolio tabs)
src/ingest.py          load + clean the 5 raw tables → data/processed/*.parquet
src/master.py          one row per account: customer, bureau and month 1–3 repayment features
src/analytics.py       segment profiles, cross-tabs
src/model.py           time-split LR + LightGBM, metrics, deciles, risk bands, SHAP, PSI/CSI
src/validation.py      bootstrap CIs, grouped retrain, fairness table → validation.json
src/dashboard.py       loaders, filters and account lookup behind the dashboard
src/ask.py             Ask the Portfolio: query generation, allow-list, execution, answer
src/run_all.py         full pipeline in one command
tests/                 pytest suite
docs/                  documentation (below)
data/raw/              raw CSVs (not in the repository)
data/processed/        pipeline outputs (generated)
```

## Documentation

- [PLAN.md](PLAN.md): sub-problems and locked design decisions
- [docs/product_brief.md](docs/product_brief.md): one-page brief (problem, users, scope, success measures)
- [docs/architecture.md](docs/architecture.md): pipeline and Ask the Portfolio diagram, controls, production path
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md): source schema
- [docs/dq_notes.md](docs/dq_notes.md): data-quality findings and fixes
- [docs/assumptions.md](docs/assumptions.md): assumptions log
- [docs/insights.md](docs/insights.md): portfolio insights with business impact
- [docs/model_report.md](docs/model_report.md): model results, SHAP, fairness, calibration, monitoring plan
- [docs/ai_feature.md](docs/ai_feature.md): Ask the Portfolio design, failure examples, human-in-the-loop step
- [docs/limitations.md](docs/limitations.md): known limitations

## Limitations

- The data is synthetic. Bureau features are snapshot values that may postdate origination, and the
  default flag has no event date or horizon, so the AUCs are optimistic for a real month-3 decision.
- Some customers have loans in both train and test. A grouped retrain gives an inconclusive effect.
- Scores are uncalibrated rankings, and there is no loss-given-default data.
- Flag rates differ sharply by age band. Fairness questions are open and recorded in the model report.

See [docs/limitations.md](docs/limitations.md) for the full list.
