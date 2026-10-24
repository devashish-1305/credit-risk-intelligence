"""Credit risk dashboard. Run: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src import ask
from src import dashboard as d
from src.model import BAND_LABELS

BAND_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#f4a261", "Very High": "#d1495b"}
BLUE, RED, GREEN = "#3b6ea8", "#d1495b", "#2a9d8f"

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")


@st.cache_data
def load_all():
    portfolio, scores = d.load_portfolio(), d.load_test_scores()
    return (portfolio, d.scored_test(portfolio, scores), d.load_lr_drivers(), d.load_model_summary(),
            ask.build_ask_frame(portfolio, scores))


portfolio, test, lr_drivers, summary, ask_df = load_all()

# --- sidebar filters (applied to every tab) ---------------------------------
st.sidebar.header("Filters")
options = d.filter_options(portfolio)
selections = {col: st.sidebar.multiselect(label, options[col], key=col) for col, label in d.FILTER_COLS.items()}
st.sidebar.caption("Empty = all. Filters apply to every tab; Model and Account tabs use the 2022-23 test accounts.")

view = d.apply_filters(portfolio, selections)
test_view = d.apply_filters(test, selections)

st.title("Credit Risk Intelligence")
tab_portfolio, tab_model, tab_account, tab_ask = st.tabs(["Portfolio", "Model", "Account", "Ask the Portfolio"])

# --- Portfolio --------------------------------------------------------------
with tab_portfolio:
    if view.empty:
        st.warning("No accounts match the selected filters.")
    else:
        k = d.kpis(view)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accounts", f"{k['accounts']:,}")
        c2.metric("Exposure", f"{k['exposure'] / 1e6:,.1f}M")
        c3.metric("Default rate", f"{k['default_rate']:.1%}")
        c4.metric("Share of defaulted exposure", f"{k['defaulted_exposure_share']:.1%}")
        st.caption("Exposure = original loan amount. Share of defaulted exposure = defaulted loan amount / total loan amount.")

        by = st.selectbox("Segment", ["region", "loan_type", "employment_type", "income_band", "age_band"],
                          format_func=lambda c: c.replace("_", " ").capitalize())
        prof = d.segment_profile(view, by)
        left, right = st.columns(2)
        fig = px.bar(prof, x=by, y="default_rate", hover_data={"n_accounts": True, "relative_risk": ":.2f"},
                     color_discrete_sequence=[BLUE], title="Default rate by segment")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title=None)
        left.plotly_chart(fig, width="stretch")
        fig = px.bar(prof, x=by, y="pct_exposure", hover_data={"exposure": ":,.0f"},
                     color_discrete_sequence=[BLUE], title="Share of exposure by segment")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title=None)
        right.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        grid = d.income_employment_grid(view)
        heat = grid.pivot(index="income_band", columns="employment_type", values="default_rate")
        counts = grid.pivot(index="income_band", columns="employment_type", values="n")
        fig = px.imshow(heat, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                        title="Default rate: income band x employment type")
        fig.update_traces(customdata=counts.reindex_like(heat).to_numpy(),
                          hovertemplate="%{y} / %{x}<br>default rate %{z:.1%}<br>%{customdata} accounts<extra></extra>")
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        left.plotly_chart(fig, width="stretch")
        left.caption("Cells with few accounts are noisy; hover for the account count.")

        year = d.default_rate_by_year(view)
        fig = px.line(year, x="orig_year", y="default_rate", markers=True, hover_data={"accounts": True},
                      color_discrete_sequence=[BLUE], title="Default rate by origination year")
        fig.update_yaxes(tickformat=".0%", title=None, rangemode="tozero")
        fig.update_xaxes(title=None, dtick=1)
        right.plotly_chart(fig, width="stretch")

# --- Model ------------------------------------------------------------------
with tab_model:
    st.caption(f"Test accounts (originated 2022-23) matching the filters: {len(test_view):,} "
               f"of {len(test):,}. Metrics and charts are recomputed on this selection; models are not retrained.")
    table = d.model_metrics_table(test_view)
    st.dataframe(table.style.format({"AUC": "{:.3f}", "KS": "{:.3f}", "Gini": "{:.3f}"}, na_rep="n/a"),
                 hide_index=True, width="stretch")
    if table["AUC"].isna().all():
        st.info(f"AUC/KS/Gini are n/a: {len(test_view):,} test accounts selected. They need at least "
                f"{d.MIN_METRIC_ROWS} accounts and both outcome classes (defaults and non-defaults) in the selection.")

    left, right = st.columns(2)
    deciles = d.lr_decile_table(test_view)
    if deciles.empty:
        left.info("Too few accounts for a decile chart.")
    else:
        fig = px.bar(deciles, x="decile", y="default_rate", hover_data={"n": True, "lift": ":.2f"},
                     color_discrete_sequence=[BLUE], title="Champion (LR): default rate by filtered-cohort decile")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title="Filtered-cohort decile (1 = highest risk)", dtick=1)
        left.plotly_chart(fig, width="stretch")
        left.caption("Filtered-cohort deciles are re-cut within the selected accounts, so they change with the filters. "
                     "Risk bands stay fixed (thresholds from training-period LR scores).")
    bands = d.risk_band_table(test_view)
    fig = px.bar(bands, x="risk_band", y="default_rate", color="risk_band", color_discrete_map=BAND_COLORS,
                 category_orders={"risk_band": BAND_LABELS}, hover_data={"n": True},
                 title="Champion (LR): default rate by risk band")
    fig.update_yaxes(tickformat=".0%", title=None)
    fig.update_layout(showlegend=False, xaxis_title=None)
    right.plotly_chart(fig, width="stretch")

    shap_df = d.shap_top10(summary).iloc[::-1]
    fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=["#8d6bb5"],
                 title="Challenger insight: LightGBM top-10 features (mean |SHAP|, full test set)")
    fig.update_layout(yaxis_title=None, xaxis_title="Mean |SHAP| (log-odds)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Challenger insight: explains the LightGBM model, not the champion. Computed on the full test set; "
               "not affected by the filters.")

# --- Account ----------------------------------------------------------------
with tab_account:
    st.caption("Early-warning score at month 3, not an application score.")
    if test_view.empty:
        st.warning("No test accounts match the selected filters.")
    else:
        account_id = st.selectbox("Account (2022-23 test accounts)", sorted(test_view["account_id"]))
        res = d.account_lookup(account_id, test_view, lr_drivers)
        c1, c2 = st.columns(2)
        c1.metric("Risk band", res["risk_band"])
        c2.metric("Test-set score decile (1 = highest risk)", res["decile"])
        drv = res["drivers"].iloc[::-1]
        fig = px.bar(drv, x="contribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"raises risk": RED, "lowers risk": GREEN},
                     custom_data=["contribution"], title="Top 5 drivers of this account's score")
        fig.update_traces(hovertemplate="%{y}<br>Contribution: %{customdata[0]:+.3f} log-odds<extra></extra>")
        fig.update_layout(yaxis_title=None, xaxis_title="Impact on risk (vs. average account)", legend_title=None)
        fig.update_xaxes(showticklabels=False, zeroline=True)
        st.plotly_chart(fig, width="stretch")
        st.caption("Red bars raise the model's predicted risk; green bars lower it. Each bar is the feature's directional "
                   "contribution to the model score relative to an average training account (signed log-odds on hover). "
                   "It does not establish a causal effect on default.")

# --- Ask the Portfolio ------------------------------------------------------
with tab_ask:
    ask_view = d.apply_filters(ask_df, selections)
    st.caption("Ask a question about the portfolio. The model writes a pandas query from the column list only "
               "(it never sees account rows); you review it and click Run query; the answer uses only the result.")
    with st.form("ask_form", border=False):
        question = st.text_input("Question", placeholder="e.g. Default rate by region, highest first")
        submitted = st.form_submit_button("Generate query")
    ask.clear_stale(st.session_state, question)
    if submitted and question.strip():
        st.session_state.pop("ask_result", None)
        try:
            expr, description = ask.generate_query(question, ask_view, ask.get_client())
            st.session_state["ask_query"] = (question.strip(), expr, description)
        except ask.AskError as e:
            st.session_state.pop("ask_query", None)
            st.error(str(e))

    if "ask_query" in st.session_state:
        asked, expr, description = st.session_state["ask_query"]
        with st.expander("Show query"):
            st.code(expr, language="python")
        st.markdown(f"**This query will compute:** {description}")
        st.caption("Written by the model — the query is what runs.")
        if st.button("Run query"):
            try:
                result, truncated = ask.run_query(expr, ask_view)
                answer = ask.answer_question(asked, result, len(ask_view), ask.get_client(), truncated,
                                             ask.uses_scores(expr), ask.uses_identifiers(expr))
                st.session_state["ask_result"] = (result, truncated, answer, len(ask_view), ask.uses_scores(expr))
            except ask.AskError as e:
                st.session_state.pop("ask_result", None)
                st.error(str(e))

    if "ask_result" in st.session_state:
        result, truncated, answer, n, scores_used = st.session_state["ask_result"]
        st.markdown(f"**Answer:** {answer}")
        st.caption(f"Answered on {n:,} accounts (current sidebar filters).")
        if scores_used:
            st.caption("Score, decile and risk band figures cover 2022-23 test accounts only.")
        if isinstance(result, (pd.DataFrame, pd.Series)):
            st.dataframe(result, width="stretch")
            if truncated:
                st.caption(f"Result capped at the first {ask.MAX_ROWS} rows.")
        else:
            st.write(result)
