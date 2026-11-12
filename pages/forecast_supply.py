"""Streamlit page; consumes the existing load_data()/filter column contract."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.forecasting import weekly_series, evaluate_models, future_forecast, MIN_WEEKS
from analytics.inventory import inventory_table
from theme import PALETTE, style, insight, tl


@st.cache_data(show_spinner=False, max_entries=12)
def cached_evaluation(y):
    return evaluate_models(y)


@st.cache_data(show_spinner=False, max_entries=18)
def cached_forecast(y, evaluation, steps):
    return future_forecast(y, evaluation, steps)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_inventory(df, lookback, lead, service, lead_std):
    return inventory_table(df, lookback, lead, service, lead_std)


def _line_figure(history, future, label):
    fig = go.Figure()
    fig.add_scatter(x=history.index, y=history.values, name="Observed complete weeks", mode="lines",
                    line=dict(color=PALETTE[0], width=2))
    fig.add_scatter(x=future.index, y=future.Upper95, name="Approx. 95% prediction band", mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=future.index, y=future.Lower95, name="Approx. 95% prediction band", mode="lines",
                    fill="tonexty", fillcolor="rgba(46,94,170,0.15)", line=dict(width=0))
    fig.add_scatter(x=future.index, y=future.Forecast, name="Selected model forecast", mode="lines",
                    line=dict(color=PALETTE[1], dash="dash", width=2))
    fig.update_layout(title="Weekly sales forecast and uncertainty", xaxis_title="Week ending Sunday", yaxis_title=label)
    return style(fig, height=430)


def render_forecast(df):
    left, right = st.columns(2)
    target_label = left.selectbox("Forecast target", ["Units", "Net revenue (TL)"], key="fs_target")
    months = right.selectbox("Forecast horizon (months, approximately)", [3, 6, 12], key="fs_horizon")
    target = "Quantity" if target_label == "Units" else "Revenue"
    unit = "Units" if target == "Quantity" else "Net revenue (TL)"
    try:
        y, meta = weekly_series(df, target)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.caption(f"{len(y)} complete weeks: {meta['complete_start']:%d %b %Y} to {meta['complete_end']:%d %b %Y}. "
               f"{meta['excluded_lines']:,} boundary-week lines omitted. Zero-sale interior weeks are included as zero.")
    if len(y) < MIN_WEEKS:
        st.warning(f"Select at least {MIN_WEEKS} complete weeks for model comparison; currently {len(y)}. "
                   "Widen the shared date filter. Inventory analysis remains available.")
        return
    if y.sum() <= 0:
        st.warning("No positive sales for the selected forecast target.")
        return
    with st.spinner("Comparing models with chronological validation..."):
        evaluation = cached_evaluation(y)
        try:
            future = cached_forecast(y, evaluation, {3: 13, 6: 26, 12: 52}[months])
        except (RuntimeError, ValueError) as exc:
            st.error(f"Selected model could not be refitted: {exc}")
            return
    winner = evaluation["winner"]
    test_scores = evaluation["test_scores"].set_index("Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", winner)
    c2.metric("Validation RMSE", f"{evaluation['cv'].iloc[0].RMSE:,.0f}")
    c3.metric("Test RMSE", f"{test_scores.loc[winner, 'RMSE']:,.0f}" if winner in test_scores.index else "Unavailable")
    c4.metric("Test MAPE", f"{test_scores.loc[winner, 'MAPE_pct']:.1f}%" if winner in test_scores.index else "Unavailable")
    st.plotly_chart(_line_figure(y, future, unit), width="stretch")
    recent = y.iloc[-13:].mean()
    change = future.Forecast.mean()/recent - 1 if recent > 0 else np.nan
    total = tl(future.Forecast.sum()) if target == "Revenue" else f"{future.Forecast.sum():,.0f} units"
    insight(f"{winner} projects {total} over {len(future)} weeks; the weekly average is {change:+.1%} versus the last 13 complete weeks. "
            "Use the short horizon for capacity planning and update the forecast as new sales arrive.")
    st.caption(f"Origin: week ending {y.index[-1]:%d %b %Y}. The first forecast week may already contain "
               f"transactions through {meta['transaction_end']:%d %b %Y}; this partial week was excluded from training. "
               "The shaded band is an approximate prediction interval, not a confidence interval for the mean. "
               f"It uses {evaluation['calibration_n']} pretest validation errors and sqrt(horizon) scaling. "
               f"Untouched test coverage: {evaluation['test_coverage']:.0%}. Nominal 95% coverage is not guaranteed.")
    if len(y) < 104:
        st.warning("Fewer than two annual cycles are available. Annual seasonality cannot be validated; Holt uses a damped trend without seasonality. "
                   "6/12-month forecasts are exploratory scenarios; the evaluation covers only 8-week horizons.")
    st.download_button("Download forecast CSV", future.to_csv(index=True).encode("utf-8-sig"),
                       f"forecast_{target}_{months}m.csv", "text/csv", key="fs_forecast_csv")

    with st.expander("Model comparison and untouched test", expanded=True):
        st.write(f"Model selection uses expanding-window validation with {len(evaluation['origins'])} origins and 8-week forecasts. "
                 f"The final test ({evaluation['test_start']:%d %b} to {evaluation['test_end']:%d %b %Y}) never selects the model.")
        comparison = evaluation["cv"].rename(columns={c: f"CV {c}" for c in ["RMSE", "MAE", "MAPE_pct", "WAPE_pct", "MAPE_n", "n"]})
        comparison = comparison.merge(evaluation["test_scores"].rename(columns={c: f"Test {c}" for c in ["RMSE", "MAE", "MAPE_pct", "WAPE_pct", "MAPE_n", "n"]}), on="Model", how="left")
        st.dataframe(comparison, hide_index=True, width="stretch")
        bars = px.bar(evaluation["cv"], x="Model", y="RMSE", title="Validation error by model", labels={"RMSE": f"RMSE ({unit})"})
        st.plotly_chart(style(bars), width="stretch")
        baseline = evaluation["cv"].set_index("Model").loc["Naive", "RMSE"]
        improvement = 1 - evaluation["cv"].iloc[0].RMSE / baseline if baseline else 0.
        insight(f"The selected model reduces validation RMSE by {improvement:.1%} versus Naive. "
                "Retain the simple baseline whenever extra model complexity does not improve validation.")
        p = evaluation["test_predictions"]
        chart = go.Figure()
        for name in dict.fromkeys(["Actual", winner, "Naive"]):
            if name in p:
                chart.add_scatter(x=p.index, y=p[name], name=name, mode="lines+markers")
        chart.update_layout(title="Final 8-week test: actual versus forecasts", xaxis_title="Week ending Sunday", yaxis_title=unit)
        st.plotly_chart(style(chart), width="stretch")
        if winner in test_scores.index:
            m = test_scores.loc[winner]
            insight(f"On {int(m['n'])} untouched weeks, {winner} has RMSE {m.RMSE:,.0f} and MAPE {m.MAPE_pct:.1f}%. "
                    "Plan an operating buffer around the forecast; do not treat the point estimate as certain demand.")
        st.caption("MAPE excludes zero actuals; MAPE_n shows its denominator. WAPE includes all test periods. "
                   "Random Forest uses lags 1/2/3/4/8/13, past rolling statistics and trend; future lags use its own predictions. "
                   "ARIMA is fixed at (1,1,1); there is no test-set tuning.")
        if not evaluation["issues"].empty:
            st.warning("Some model fits failed and were excluded at the stated stage.")
            st.dataframe(evaluation["issues"], hide_index=True)
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode("utf-8-sig"),
                           "model_comparison.csv", "text/csv", key="fs_compare_csv")


def render_inventory(df):
    st.subheader("Product demand and replenishment policy")
    a, b, c, d = st.columns(4)
    lookback = a.selectbox("Lookback (calendar days)", [90, 180, 366], key="fs_lookback")
    lead = b.number_input("Assumed lead time (days)", min_value=1, max_value=180, value=14, key="fs_lead")
    service = c.selectbox("Cycle service level", [90, 95, 99], index=1, key="fs_service") / 100
    lead_std = d.number_input("Assumed lead-time std (days)", min_value=0., max_value=60., value=0., step=1., key="fs_lead_std")
    with st.spinner("Calculating product demand including zero-sale days..."):
        table, meta = cached_inventory(df, lookback, lead, service, lead_std)
    st.caption(f"Window: {meta['start']:%d %b %Y} to {meta['end']:%d %b %Y}; {meta['days']} calendar days, "
               f"{meta['complete_weeks']} complete weeks for XYZ. ABC is based on net revenue; XYZ is based on weekly units.")
    a, b, c, d = st.columns(4)
    a.metric("Products in window", f"{len(table):,}")
    aclass = table.loc[table.ABC == "A"]
    b.metric("Class A revenue share", f"{aclass.RevenueShare.sum():.1%}")
    c.metric("Z products", f"{(table.XYZ == 'Z').mean():.1%}")
    d.metric("Products needing manual review", f"{table.ManualReview.sum():,}")
    grouped = table.groupby(["ABC", "XYZ"], observed=True).agg(Products=("StockCode", "size"), Revenue=("Revenue", "sum")).reset_index()
    grid = grouped.pivot(index="ABC", columns="XYZ", values="Products").reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"]).fillna(0)
    heatmap = px.imshow(grid, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                        title="ABC-XYZ matrix: number of products", labels={"x": "Weekly demand variability", "y": "Revenue class", "color": "Products"})
    st.plotly_chart(style(heatmap), width="stretch")
    az = table.loc[table.ABC_XYZ == "AZ"]
    insight(f"{len(aclass):,} A products ({len(aclass)/len(table):.1%} of the window's products) generate {aclass.RevenueShare.sum():.1%} of revenue. "
            f"{len(az):,} are AZ and account for {az.RevenueShare.sum():.1%} of revenue; prioritize these for supplier and order review.")
    if (table.XYZ == "Unclassified").any():
        st.caption(f"{(table.XYZ == 'Unclassified').sum():,} products lack complete-week demand statistics and are excluded from the X/Y/Z matrix.")
    scatter = table.nlargest(500, "Revenue")
    fig = px.scatter(scatter, x="MeanWeekly", y="CV", color="ABC", size="Revenue", hover_name="Description",
                     hover_data=["StockCode", "ABC_XYZ", "ActiveWeeks"], size_max=28,
                     title="Demand variability: top 500 products by window revenue", labels={"MeanWeekly": "Mean weekly units", "CV": "Weekly coefficient of variation"})
    st.plotly_chart(style(fig), width="stretch")
    insight(f"{table.Intermittent.sum():,} products have average demand intervals of at least 1.32 weeks. "
            "Review these products individually because intermittent purchases weaken normal-distribution stock estimates.")
    st.info("There are no stock-on-hand, open purchase order, promised-delivery or actual-delivery fields. "
            "Reorder points below are policy thresholds under the entered assumptions; actual order quantities and delivery performance cannot be calculated.")
    selected_classes = st.multiselect("Show ABC-XYZ classes", sorted(table.ABC_XYZ.unique()), key="fs_classes", placeholder="All classes")
    view = table.loc[table.ABC_XYZ.isin(selected_classes)] if selected_classes else table
    display_cols = ["StockCode", "Description", "Category", "ABC_XYZ", "Revenue", "Units", "MeanDaily", "StdDaily", "CV", "ADI", "SafetyStockUnits", "ReorderPoint", "ManualReview", "Policy"]
    st.dataframe(view[display_cols].head(500), hide_index=True, width="stretch")
    st.caption(f"Showing {min(500, len(view)):,} of {len(view):,} selected products. CSV includes every selected product.")
    st.download_button("Download stock policy CSV", view.to_csv(index=False).encode("utf-8-sig"),
                       "abc_xyz_stock_policy.csv", "text/csv", key="fs_stock_csv")
    with st.expander("Stock formulas and assumptions"):
        st.markdown("**ABC:** prior cumulative net-revenue share <80% = A, <95% = B, otherwise C. "
                    "The product crossing a threshold stays in the preceding class. **XYZ:** weekly CV ≤0.5 = X, ≤1.0 = Y, >1.0 = Z. "
                    "Products are keyed by StockCode; zero-sale days/weeks are included over the common window.")
        st.latex(r"SS = z\sqrt{L\sigma_d^2 + \mu_d^2\sigma_L^2}, \qquad ROP = \lceil L\mu_d + SS\rceil")
        st.write(f"z = {meta['z']:.3f}; L = {lead} calendar days; lead-time std = {lead_std:g} days. "
                 "mu_d and sigma_d are the daily sample mean and sample standard deviation. "
                 "Service level means cycle service probability, not fill rate. Demand and lead time are assumed independent; "
                 "serial dependence, product availability and stockouts are not observed. Short histories (<28 days or <4 active weeks) "
                 "and intermittent demand (ADI >=1.32) are flagged for manual review. Reorder only when inventory position reaches the threshold.")


def render(df):
    """Required group contract: render the already cleaned and filtered frame."""
    st.title("Supply Chain & Forecast")
    st.caption("Sales forecasts, product demand variability and replenishment thresholds.")
    if df.empty:
        st.warning("No transactions in the selected filters.")
        return
    forecast, inventory = st.tabs(["Sales forecast", "Supply chain"])
    with forecast:
        render_forecast(df)
    with inventory:
        render_inventory(df)
