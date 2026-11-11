import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest


def abc_analysis(df):
    """Pareto / ABC classification of products by revenue.
    A = products making the first 80% of revenue, B = next 15%, C = last 5%."""
    products = (df.groupby(["StockCode", "Description"], as_index=False)["Revenue"].sum()
                .sort_values("Revenue", ascending=False).reset_index(drop=True))
    products["CumShare"] = products["Revenue"].cumsum() / products["Revenue"].sum()
    products["ProductShare"] = (products.index + 1) / len(products)
    products["Class"] = pd.cut(products["CumShare"], bins=[0, 0.80, 0.95, 1.0],
                               labels=["A", "B", "C"], include_lowest=True)
    return products


def detect_anomalies(series, method, contamination=0.03):
    """Flag unusual periods in a revenue time series.
    - Mean ± 3 std: a period is an anomaly if it is more than 3 standard
      deviations away from the mean.
    - IsolationForest: ML model that isolates points that are 'easy to separate'
      from the rest (uses revenue and number of orders)."""
    out = series.copy()
    if method == "Mean ± 3 std":
        mean, std = out["Revenue"].mean(), out["Revenue"].std()
        out["Lower"], out["Upper"] = mean - 3 * std, mean + 3 * std
        out["Anomaly"] = (out["Revenue"] < out["Lower"]) | (out["Revenue"] > out["Upper"])
    else:
        model = IsolationForest(contamination=contamination, random_state=42)
        out["Anomaly"] = model.fit_predict(out[["Revenue", "Orders"]]) == -1
    return out


def show(df):
    st.title("Sales Analysis")

    # --- Filter: quarters ---
    quarters = sorted(df["Quarter"].unique())
    selected_q = st.multiselect("Quarter", quarters, default=quarters)
    data = df[df["Quarter"].isin(selected_q)]
    if data.empty:
        st.warning("Select at least one quarter.")
        return

    tab1, tab2, tab3 = st.tabs(["📚 Category & Product", "📊 Pareto (ABC)", "⚠️ Anomaly Detection"])

    # ------------------------------------------------------------------
    # 1) Category and product breakdown
    # ------------------------------------------------------------------
    with tab1:
        cat = (data.groupby("Category", as_index=False)
               .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                    Orders=("InvoiceNo", "nunique"), AvgDiscount=("DiscountRate", "mean"))
               .sort_values("Revenue", ascending=False))
        cat["RevenueShare"] = cat["Revenue"] / cat["Revenue"].sum()

        n = st.slider("Number of categories to show", 5, 30, 15)
        top = cat.head(n).sort_values("Revenue")
        fig = px.bar(top, x="Revenue", y="Category", orientation="h",
                     color="AvgDiscount", color_continuous_scale="Blues",
                     labels={"Category": "", "Revenue": "Revenue (₺)", "AvgDiscount": "Avg. discount"})
        fig.update_layout(height=30 * n + 100)
        st.plotly_chart(fig, width="stretch")

        top3_share = cat["RevenueShare"].head(3).sum()
        st.info(f"💡 Top 3 categories ({', '.join(cat['Category'].head(3))}) "
                f"generate **{top3_share:.0%}** of revenue out of {len(cat)} categories.")

        # Drill-down: products in one category
        st.subheader("Products in a category")
        chosen = st.selectbox("Category", cat["Category"])
        prod = (data[data["Category"] == chosen]
                .groupby("Description", as_index=False)
                .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                     Customers=("CustomerID", "nunique"))
                .sort_values("Revenue", ascending=False).head(15))
        st.dataframe(prod, hide_index=True, width="stretch",
                     column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f")})

    # ------------------------------------------------------------------
    # 2) Pareto / ABC analysis
    # ------------------------------------------------------------------
    with tab2:
        products = abc_analysis(data)
        n_a = (products["Class"] == "A").sum()
        share_a = n_a / len(products)

        c1, c2, c3 = st.columns(3)
        c1.metric("Products sold", f"{len(products):,}")
        c2.metric("A-class products (80% of revenue)", f"{n_a:,}")
        c3.metric("Share of products in A", f"{share_a:.1%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=products["ProductShare"] * 100, y=products["CumShare"] * 100,
                                 mode="lines", name="Cumulative revenue"))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% revenue")
        fig.add_vline(x=share_a * 100, line_dash="dash", line_color="red",
                      annotation_text=f"{share_a:.1%} of products")
        fig.update_layout(xaxis_title="% of products (sorted by revenue)",
                          yaxis_title="% of total revenue", height=420)
        st.plotly_chart(fig, width="stretch")

        summary = (products.groupby("Class", observed=True)
                   .agg(Products=("StockCode", "count"), Revenue=("Revenue", "sum")))
        summary["% of products"] = summary["Products"] / summary["Products"].sum()
        summary["% of revenue"] = summary["Revenue"] / summary["Revenue"].sum()
        st.dataframe(summary, width="stretch",
                     column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                    "% of products": st.column_config.NumberColumn(format="percent"),
                                    "% of revenue": st.column_config.NumberColumn(format="percent")})

        st.info(f"💡 Only **{share_a:.1%}** of products generate 80% of revenue. "
                "A-class items should never be out of stock (supply chain priority); "
                "C-class items are candidates for range reduction or on-demand ordering.")

        with st.expander("A-class product list"):
            st.dataframe(products[products["Class"] == "A"][["Description", "Revenue", "CumShare"]],
                         hide_index=True, width="stretch",
                         column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                        "CumShare": st.column_config.NumberColumn("Cumulative share",
                                                                                  format="percent")})

    # ------------------------------------------------------------------
    # 3) Anomaly detection on daily / weekly sales
    # ------------------------------------------------------------------
    with tab3:
        c1, c2, c3 = st.columns(3)
        level = c1.radio("Granularity", ["Daily", "Weekly"], horizontal=True)
        method = c2.radio("Method", ["Mean ± 3 std", "IsolationForest"], horizontal=True)
        contamination = c3.slider("Expected anomaly share (IsolationForest)", 0.01, 0.10, 0.03,
                                  disabled=(method != "IsolationForest"))

        period = "D" if level == "Daily" else "W"
        series = (data.set_index("InvoiceDate").resample(period)
                  .agg({"Revenue": "sum", "InvoiceNo": "nunique"})
                  .rename(columns={"InvoiceNo": "Orders"}))
        series = series[series["Orders"] > 0].reset_index()   # skip days with no sales (holidays)

        result = detect_anomalies(series, method, contamination)
        anomalies = result[result["Anomaly"]]

        fig = px.line(result, x="InvoiceDate", y="Revenue",
                      labels={"InvoiceDate": "", "Revenue": "Revenue (₺)"})
        fig.add_scatter(x=anomalies["InvoiceDate"], y=anomalies["Revenue"], mode="markers",
                        marker=dict(color="red", size=10), name="Anomaly")
        if method == "Mean ± 3 std":
            fig.add_hline(y=result["Upper"].iloc[0], line_dash="dot", line_color="gray",
                          annotation_text="mean + 3 std")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

        st.write(f"**{len(anomalies)}** unusual {level.lower()} periods found.")
        show_cols = anomalies[["InvoiceDate", "Revenue", "Orders"]].copy()
        show_cols["vs. average"] = show_cols["Revenue"] / result["Revenue"].mean() - 1
        st.dataframe(show_cols.sort_values("Revenue", ascending=False), hide_index=True,
                     width="stretch",
                     column_config={"InvoiceDate": st.column_config.DateColumn("Date"),
                                    "Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                    "vs. average": st.column_config.NumberColumn(format="percent")})
        st.caption("Mean ± 3 std only catches extreme spikes/drops in revenue. "
                   "IsolationForest also looks at the number of orders, so it can flag "
                   "periods where few orders produced unusually high revenue (bulk orders).")
