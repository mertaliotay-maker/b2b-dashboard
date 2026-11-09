import streamlit as st


def kpis(data):
    """Calculate the main KPIs for a (filtered) DataFrame."""
    revenue = data["Revenue"].sum()
    orders = data["InvoiceNo"].nunique()      # one invoice = one order
    customers = data["CustomerID"].nunique()
    aov = revenue / orders if orders else 0   # average order value
    return revenue, orders, customers, aov


def pct_change(now, before):
    """% change vs previous period, as text for st.metric."""
    if not before:
        return None
    return f"{(now - before) / before:+.1%}"


def show(df):
    st.title("Overview")

    # --- Period selection: one month, compared with the month before ---
    months = sorted(df["YearMonth"].unique())
    selected = st.selectbox("Select month", months, index=len(months) - 1)
    i = months.index(selected)
    previous = months[i - 1] if i > 0 else None

    now = kpis(df[df["YearMonth"] == selected])
    before = kpis(df[df["YearMonth"] == previous]) if previous else (0, 0, 0, 0)

    st.caption(f"Change is compared with the previous month ({previous})." if previous
               else "No previous month in the data to compare with.")

    # --- KPI cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₺{now[0]:,.0f}", pct_change(now[0], before[0]))
    c2.metric("Orders", f"{now[1]:,}", pct_change(now[1], before[1]))
    c3.metric("Customers", f"{now[2]:,}", pct_change(now[2], before[2]))
    c4.metric("Avg. Order Value", f"₺{now[3]:,.0f}", pct_change(now[3], before[3]))