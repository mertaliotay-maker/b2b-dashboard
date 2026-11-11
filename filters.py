"""Shared sidebar filters. streamlit_app.py calls apply_filters(df) once and passes
the filtered DataFrame to every page, so all pages show the same filtered data."""
import pandas as pd
import streamlit as st


def apply_filters(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # Date range (default = whole period in the data)
    min_d, max_d = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    dates = st.sidebar.date_input("Date range", (min_d, max_d),
                                  min_value=min_d, max_value=max_d)
    # While the user is picking, date_input returns only one date -> wait for both
    start, end = (dates if len(dates) == 2 else (min_d, max_d))

    # Empty selection = no filter (all values)
    categories = st.sidebar.multiselect(
        "Category", sorted(df["Category"].unique()),
        placeholder="All categories")
    cities = st.sidebar.multiselect(
        "City", sorted(df["City"].unique()), placeholder="All cities")
    series = st.sidebar.multiselect(
        "Invoice series (channel)", sorted(df["InvoiceSeries"].unique()),
        placeholder="All series",
        help="First 3 letters of the invoice number, used as a sales-channel proxy.")

    mask = df["InvoiceDate"].between(pd.Timestamp(start), pd.Timestamp(end))
    if categories:
        mask &= df["Category"].isin(categories)
    if cities:
        mask &= df["City"].isin(cities)
    if series:
        mask &= df["InvoiceSeries"].isin(series)

    out = df[mask]
    st.sidebar.caption(f"{len(out):,} of {len(df):,} invoice lines selected")
    if out.empty:
        st.warning("No data for the selected filters. Please widen the filters.")
        st.stop()
    return out
