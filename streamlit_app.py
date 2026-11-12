"""B2B distributor dashboard entry point."""
import streamlit as st

import customers
import overview
import sales
from data_loader import load_data
from filters import apply_filters
from pages.forecast_supply import render as render_forecast_supply

st.set_page_config(
    page_title="B2B Book Distributor Dashboard", page_icon="📚", layout="wide"
)

df = load_data()
st.sidebar.title("📚 B2B Book Distributor")
page = st.sidebar.radio(
    "Select page",
    ["Overview", "Sales Analysis", "Customers & Marketing", "Supply Chain & Forecast"],
)
df = apply_filters(df)

if page == "Overview":
    overview.show(df)
elif page == "Sales Analysis":
    sales.show(df)
elif page == "Customers & Marketing":
    customers.show(df)
else:
    render_forecast_supply(df)
