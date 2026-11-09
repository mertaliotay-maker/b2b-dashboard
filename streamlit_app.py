import streamlit as st
from data_loader import load_data

st.set_page_config(page_title="B2B Book Distributor Dashboard", page_icon="📚", layout="wide")

df = load_data()

# Sidebar: page selection
st.sidebar.title("📚 B2B Book Distributor")
page = st.sidebar.radio(
    "Select page",
    ["Overview", "Sales Analysis", "Customers & Marketing", "Supply Chain & Forecast"],
)

if page == "Overview":
    st.title("Overview")
    st.write(f"Clean dataset: {len(df):,} invoice lines")
    st.dataframe(df.head(20))
elif page == "Sales Analysis":
    st.title("Sales Analysis")
    st.info("Coming soon")
elif page == "Customers & Marketing":
    st.title("Customers & Marketing")
    st.info("Coming soon")
elif page == "Supply Chain & Forecast":
    st.title("Supply Chain & Forecast")
    st.info("Coming soon")