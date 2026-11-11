import streamlit as st
from data_loader import load_data
import overview
import customers #sonradan eklendi
from filters import apply_filters #sonradan eklendi
df = apply_filters(df) #sonradan eklendi
st.set_page_config(page_title="B2B Book Distributor Dashboard", page_icon="📚", layout="wide")

df = load_data()

# Sidebar: page selection
st.sidebar.title("📚 B2B Book Distributor")
page = st.sidebar.radio(
    "Select page",
    ["Overview", "Sales Analysis", "Customers & Marketing", "Supply Chain & Forecast"],
)

if page == "Overview":
    overview.show(df)
elif page == "Sales Analysis":
    st.title("Sales Analysis")
    st.info("Coming soon")
elif page == "Customers & Marketing":
    customers.show(df) #alttaki satır zaten bu fonksiyonun içinde var o nedenle çıkarıldı.
    #st.title("Customers & Marketing")
    #st.info("Coming soon")
elif page == "Supply Chain & Forecast":
    st.title("Supply Chain & Forecast")
    st.info("Coming soon")
