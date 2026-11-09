import pandas as pd
import streamlit as st

DATA_PATH = "data/B2B_Transaction_Data_.xlsx"


@st.cache_data
def load_data():
    """Read the raw Excel file, clean it and return the clean DataFrame.
    One row = one invoice line (one product on one invoice)."""

    # 1) Read. StockCode is an ISBN/barcode -> identifier, read as text.
    df = pd.read_excel(DATA_PATH, dtype={"StockCode": str})

    # 2) Strip spaces from text columns: "Dünya Politika " and "Dünya Politika"
    #    were counted as different categories. astype("string") first because
    #    some titles are numbers/dates (e.g. "1984") and would become NaN.
    for col in ["InvoiceNo", "CustomerID", "Description", "Category", "City"]:
        df[col] = df[col].astype("string").str.strip()

    # 3) Remove 6,213 rows identical in all columns (double records;
    #    there is no line id to tell two genuine lines apart).
    df = df.drop_duplicates()

    # 4) Remove 11 lines with missing or zero Quantity. The zero-quantity
    #    lines have negative prices -> price corrections, not sales.
    df = df[df["Quantity"].notna() & (df["Quantity"] > 0)]

    # 5) Remove 75 lines with price < 0.01 TL (free goods / placeholder prices).
    #    They would distort average price and discount statistics.
    df = df[(df["NetPrice"] >= 0.01) & (df["UnitPrice"] >= 0.01)]

    # 6) Missing City (25 customers) and Category (1 product) cannot be
    #    recovered from other rows -> "Unknown". Rows kept: revenue is valid.
    df["City"] = df["City"].fillna("Unknown")
    df["Category"] = df["Category"].fillna("Unknown")

    # 7) Fix data types: date stored as text, quantity stored as float.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = df["Quantity"].astype(int)

    # 8) Time columns for monthly / weekly / quarterly analysis.
    df["Year"] = df["InvoiceDate"].dt.year
    df["Quarter"] = "Q" + df["InvoiceDate"].dt.quarter.astype(str)
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Week"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    # 9) Money columns. NetPrice = discounted price actually paid,
    #    UnitPrice = list price.
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["GrossRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountRate"] = 1 - df["NetPrice"] / df["UnitPrice"]
    # First 3 letters of InvoiceNo (BAD, BSR, DPF, DPE, MUH) -> sales channel proxy
    df["InvoiceSeries"] = df["InvoiceNo"].str[:3]

    return df.reset_index(drop=True)