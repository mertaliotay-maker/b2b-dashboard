"""ABC-XYZ and stock policy estimates. No actual inventory data is available."""
from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd

REQUIRED = {"StockCode", "Description", "Category", "InvoiceDate", "Quantity", "Revenue"}
POLICIES = {
    "AX": "Frequent review; protect high-value stable sales.",
    "AY": "Frequent review; adjust buffers to demand variation.",
    "AZ": "High-value volatile demand; confirm orders before replenishing.",
    "BX": "Standard periodic review and modest buffers.",
    "BY": "Periodic review; monitor forecast error.",
    "BZ": "Small batches; review intermittent demand manually.",
    "CX": "Consolidate replenishment to limit handling cost.",
    "CY": "Low-cost periodic review; avoid excess stock.",
    "CZ": "Consider order-on-demand; avoid blanket stock targets."
}


def inventory_table(df: pd.DataFrame, lookback_days: int = 90,
                    lead_days: float = 14, service_level: float = .95,
                    lead_std_days: float = 0., x_limit: float = .5,
                    y_limit: float = 1.) -> tuple[pd.DataFrame, dict]:
    """Product is keyed only by StockCode, not title/category spelling.

    Daily/weekly zero-sale periods are included analytically using sum/sum-square
    moments; no 43,575 x 365 dense product/day matrix is constructed.
    Demand independence, stationarity and availability throughout the lookback
    window are assumptions, not properties proved by transaction data.
    """
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or lookback_days < 7 or lead_days <= 0 or lead_std_days < 0 or not .5 < service_level < 1 or not 0 <= x_limit < y_limit:
        raise ValueError("Invalid stock policy inputs or empty data.")
    end = pd.Timestamp(df.InvoiceDate.max()).normalize()
    available_start = pd.Timestamp(df.InvoiceDate.min()).normalize()
    start = max(available_start, end - pd.Timedelta(days=lookback_days - 1))
    data = df.loc[df.InvoiceDate.between(start, end)].copy()
    if (data.Quantity < 0).any() or (data.Revenue < 0).any() or data.StockCode.isna().any():
        raise ValueError("Inventory calculations require valid product keys and nonnegative sales.")
    days = (end-start).days + 1
    by = data.groupby("StockCode", observed=True)
    out = by.agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                 FirstSale=("InvoiceDate", "min"), LastSale=("InvoiceDate", "max"))
    # Most recent label is descriptive only; all measures retain one row/SKU.
    labels = data.sort_values("InvoiceDate").drop_duplicates("StockCode", keep="last").set_index("StockCode")
    out = out.join(labels[["Description", "Category"]])
    out = out.sort_values(["Revenue", "StockCode"], ascending=[False, True])
    out["RevenueShare"] = out.Revenue / out.Revenue.sum()
    prior_share = out.RevenueShare.cumsum() - out.RevenueShare
    # The item crossing 80%/95% stays in the class that started before the
    # threshold, so the largest product is always A (even with a one-SKU filter).
    out["ABC"] = np.select([prior_share < .80, prior_share < .95], ["A", "B"], default="C")
    daily = data.groupby(["StockCode", "InvoiceDate"], observed=True).Quantity.sum()
    d_sq = daily.pow(2).groupby(level=0).sum().reindex(out.index)
    out["MeanDaily"] = out.Units / days
    out["StdDaily"] = np.sqrt(np.maximum(0., (d_sq - out.Units.pow(2) / days) / (days-1))) if days > 1 else np.nan
    # XYZ uses complete calendar weeks only, independent of partial-week sales
    # included in the daily stock calculations and ABC revenue totals.
    first_monday = start + pd.Timedelta(days=(7-start.weekday()) % 7)
    last_sunday = end - pd.Timedelta(days=(end.weekday()+1) % 7)
    weeks = max(0, ((last_sunday - first_monday).days + 1) // 7)
    wdata = data.loc[data.InvoiceDate.between(first_monday, last_sunday)]
    weekly = wdata.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="W-SUN")], observed=True).Quantity.sum()
    w_sum = weekly.groupby(level=0).sum().reindex(out.index, fill_value=0)
    w_sq = weekly.pow(2).groupby(level=0).sum().reindex(out.index, fill_value=0)
    active = weekly.gt(0).groupby(level=0).sum().reindex(out.index, fill_value=0)
    out["ActiveWeeks"] = active
    out["MeanWeekly"] = w_sum / weeks if weeks else np.nan
    out["StdWeekly"] = np.sqrt(np.maximum(0., (w_sq-w_sum.pow(2)/weeks)/(weeks-1))) if weeks > 1 else np.nan
    out["CV"] = out.StdWeekly / out.MeanWeekly.replace(0, np.nan)
    out["ADI"] = weeks / active.replace(0, np.nan)
    out["XYZ"] = np.select([out.CV.isna(), out.CV <= x_limit, out.CV <= y_limit], ["Unclassified", "X", "Y"], default="Z")
    out["ABC_XYZ"] = out.ABC + out.XYZ
    out["Policy"] = out.ABC_XYZ.map(POLICIES).fillna("Insufficient complete-week observations.")
    z = NormalDist().inv_cdf(service_level)
    out["LeadDemand"] = out.MeanDaily * lead_days
    out["SafetyStock"] = z * np.sqrt(lead_days * out.StdDaily.pow(2) + out.MeanDaily.pow(2) * lead_std_days ** 2)
    out["SafetyStockUnits"] = np.ceil(out.SafetyStock).astype("Int64")
    out["ReorderPoint"] = np.ceil(out.LeadDemand + out.SafetyStock).astype("Int64")
    out["ShortHistory"] = ((end - out.FirstSale).dt.days + 1 < 28) | (out.ActiveWeeks < 4)
    out["Intermittent"] = out.ADI >= 1.32
    out["ManualReview"] = out.ShortHistory | out.Intermittent | (out.XYZ == "Unclassified")
    return out.reset_index(), {"start": start, "end": end, "days": days, "complete_weeks": weeks,
                              "lead_days": lead_days, "lead_std_days": lead_std_days,
                              "service_level": service_level, "z": z, "x_limit": x_limit, "y_limit": y_limit}
