"""Steps:
1. RFM per customer (Recency, Frequency, Monetary).
2. KMeans segmentation; number of clusters chosen with elbow + silhouette.
3. Segments get business names (Loyal / Potential / At Risk / Lost).
4. Segment charts, top-10 customers, category preference, basket analysis.
5. A marketing action for each segment.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from theme import SEGMENT_COLORS, insight, style, tl

N_CLUSTERS = 4          # chosen with the elbow + silhouette tab (see explanation there)
K_RANGE = range(2, 9)   # k values tested for elbow / silhouette
SEGMENT_ORDER = ["Loyal", "Potential", "At Risk", "Lost"]
# Streamlit text colours matching theme.SEGMENT_COLORS
BADGE_COLORS = {"Loyal": "blue", "Potential": "green", "At Risk": "orange", "Lost": "red"}


# ---------------------------------------------------------------------------
# 1) RFM table
# ---------------------------------------------------------------------------
def build_rfm(df):
    """One row per customer.
    Recency   = days since last purchase, counted from the day after the last
                date in the (filtered) data, so the most recent buyer has 1.
    Frequency = number of distinct invoices (one invoice = one order).
    Monetary  = total net revenue (after discount)."""
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        GrossRevenue=("GrossRevenue", "sum"),
        City=("City", "first"),           # every customer has exactly one city
    )
    rfm["Recency"] = (ref_date - rfm["LastPurchase"]).dt.days
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]   # average order value

    # Favourite category = category with the highest revenue for that customer
    cat_rev = df.groupby(["CustomerID", "Category"])["Revenue"].sum().reset_index()
    fav = cat_rev.loc[cat_rev.groupby("CustomerID")["Revenue"].idxmax()]
    rfm["FavCategory"] = fav.set_index("CustomerID")["Category"]
    return rfm


def scale_features(rfm):
    """RFM values are very skewed (median revenue ₺4K, max ₺1.5M). Without a log,
    KMeans would only separate the few giant customers. log1p + StandardScaler puts
    R, F and M on the same scale so each has equal weight in the distance."""
    X = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# 2) Choosing k and clustering (cached: only the small RFM table is hashed)
# ---------------------------------------------------------------------------
@st.cache_data
def k_scores(rfm):
    """Inertia (for the elbow plot) and silhouette score for each k."""
    X = scale_features(rfm)
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_data
def segment_customers(rfm):
    """KMeans with N_CLUSTERS, then give each cluster a business name."""
    X = scale_features(rfm)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit(X)
    out = rfm.copy()
    out["Cluster"] = km.labels_
    out["Segment"] = out["Cluster"].map(name_clusters(out))
    return out


def name_clusters(rfm):
    """KMeans numbers clusters randomly (0,1,2,3), so names are given by rules
    on each cluster's median R, F, M instead of by cluster number:
      - highest median revenue      -> Loyal     (buy often, recently, a lot)
      - highest median recency      -> Lost      (have not bought for months)
      - of the other two, the more recent one -> Potential (active, still small)
      - the remaining one           -> At Risk   (were buying, now slipping away)"""
    prof = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].median()
    names = {prof["Monetary"].idxmax(): "Loyal"}
    rest = prof.drop(index=list(names))
    names[rest["Recency"].idxmax()] = "Lost"
    rest = rest.drop(index=rest["Recency"].idxmax()).sort_values("Recency")
    names[rest.index[0]] = "Potential"
    names[rest.index[1]] = "At Risk"
    return names


def segment_summary(seg):
    s = seg.groupby("Segment").agg(
        Customers=("Recency", "size"),
        Revenue=("Monetary", "sum"),
        Gross=("GrossRevenue", "sum"),
        Orders=("Frequency", "sum"),
        MedianRecency=("Recency", "median"),
        MedianFrequency=("Frequency", "median"),
    ).reindex(SEGMENT_ORDER)
    s["CustomerShare"] = s["Customers"] / s["Customers"].sum()
    s["RevenueShare"] = s["Revenue"] / s["Revenue"].sum()
    s["AOV"] = s["Revenue"] / s["Orders"]
    s["AvgDiscount"] = 1 - s["Revenue"] / s["Gross"]   # revenue-weighted discount
    return s


# ---------------------------------------------------------------------------
# 3) Basket analysis on categories
# ---------------------------------------------------------------------------
@st.cache_data
def category_pairs(inv_cat):
    """Which categories are bought together on the same invoice?
    Done on categories, not single books: there are 43K books, so most book
    pairs are too rare to say anything; 306 categories give stable results.
      support = share of invoices that contain both categories
      lift    = how many times more often they appear together than by chance
                (lift > 1 -> real association)"""
    n_inv = inv_cat["InvoiceNo"].nunique()
    count = inv_cat["Category"].value_counts()
    pairs = inv_cat.merge(inv_cat, on="InvoiceNo")
    pairs = pairs[pairs["Category_x"] < pairs["Category_y"]]   # each pair once
    pairs = pairs.groupby(["Category_x", "Category_y"]).size().rename("Invoices").reset_index()
    pairs["Support"] = pairs["Invoices"] / n_inv
    pairs["Lift"] = (pairs["Invoices"] * n_inv
                     / (pairs["Category_x"].map(count) * pairs["Category_y"].map(count)))
    return pairs.rename(columns={"Category_x": "Category A", "Category_y": "Category B"})


# ---------------------------------------------------------------------------
# 4) Marketing actions
# ---------------------------------------------------------------------------
ACTIONS = {
    "Loyal": ("Keep & grow",
              "VIP programme: dedicated sales rep, early access to new releases, "
              "volume-based tiered discount instead of extra general discounts "
              "(they already buy without them)."),
    "Potential": ("Increase frequency",
                  "Cross-sell from their favourite category, loyalty points per order, "
                  "free shipping above a basket threshold to grow order value."),
    "At Risk": ("Win back now",
                "Time-limited win-back offer (extra discount for the next order), "
                "a phone call from the sales rep to ask why orders dropped."),
    "Lost": ("Low-cost reactivation",
             "Cheap channels only (e-mail / newsletter with new catalogue). "
             "Do not spend sales-rep time unless they reply."),
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def show(df):
    st.title("Customers & Marketing")
    st.caption("Customers are grouped by RFM (Recency, Frequency, Monetary) with "
               "KMeans clustering. All numbers follow the sidebar filters.")

    rfm = build_rfm(df)
    if len(rfm) < 20:
        st.warning("Too few customers in the current filter for segmentation "
                   f"({len(rfm)}). Please widen the filters.")
        st.dataframe(rfm)
        return

    seg = segment_customers(rfm)
    summary = segment_summary(seg)

    # --- KPI row: one card per segment ---
    cols = st.columns(len(SEGMENT_ORDER))
    for col, name in zip(cols, SEGMENT_ORDER):
        r = summary.loc[name]
        col.metric(f"{name} customers", f"{int(r.Customers):,}",
                   f"{r.RevenueShare:.0%} of revenue", delta_color="off")

    tabs = st.tabs(["Segments", "Choosing k", "Top customers",
                    "Category preference", "Basket analysis", "Marketing actions"])

    with tabs[0]:
        show_segments(seg, summary)
    with tabs[1]:
        show_k_choice(rfm)
    with tabs[2]:
        show_top_customers(seg)
    with tabs[3]:
        show_category_preference(df, seg)
    with tabs[4]:
        show_basket(df)
    with tabs[5]:
        show_actions(summary)


def show_segments(seg, summary):
    # Customer share vs revenue share side by side
    shares = summary[["CustomerShare", "RevenueShare"]].reset_index().melt(
        id_vars="Segment", var_name="Measure", value_name="Share")
    shares["Measure"] = shares["Measure"].map(
        {"CustomerShare": "% of customers", "RevenueShare": "% of revenue"})
    fig = px.bar(shares, x="Segment", y="Share", color="Measure", barmode="group",
                 text_auto=".0%", title="Share of customers vs share of revenue")
    fig.update_yaxes(tickformat=".0%", title=None)
    st.plotly_chart(style(fig))

    loyal, risk = summary.loc["Loyal"], summary.loc["At Risk"]
    insight(f"Loyal customers are {loyal.CustomerShare:.0%} of customers but bring "
            f"{loyal.RevenueShare:.0%} of revenue ({tl(loyal.Revenue)}). Losing even a "
            f"few of them hurts more than losing the whole Lost segment, so they need "
            f"personal attention first.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(summary.reset_index(), x="Segment", y="AOV", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, text_auto=",.0f",
                     title="Average order value (₺)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 340))
    with c2:
        fig = px.bar(summary.reset_index(), x="Segment", y="MedianRecency",
                     color="Segment", color_discrete_map=SEGMENT_COLORS, text_auto=".0f",
                     title="Median days since last order")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 340))

    insight(f"At Risk customers last ordered a median {risk.MedianRecency:.0f} days ago "
            f"and produced {tl(risk.Revenue)} so far. This is the revenue a win-back "
            f"campaign can still save.")

    # Scatter: every dot is a customer
    fig = px.scatter(seg.reset_index(), x="Recency", y="Monetary", color="Segment",
                     size="Frequency", size_max=25, log_y=True,
                     color_discrete_map=SEGMENT_COLORS,
                     category_orders={"Segment": SEGMENT_ORDER},
                     hover_data={"CustomerID": True, "City": True, "Frequency": True},
                     labels={"Recency": "Days since last order",
                             "Monetary": "Total revenue (₺, log scale)"},
                     title="Each dot is a customer (dot size = number of orders)")
    st.plotly_chart(style(fig, 450))
    st.caption("Revenue is on a log scale because a few customers buy hundreds of "
               "times more than the median customer.")

    # Detailed table
    table = summary[["Customers", "Revenue", "AOV", "MedianRecency",
                     "MedianFrequency", "AvgDiscount"]].reset_index()
    table[["Revenue", "AOV"]] = table[["Revenue", "AOV"]].round().astype(int)
    st.dataframe(table, hide_index=True, column_config={
        "Revenue": st.column_config.NumberColumn("Revenue (₺)", format="localized"),
        "AOV": st.column_config.NumberColumn("Avg order (₺)", format="localized"),
        "MedianRecency": st.column_config.NumberColumn("Median recency (days)", format="%.0f"),
        "MedianFrequency": st.column_config.NumberColumn("Median orders", format="%.0f"),
        "AvgDiscount": st.column_config.NumberColumn(
            "Avg discount", format="percent",
            help="Revenue-weighted: 1 - net revenue / list-price revenue"),
    })


def show_k_choice(rfm):
    scores = k_scores(rfm)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(scores, x="k", y="Inertia", markers=True,
                      title="Elbow method (lower = tighter clusters)")
        fig.add_vline(x=N_CLUSTERS, line_dash="dash", line_color="gray")
        st.plotly_chart(style(fig, 330))
    with c2:
        fig = px.line(scores, x="k", y="Silhouette", markers=True,
                      title="Silhouette score (higher = better separated)")
        fig.add_vline(x=N_CLUSTERS, line_dash="dash", line_color="gray")
        st.plotly_chart(style(fig, 330))

    best = scores.loc[scores["Silhouette"].idxmax()]
    chosen = scores.set_index("k").loc[N_CLUSTERS]
    insight(f"Silhouette is highest at k={int(best.k)} ({best.Silhouette:.2f}), but two "
            f"groups (good vs. bad customers) are too rough for marketing. The elbow bends "
            f"at k=3–4, and k={N_CLUSTERS} still has an acceptable silhouette "
            f"({chosen.Silhouette:.2f}) while splitting the middle customers into "
            f"'Potential' and 'At Risk', who need opposite actions. So we use k={N_CLUSTERS}.")
    st.caption("Features: log(1+Recency), log(1+Frequency), log(1+Monetary), standardised. "
               "KMeans with 10 initialisations, random_state=42 for reproducible results.")


def show_top_customers(seg):
    top = seg.sort_values("Monetary", ascending=False).head(10).reset_index()
    top_share = top["Monetary"].sum() / seg["Monetary"].sum()
    top[["Monetary", "AOV"]] = top[["Monetary", "AOV"]].round().astype(int)
    st.subheader("Top 10 customers by revenue")
    st.dataframe(
        top[["CustomerID", "City", "Segment", "Monetary", "Frequency", "AOV",
             "LastPurchase", "FavCategory"]],
        hide_index=True,
        column_config={
            "Monetary": st.column_config.NumberColumn("Revenue (₺)", format="localized"),
            "Frequency": st.column_config.NumberColumn("Orders"),
            "AOV": st.column_config.NumberColumn("Avg order (₺)", format="localized"),
            "LastPurchase": st.column_config.DateColumn("Last order"),
            "FavCategory": "Top category",
        })
    insight(f"The top 10 customers (out of {len(seg):,}) bring {top_share:.0%} of "
            f"revenue. Each should have a named account manager; a delay or stock "
            f"problem for any of them is visible in total revenue.")

    # Top cities
    city = seg.groupby("City").agg(Customers=("Recency", "size"),
                                   Revenue=("Monetary", "sum"))
    city = city.sort_values("Revenue", ascending=False).head(10).reset_index()
    fig = px.bar(city, x="Revenue", y="City", orientation="h", text_auto=",.0f",
                 hover_data=["Customers"], title="Top 10 cities by revenue (₺)")
    fig.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(style(fig, 400))
    first = city.iloc[0]
    insight(f"{first.City} alone brings {first.Revenue / seg['Monetary'].sum():.0%} of "
            f"revenue from {int(first.Customers)} customers; regional campaigns should "
            f"start there.")


def show_category_preference(df, seg):
    # Revenue share of the top categories inside each segment
    lines = df[["CustomerID", "Category", "Revenue"]].merge(
        seg["Segment"], left_on="CustomerID", right_index=True)
    top_cats = lines.groupby("Category")["Revenue"].sum().nlargest(10).index
    mix = (lines[lines["Category"].isin(top_cats)]
           .groupby(["Segment", "Category"])["Revenue"].sum())
    mix = mix / lines.groupby("Segment")["Revenue"].sum()   # share within segment
    heat = mix.unstack("Category").reindex(SEGMENT_ORDER)[list(top_cats)]

    fig = px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                    title="Share of each segment's revenue in the top 10 categories")
    fig.update_xaxes(title=None, tickangle=-35)
    fig.update_yaxes(title=None)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style(fig, 380))

    # Category with the biggest difference between Loyal and Potential customers
    diff = (heat.loc["Potential"] - heat.loc["Loyal"]).sort_values()
    insight(f"Potential customers spend {heat.loc['Potential', diff.index[-1]]:.0%} of "
            f"their money on '{diff.index[-1]}' vs {heat.loc['Loyal', diff.index[-1]]:.0%} "
            f"for Loyal customers. Cross-sell offers for Potential customers should start "
            f"from this category and add the categories Loyal customers buy more of "
            f"(e.g. '{diff.index[0]}').")


def show_basket(df):
    inv_cat = df[["InvoiceNo", "Category"]].drop_duplicates()
    pairs = category_pairs(inv_cat)
    n_inv = inv_cat["InvoiceNo"].nunique()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most frequent pairs")
        st.dataframe(pairs.nlargest(10, "Invoices"), hide_index=True,
                     column_config={"Support": st.column_config.NumberColumn(format="percent"),
                                    "Lift": st.column_config.NumberColumn(format="%.2f")})
    with c2:
        st.subheader("Strongest associations")
        # Minimum number of invoices so rare pairs with random high lift are left out
        min_inv = max(30, int(n_inv * 0.005))
        strong = pairs[pairs["Invoices"] >= min_inv].nlargest(10, "Lift")
        st.dataframe(strong, hide_index=True,
                     column_config={"Support": st.column_config.NumberColumn(format="percent"),
                                    "Lift": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Only pairs on at least {min_inv:,} invoices.")

    if pairs.empty:
        return
    top = pairs.loc[pairs["Invoices"].idxmax()]
    msg = (f"'{top['Category A']}' and '{top['Category B']}' are on the same invoice in "
           f"{top.Support:.0%} of orders: a good default bundle.")
    if not strong.empty:
        s = strong.iloc[0]
        msg += (f" '{s['Category A']}' + '{s['Category B']}' appear together "
                f"{s.Lift:.1f}× more often than by chance: when a customer orders one, "
                f"the sales rep should suggest the other.")
    insight(msg)


def show_actions(summary):
    st.subheader("What to do with each segment")
    for name in SEGMENT_ORDER:
        r = summary.loc[name]
        goal, action = ACTIONS[name]
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### :{BADGE_COLORS[name]}[{name}]")
            c1.caption(f"{int(r.Customers):,} customers · {r.RevenueShare:.0%} of revenue")
            c2.markdown(f"**Goal: {goal}**")
            c2.write(action)
            c2.caption(f"Avg order {tl(r.AOV)} · median {r.MedianRecency:.0f} days since "
                       f"last order · avg discount {r.AvgDiscount:.0%}")

    risk = summary.loc["At Risk"]
    insight(f"Priority order for the sales team: Loyal (protect {tl(summary.loc['Loyal'].Revenue)}), "
            f"then At Risk ({int(risk.Customers)} customers, {tl(risk.Revenue)} revenue to win "
            f"back). Lost customers get only automated e-mails.")
