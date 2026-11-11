"""Shared look for the whole app: colour palette, Plotly defaults, small UI helpers.
Every page imports this file so all charts look the same."""
import plotly.express as px
import streamlit as st

# Main palette (used in order for categories without a fixed colour)
PALETTE = ["#2E5EAA", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
           "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]

# Fixed colours for customer segments, so a segment has the same colour everywhere
SEGMENT_COLORS = {
    "Loyal": "#2E5EAA",
    "Potential": "#59A14F",
    "At Risk": "#F28E2B",
    "Lost": "#E15759",
}

# Plotly Express defaults: every px chart in the app gets the palette and template
px.defaults.color_discrete_sequence = PALETTE
px.defaults.template = "plotly_white"

def style(fig, height=380):
    """Same size, margins and legend position for every chart.
    Title at the very top, legend in its own row under the title, so they never overlap."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=90, b=10),           # t=50 iken üst üste bindi 90 olarak değiştirdik
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    title_text=""),                    # hide legend title ("Measure", "Segment") az önce üst üste binenleri komple kaldırdık
        hoverlabel=dict(font_size=13),
    )
    return fig

def insight(text):
    """'What do we see, what should we do' box under a chart."""
    st.info(f"💡 {text}")


def tl(x):
    """Format a number as Turkish lira, e.g. ₺1,234,567."""
    return f"₺{x:,.0f}"
