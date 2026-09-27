import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.analysis import analyze_data, get_available_tickers

# -------------------------------
# PAGE CONFIG  (must be first)
# -------------------------------
st.set_page_config(page_title="Market Pulse", layout="wide")

# -------------------------------
# DARK THEME STYLE
# -------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Market Pulse Dashboard")
st.caption("Stock Price vs News Sentiment Analysis")

# -------------------------------
# SIDEBAR — STOCK PICKER
# -------------------------------
# Human-readable display names mapped to ticker keys on disk
TICKER_LABELS = {
    # Technology
    "RELIANCE":   "Reliance Industries",
    "TCS":        "TCS",
    "INFOSYS":    "Infosys",
    # Banking
    "HDFC":       "HDFC Bank",
    "ICICI":      "ICICI Bank",
    "AXISBANK":   "Axis Bank",
    "SBI":        "SBI",
    # Energy
    "ONGC":       "ONGC",
    # FMCG
    "HUL":        "Hindustan Unilever",
    "ITC":        "ITC",
    # Auto
    "MARUTI":     "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    # Pharma
    "SUNPHARMA":  "Sun Pharma",
    "DRREDDY":    "Dr Reddy's",
    # Metals & Infra
    "TATASTEEL":  "Tata Steel",
    "LT":         "L&T",
    # Telecom
    "AIRTEL":     "Bharti Airtel",
}

available_tickers = get_available_tickers()

# Build display list in the order above, only for tickers we actually have
ordered = [t for t in TICKER_LABELS if t in available_tickers]
# Append any tickers on disk not in our label map
ordered += [t for t in available_tickers if t not in TICKER_LABELS]

display_names  = [TICKER_LABELS.get(t, t) for t in ordered]

with st.sidebar:
    st.header("⚙️ Settings")
    selected_display = st.selectbox(
        "Select Stock",
        options=display_names,
        index=0,
    )
    # Reverse-lookup the ticker key from the chosen display name
    selected_ticker = ordered[display_names.index(selected_display)]

    st.markdown("---")
    st.caption(f"Ticker: **{selected_ticker}**")
    st.caption("News covers Sep 2026. Switch stocks to compare price trends and sentiment.")

# -------------------------------
# LOAD DATA FOR SELECTED STOCK
# -------------------------------

# Step 1: Score ALL headlines once per session with FinBERT.
# This cache is shared across all ticker selections — switching companies
# reuses these scores instead of re-running the model.
@st.cache_data(show_spinner="Analysing sentiment (one-time)...")
def load_scored_news() -> pd.DataFrame:
    from src.sentiment import analyze_sentiment
    return analyze_sentiment()

# Step 2: For the selected ticker, filter + merge with stock prices.
# Fast — no model inference, just pandas filtering and joining.
@st.cache_data(show_spinner="Loading stock data...")
def load(ticker: str, _scored_news: pd.DataFrame) -> pd.DataFrame:
    return analyze_data(ticker=ticker, pre_scored_news=_scored_news)

scored_news = load_scored_news()

try:
    merged_df = load(selected_ticker, scored_news)
except ValueError as e:
    st.error(str(e))
    st.stop()

if merged_df.empty:
    st.warning(f"No data found for **{selected_display}** in the current date range. "
               "Try a different stock or re-run the news fetcher.")
    st.stop()

# -------------------------------
# PREPARE VIEWS
# -------------------------------
stock_df     = merged_df.copy()
headlines_df = merged_df[["Date", "Headline", "Company", "Sentiment"]].copy()

# -------------------------------
# KPI CALCULATIONS
# -------------------------------
latest_price = merged_df["Close"].iloc[-1]

def safe_mean(df, sentiment):
    subset = df[df["Sentiment"] == sentiment]["Return"]
    return subset.mean() if not subset.empty else float("nan")

avg_pos = safe_mean(merged_df, "Positive")
avg_neg = safe_mean(merged_df, "Negative")
avg_neu = safe_mean(merged_df, "Neutral")

def fmt_pct(v):
    return f"{v:.2%}" if v == v else "N/A"   # nan-safe

# Pearson r between continuous sentiment score and return
def pearson_r(df):
    if "Sentiment_Score" not in df.columns or len(df) < 3:
        return float("nan")
    return df["Sentiment_Score"].corr(df["Return"])

def interpret_r(r):
    """Plain-English label for a Pearson r value."""
    if r != r:          return "N/A (too little data)"
    a = abs(r)
    sign = "positive" if r >= 0 else "negative"
    if a < 0.1:         return f"r = {r:.2f}  (no relationship)"
    elif a < 0.3:       return f"r = {r:.2f}  (weak {sign})"
    elif a < 0.5:       return f"r = {r:.2f}  (moderate {sign})"
    elif a < 0.7:       return f"r = {r:.2f}  (strong {sign})"
    else:               return f"r = {r:.2f}  (very strong {sign})"

corr = pearson_r(merged_df)
corr_label = interpret_r(corr)
n_headlines = len(merged_df)

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Price",       f"₹{latest_price:,.2f}")
col2.metric("Avg Return (Pos)",   fmt_pct(avg_pos))
col3.metric("Avg Return (Neg)",   fmt_pct(avg_neg))
col4.metric("Avg Return (Neu)",   fmt_pct(avg_neu))
col5.metric("Sentiment↔Return",   corr_label, delta=f"n = {n_headlines} headlines", delta_color="off")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Price", "📊 Analysis", "📰 Headlines"])

# ── TAB 1 — PRICE ──────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{selected_display} — Price Trend")

    fig = px.line(
        stock_df,
        x="Date",
        y="Close",
        title=f"{selected_display} Stock Price (Sep 2026)",
        markers=True,
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2 — ANALYSIS ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Return vs Sentiment")

    fig2 = px.box(
        merged_df,
        x="Sentiment",
        y="Return",
        color="Sentiment",
        title=f"{selected_display} — Return Distribution by Sentiment",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral":  "#3498db",
            "Negative": "#e74c3c",
        },
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter: sentiment score vs return — the raw data behind the r value
    if "Sentiment_Score" in merged_df.columns:
        st.subheader(f"Sentiment Score vs Return  —  {corr_label}")
        st.caption(
            "Each dot is one headline matched to that day's stock return. "
            "A diagonal trend (bottom-left to top-right) means sentiment predicts returns. "
            "No pattern means it doesn't — at least not with this amount of data."
        )
        fig_scatter = px.scatter(
            merged_df,
            x="Sentiment_Score",
            y="Return",
            color="Sentiment",
            hover_data=["Headline", "Date"] if "Headline" in merged_df.columns else None,
            color_discrete_map={
                "Positive": "#2ecc71",
                "Neutral":  "#3498db",
                "Negative": "#e74c3c",
            },
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#f39c12",
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            xaxis_title="Sentiment Score  (−1 = very negative, +1 = very positive)",
            yaxis_title="Stock Return that day",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Sentiment Distribution")
    sentiment_counts = merged_df["Sentiment"].value_counts()
    fig3 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.5,
        color=sentiment_counts.index,
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral":  "#3498db",
            "Negative": "#e74c3c",
        },
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3 — HEADLINES ──────────────────────────────────────────────────────
with tab3:
    st.subheader("News Headlines with Sentiment")
    st.dataframe(headlines_df, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit · Data: GNews API + yfinance · Sentiment: VADER/NLTK")
