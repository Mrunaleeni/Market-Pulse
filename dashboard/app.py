import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from src.analysis import analyze_data, analyze_all_companies, get_available_tickers, TICKER_SECTOR

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

# Build display list:
# First: pooled views (All companies + each sector)
# Then:  individual companies in sector order
SECTORS = sorted(set(TICKER_SECTOR.values()))
POOLED_OPTIONS = ["📊 All companies"] + [f"📂 {s}" for s in SECTORS]

ordered = [t for t in TICKER_LABELS if t in available_tickers]
ordered += [t for t in available_tickers if t not in TICKER_LABELS]
display_names = [TICKER_LABELS.get(t, t) for t in ordered]

# All selectable options = pooled views + individual stocks
all_options = POOLED_OPTIONS + display_names

with st.sidebar:
    st.header("⚙️ Settings")
    selected_option = st.selectbox(
        "Select Stock / View",
        options=all_options,
        index=0,
    )

    # Determine if this is a pooled view or a single ticker
    is_pooled    = selected_option.startswith("📊") or selected_option.startswith("📂")
    is_all       = selected_option == "📊 All companies"
    selected_sector = selected_option.replace("📂 ", "") if selected_option.startswith("📂") else None

    if is_pooled:
        selected_ticker  = None
        selected_display = selected_option
    else:
        selected_ticker  = ordered[display_names.index(selected_option)]
        selected_display = selected_option

    st.markdown("---")

    # Return window selector — this is the core of Step 2
    # Same-day return might just reflect news reporting what already happened.
    # Next-day and 3-day returns tell you if sentiment actually predicted the move.
    RETURN_OPTIONS = {
        "Same day  (ret on publication date)":  "Return",
        "Next day  (ret the day after)":        "ret_next_day",
        "3-day     (ret over next 3 days)":     "ret_3day",
    }
    selected_return_label = st.selectbox(
        "Return window",
        options=list(RETURN_OPTIONS.keys()),
        index=0,
        help="Which return to correlate against sentiment. "
             "Next-day is most meaningful for prediction."
    )
    selected_return_col = RETURN_OPTIONS[selected_return_label]

    st.markdown("---")
    st.caption(f"{'Pooled view' if is_pooled else 'Ticker: ' + str(selected_ticker)}")
    st.caption("News covers Sep 2026. Switch stocks to compare price trends and sentiment.")

# -------------------------------
# LOAD DATA FOR SELECTED STOCK
# -------------------------------
# Scores are pre-saved in news.csv by fetch_news.py — no FinBERT at startup.
# The dashboard just reads a CSV and does pandas filtering/merging.

@st.cache_data(show_spinner="Loading data...")
def load(ticker: str | None, sector: str | None = None) -> pd.DataFrame:
    if ticker is None:
        return analyze_all_companies(sector=sector)
    return analyze_data(ticker=ticker)

try:
    merged_df = load(selected_ticker, sector=selected_sector)
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

# Correlation stats: Pearson r, p-value, Spearman rho, reliability flag
def correlation_stats(df, min_n=30, return_col="Return"):
    """
    Returns a dict with:
      n        — number of data points
      r        — Pearson r  (linear correlation, -1 to +1)
      p        — p-value for r  (< 0.05 = statistically significant)
      rho      — Spearman rho  (rank correlation, more robust with small n)
      reliable — True only when n >= min_n AND p < 0.05

    return_col controls which return window to correlate against:
      "Return"       — same day
      "ret_next_day" — next trading day
      "ret_3day"     — 3 days forward
    """
    if "Sentiment_Score" not in df.columns or return_col not in df.columns:
        return {"n": len(df), "r": None, "p": None, "rho": None, "reliable": False}

    # Drop rows where the chosen return column is NaN
    # (last few rows always have NaN for lagged returns — no future data)
    clean = df[["Sentiment_Score", return_col]].dropna()
    n = len(clean)

    if n < 3:
        return {"n": n, "r": None, "p": None, "rho": None, "reliable": False}

    x = clean["Sentiment_Score"].values
    y = clean[return_col].values

    r, p       = pearsonr(x, y)
    rho, _     = spearmanr(x, y)
    reliable   = (n >= min_n) and (p < 0.05)

    return {"n": n, "r": r, "p": p, "rho": rho, "reliable": reliable}


def fmt_corr(stats):
    """Format the correlation stats into a single KPI string."""
    if stats["r"] is None:
        return "N/A (too little data)"
    r, p, rho = stats["r"], stats["p"], stats["rho"]
    a    = abs(r)
    sign = "positive" if r >= 0 else "negative"
    if a < 0.1:   strength = "no relationship"
    elif a < 0.3: strength = f"weak {sign}"
    elif a < 0.5: strength = f"moderate {sign}"
    elif a < 0.7: strength = f"strong {sign}"
    else:         strength = f"very strong {sign}"
    return f"r={r:.2f}, p={p:.2f}, ρ={rho:.2f} ({strength})"


stats          = correlation_stats(merged_df, return_col=selected_return_col)
corr_label     = fmt_corr(stats)
n_headlines    = stats["n"]

# Keep corr_label available for the scatter plot subheader below
# (uses the same variable name as before so nothing else breaks)

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Price",       f"₹{latest_price:,.2f}")
col2.metric("Avg Return (Pos)",   fmt_pct(avg_pos))
col3.metric("Avg Return (Neg)",   fmt_pct(avg_neg))
col4.metric("Avg Return (Neu)",   fmt_pct(avg_neu))
col5.metric("Sentiment↔Return",   corr_label, delta=f"n = {n_headlines} headlines", delta_color="off")

# Reliability warning — shown whenever the result can't be trusted statistically
if not stats["reliable"]:
    if stats["r"] is None:
        st.warning("⚠️ Not enough data to compute correlation. Need at least 3 headlines.")
    elif stats["n"] < 30 and stats["p"] is not None and stats["p"] >= 0.05:
        st.warning(
            f"⚠️ Low sample size (n={stats['n']}) and p={stats['p']:.2f} — "
            "not statistically significant. Collect more headlines before drawing conclusions."
        )
    elif stats["n"] < 30:
        st.warning(
            f"⚠️ Low sample size (n={stats['n']}) — result may not be reliable. "
            "Need at least 30 headlines per company for meaningful correlation."
        )
    elif stats["p"] is not None and stats["p"] >= 0.05:
        st.warning(
            f"⚠️ p={stats['p']:.2f} — not statistically significant. "
            "The correlation could easily be due to chance with this data."
        )

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price", "📊 Analysis", "🤖 ML Model", "📰 Headlines"])

# ── TAB 1 — PRICE ──────────────────────────────────────────────────────────
with tab1:
    if is_pooled and "Ticker" in merged_df.columns:
        st.subheader(f"{selected_display} — Correlation by Company")
        st.caption("Pearson r between sentiment score and same-day return, per company. "
                   "Grey bars have n < 5 (too few to interpret).")

        # Compute per-company correlation for the bar chart
        rows = []
        for t, grp in merged_df.groupby("Ticker"):
            clean = grp[["Sentiment_Score", selected_return_col]].dropna()
            n = len(clean)
            if n >= 3:
                from scipy.stats import pearsonr as _pr
                r, p = _pr(clean["Sentiment_Score"].values, clean[selected_return_col].values)
            else:
                r, p = float("nan"), float("nan")
            rows.append({
                "Ticker": t,
                "Company": TICKER_LABELS.get(t, t),
                "Sector": TICKER_SECTOR.get(t, "Other"),
                "r": r, "p": p, "n": n,
            })
        bar_df = pd.DataFrame(rows).dropna(subset=["r"]).sort_values("r")
        bar_df["color"] = bar_df.apply(
            lambda x: ("#2ecc71" if x["r"] > 0 else "#e74c3c") if x["n"] >= 5 else "#555555", axis=1
        )
        fig_bar = px.bar(
            bar_df, x="r", y="Company", orientation="h",
            color="Sector",
            hover_data=["n", "p"],
            title=f"Pearson r per company  ({selected_return_label.strip()})",
        )
        fig_bar.update_layout(template="plotly_dark", xaxis_title="Pearson r",
                              yaxis_title="", xaxis=dict(range=[-1, 1]))
        fig_bar.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.subheader(f"{selected_display} — Price Trend")
        fig = px.line(
            merged_df,
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
        y=selected_return_col,
        color="Sentiment",
        title=f"{selected_display} — Return Distribution by Sentiment  ({selected_return_label.strip()})",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral":  "#3498db",
            "Negative": "#e74c3c",
        },
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter: sentiment score vs selected return window
    if "Sentiment_Score" in merged_df.columns and selected_return_col in merged_df.columns:
        st.subheader(f"Sentiment Score vs Return  —  {corr_label}")
        st.caption(
            "Each dot is one headline matched to the selected return window. "
            "A diagonal trend means sentiment predicts returns. "
            "No pattern means it doesn't — at least not with this amount of data."
        )
        scatter_df = merged_df.dropna(subset=["Sentiment_Score", selected_return_col])
        fig_scatter = px.scatter(
            scatter_df,
            x="Sentiment_Score",
            y=selected_return_col,
            color="Sentiment",
            hover_data=["Headline", "Date"] if "Headline" in scatter_df.columns else None,
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
            yaxis_title=selected_return_label.strip(),
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

# ── TAB 3 — ML MODEL ───────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 ML Model — Honest Evaluation")
    st.caption(
        "Linear regression: does sentiment score predict stock returns? "
        "Trained on the earliest 70% of dates, tested on the remaining 30%. "
        "Compared against a baseline that always predicts the mean return."
    )

    @st.cache_data(show_spinner="Running model evaluation...")
    def run_ml():
        from src.ml_model import train_and_evaluate
        return train_and_evaluate()

    ml = run_ml()

    if ml["status"] == "insufficient_data":
        st.warning(ml["message"])
    else:
        # Split info
        st.markdown(f"**Data:** {ml['n_total']} trading days total  ·  "
                    f"Train: {ml['n_train']} days ({ml['train_date_range']})  ·  "
                    f"Test: {ml['n_test']} days ({ml['test_date_range']})")
        st.markdown(f"**Model:** return = {ml['coef']:.5f} × sentiment_score + {ml['intercept']:.5f}")
        st.markdown("---")

        # Metrics table
        metrics = {
            "Metric":       ["R² (test set)", "Directional accuracy", "MAE"],
            "Model":        [f"{ml['r2_model']:.3f}",
                             f"{ml['dir_model']:.1%}",
                             f"{ml['mae_model']:.4f}"],
            "Baseline":     [f"{ml['r2_baseline']:.3f}",
                             f"{ml['dir_baseline']:.1%}",
                             f"{ml['mae_baseline']:.4f}"],
            "Beats baseline?": [
                "✅ Yes" if ml["r2_model"]  > ml["r2_baseline"]  else "❌ No",
                "✅ Yes" if ml["dir_model"] > ml["dir_baseline"] else "❌ No",
                "✅ Yes" if ml["mae_model"] < ml["mae_baseline"] else "❌ No",
            ],
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Verdict
        if ml["beats_baseline"]:
            st.success(
                "✅ The model beats the naive baseline on R². "
                "However, with only a few weeks of data this result is not yet "
                "statistically reliable — treat it as directionally interesting, not conclusive."
            )
        else:
            st.warning(
                "❌ The model does **not** beat the naive baseline on the test set. "
                "This is expected with a small dataset and does not mean the approach is wrong. "
                "Collect more data (aim for 100+ trading days) and re-evaluate."
            )

        # Explanation of each metric
        with st.expander("What do these metrics mean?"):
            st.markdown("""
**R²** (coefficient of determination) — how much of the variation in returns does the model explain?
- `1.0` = perfect prediction
- `0.0` = no better than always predicting the mean (the baseline)
- Negative = worse than the baseline

**Directional accuracy** — did the model predict the correct direction (up vs down)?
- `50%` = random guessing
- The baseline direction is whichever way the training mean return points

**MAE** (mean absolute error) — average size of the prediction error, in return units.
- `0.005` = off by 0.5% on average
- Lower is better

**The baseline** is always predicting the mean return from the training set. Any model worth using must beat this.
            """)

# ── TAB 4 — HEADLINES ──────────────────────────────────────────────────────
with tab4:
    st.subheader("News Headlines with Sentiment")
    show_cols = [c for c in ["Date", "Ticker", "Sector", "Company", "Headline", "Sentiment", "n_headlines"]
                 if c in merged_df.columns]
    st.dataframe(merged_df[show_cols], use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit · Data: GNews API + yfinance · Sentiment: VADER/NLTK")
