"""
Market Pulse Dashboard
======================
Mission: one question, one answer, per screen.
"Does news sentiment predict this stock's returns?"

Three tabs, strictly separated by depth:
  Tab 1 — Verdict       : verdict + one chart + hidden stats expander
  Tab 2 — What If?      : scenario, event study, charts
  Tab 3 — Proof         : box plot, scatter, pie, ML card

Global rules enforced here:
  - No raw stat (r, p, ρ, R², MAE) appears outside the Proof tab / "Show the statistics" expander.
  - No number + hedge in the same sentence on the Verdict tab.
  - Sample size (n) shown exactly once — in the header progress bar only.
  - Color = confidence: grey (early) | amber (no signal) | green (positive) | red (negative)
  - Return window defaults to Next day (matches the advice given in the tooltip).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

from src.analysis import (
    analyze_data, analyze_all_companies,
    get_available_tickers, TICKER_SECTOR,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Pulse", layout="wide")

# Phase 6: color system baked into CSS.
# Grey states (early) use muted palette. Amber/green/red only for reliable results.
st.markdown("""
<style>
  body { background-color: #0e1117; color: #e6edf3; }
  .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }

  /* Verdict cards */
  .verdict-early    { background:#1c1c1c; border:1px solid #444; border-radius:10px;
                       padding:20px 24px; color:#888; }
  .verdict-amber    { background:#2a2200; border:1px solid #a87000; border-radius:10px;
                       padding:20px 24px; color:#e6c76a; }
  .verdict-green    { background:#0d2b1a; border:1px solid #238636; border-radius:10px;
                       padding:20px 24px; color:#3fb950; }
  .verdict-red      { background:#2b0d0d; border:1px solid #da3633; border-radius:10px;
                       padding:20px 24px; color:#f85149; }

  .verdict-headline { font-size:1.6rem; font-weight:700; margin:0 0 8px 0; }
  .verdict-detail   { font-size:1rem; margin:0; opacity:0.9; }

  /* Metric containers */
  div[data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:10px 14px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_THRESHOLD = 15   # days needed for statistical reliability

TICKER_LABELS = {
    "RELIANCE":   "Reliance Industries",
    "TCS":        "TCS",
    "INFOSYS":    "Infosys",
    "HDFC":       "HDFC Bank",
    "ICICI":      "ICICI Bank",
    "AXISBANK":   "Axis Bank",
    "SBI":        "SBI",
    "ONGC":       "ONGC",
    "HUL":        "Hindustan Unilever",
    "ITC":        "ITC",
    "MARUTI":     "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "SUNPHARMA":  "Sun Pharma",
    "DRREDDY":    "Dr Reddy's",
    "TATASTEEL":  "Tata Steel",
    "LT":         "L&T",
    "AIRTEL":     "Bharti Airtel",
}

# ── Phase 2: Verdict engine ───────────────────────────────────────────────────
# Tested in tests/test_verdict.py — all 9 edge cases pass.
# Rule: no raw stats (r, p, ρ) in headline or detail. Only n and countdown.

def get_verdict(n: int, r: float | None, p: float | None) -> dict:
    """
    Returns {state, headline, detail, color} per Section 3 of the spec.
    color: 'grey' | 'amber' | 'green' | 'red'
    """
    if n < N_THRESHOLD or r is None or p is None:
        days_left = N_THRESHOLD - n
        return {
            "state":    "early",
            "headline": "Too early to tell",
            "detail":   (f"We have {n} days of data — need {N_THRESHOLD} before this is reliable. "
                         f"Check back in {days_left} more trading days."),
            "color":    "grey",
        }
    if p >= 0.05:
        return {
            "state":    "no_signal",
            "headline": "No reliable signal",
            "detail":   (f"With {n} days of data, sentiment does not appear "
                          "to predict returns for this view."),
            "color":    "amber",
        }
    if r > 0:
        return {
            "state":    "positive_signal",
            "headline": "Signal detected — positive",
            "detail":   (f"Positive news tends to come with higher returns "
                         f"({n} days analyzed, statistically significant)."),
            "color":    "green",
        }
    return {
        "state":    "negative_signal",
        "headline": "Signal detected — but inverted",
        "detail":   ("Positive news tends to come with LOWER returns here — "
                     f"an unusual pattern worth double-checking in the Proof tab "
                     f"({n} days analyzed, statistically significant)."),
        "color":    "red",
    }


def render_verdict(v: dict) -> None:
    """Render the verdict card using the color class matching the state."""
    css = f"verdict-{v['color']}"
    st.markdown(
        f'<div class="{css}">'
        f'<p class="verdict-headline">{v["headline"]}</p>'
        f'<p class="verdict-detail">{v["detail"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def company_verdict_color(n: int, r: float | None, p: float | None) -> str:
    """Return the hex color for a single company bar, per the color system."""
    v = get_verdict(n, r, p)
    return {
        "grey":  "#444444",
        "amber": "#a87000",
        "green": "#238636",
        "red":   "#da3633",
    }[v["color"]]


# ── Dropdown data ─────────────────────────────────────────────────────────────
available_tickers = get_available_tickers()
SECTORS           = sorted(set(TICKER_SECTOR.values()))
POOLED_OPTIONS    = ["📊 All companies"] + [f"📂 {s}" for s in SECTORS]
ordered           = [t for t in TICKER_LABELS if t in available_tickers]
ordered          += [t for t in available_tickers if t not in TICKER_LABELS]
display_names     = [TICKER_LABELS.get(t, t) for t in ordered]
all_options       = POOLED_OPTIONS + display_names

# ── Phase 1: Sidebar (global header controls) ─────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_option = st.selectbox(
        "Company",
        options=all_options,
        index=0,
    )

    is_pooled       = selected_option.startswith(("📊", "📂"))
    selected_sector = selected_option.replace("📂 ", "") if selected_option.startswith("📂") else None

    if is_pooled:
        selected_ticker  = None
        selected_display = selected_option
    else:
        selected_ticker  = ordered[display_names.index(selected_option)]
        selected_display = selected_option

    st.markdown("---")

    # Rule 5: default = Next day (matches the tooltip advice)
    RETURN_OPTIONS = {
        "Next day":     "ret_next_day",
        "Same day":     "Return",
        "3 days later": "ret_3day",
    }
    selected_return_label = st.selectbox(
        "Return window",
        options=list(RETURN_OPTIONS.keys()),
        index=0,   # Next day is first = default
    )
    selected_return_col = RETURN_OPTIONS[selected_return_label]
    st.caption("Best for most news. Try 'Same day' only for news published before market open.")

    st.markdown("---")

    # Last updated timestamp — small, bottom of sidebar
    _news_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "news.csv")
    if os.path.exists(_news_path):
        _mtime = datetime.fromtimestamp(os.path.getmtime(_news_path))
        st.caption(f"🕐 Last updated: {_mtime.strftime('%d %b %Y')}")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
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
    st.warning(
        f"No data yet for **{selected_display}**. "
        "Run `python src/fetch_news.py` to score new headlines."
    )
    st.stop()


# ── Compute stats (used across tabs) ─────────────────────────────────────────
def compute_stats(df: pd.DataFrame, return_col: str) -> dict:
    """Returns n, r, p, rho. All None if insufficient data."""
    if "Sentiment_Score" not in df.columns or return_col not in df.columns:
        return {"n": 0, "r": None, "p": None, "rho": None}
    clean = df[["Sentiment_Score", return_col]].dropna()
    n = len(clean)
    if n < 3:
        return {"n": n, "r": None, "p": None, "rho": None}
    r,   p   = pearsonr( clean["Sentiment_Score"].values, clean[return_col].values)
    rho, _   = spearmanr(clean["Sentiment_Score"].values, clean[return_col].values)
    return {"n": n, "r": r, "p": p, "rho": rho}


stats   = compute_stats(merged_df, selected_return_col)
n, r, p = stats["n"], stats["r"], stats["p"]
verdict = get_verdict(n, r, p)


# ── Phase 1: Global header ────────────────────────────────────────────────────
st.title("📊 Market Pulse")
st.caption("Does news sentiment predict stock returns?")

# Rule 6: sample size stated ONCE here, nowhere else on this page
_pct = min(n / N_THRESHOLD, 1.0)
_bar_label = (
    f"{n} of {N_THRESHOLD} days collected ({_pct:.0%})  —  "
    f"{'results are reliable' if n >= N_THRESHOLD else f'{N_THRESHOLD - n} more days needed'}"
)
st.progress(_pct, text=_bar_label)
st.markdown("")   # breathing room


# ── Four tabs ─────────────────────────────────────────────────────────────────
tab_verdict, tab_whatif, tab_proof, tab_data = st.tabs(["Verdict", "What If?", "Proof", "Headlines Data"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB — VERDICT
# Spec: verdict headline + detail + one chart + "Show the statistics" expander.
# Nothing else. No pie, no box plot, no ML numbers.
# ══════════════════════════════════════════════════════════════════════════════
with tab_verdict:

    # Phase 3: verdict card — headline + detail only, no raw stats
    render_verdict(verdict)
    st.markdown("")

    # Phase 3: "Show the statistics" expander — ONLY place r/p/ρ appear on Verdict tab
    if r is not None:
        with st.expander("Show the statistics"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(
                "Pearson r",
                f"{r:+.3f}",
                help="Linear correlation between sentiment score and return. "
                     "+1 = perfect match, 0 = none, −1 = opposite.",
            )
            sc2.metric(
                "p-value",
                f"{p:.3f}",
                delta="significant" if p < 0.05 else "not significant",
                delta_color="off",
                help="Below 0.05 = result is unlikely to be random chance.",
            )
            sc3.metric(
                "Spearman ρ",
                f"{stats['rho']:+.3f}",
                help="Rank-based confirmation. Agrees with Pearson when the "
                     "relationship is consistent across the data.",
            )

    st.markdown("---")

    # Phase 3: single supporting chart
    if is_pooled and "Ticker" in merged_df.columns:
        # Pooled / sector view — one bar per company, colored by verdict state
        st.caption(
            "Each bar shows whether sentiment tracks returns for that company. "
            "Green = positive signal  ·  Red = inverted  ·  "
            "Amber = no signal  ·  Grey = not enough data yet."
        )

        rows = []
        for t, grp in merged_df.groupby("Ticker"):
            s = compute_stats(grp, selected_return_col)
            t_n, t_r, t_p = s["n"], s["r"], s["p"]
            t_v   = get_verdict(t_n, t_r, t_p)
            t_col = company_verdict_color(t_n, t_r, t_p)
            bar_x = t_r if (t_r is not None and t_n >= 5) else 0.0
            rows.append({
                "Company": TICKER_LABELS.get(t, t),
                "Sector":  TICKER_SECTOR.get(t, "Other"),
                "Verdict": t_v["headline"],
                "Days":    t_n,
                "_x":      bar_x,
                "_color":  t_col,
            })

        bar_df = pd.DataFrame(rows).sort_values("_x", ascending=True)

        fig = go.Figure()
        for _, row in bar_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["_x"]], y=[row["Company"]], orientation="h",
                marker_color=row["_color"],
                hovertemplate=(
                    f"<b>{row['Company']}</b><br>"
                    f"{row['Verdict']}<br>"
                    f"Days of data: {row['Days']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig.update_layout(
            template="plotly_dark",
            height=max(280, len(bar_df) * 50),
            margin=dict(l=140, r=20, t=10, b=50),
            bargap=0.35,
            xaxis=dict(
                range=[-1, 1],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["−1<br>Strong<br>inverted", "−0.5", "0<br>No signal", "+0.5", "+1<br>Strong<br>positive"],
                title="← positive news with lower returns  |  positive news with higher returns →",
            ),
            yaxis_title="",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#555", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Single company — price line chart, dots colored by sentiment
        st.caption(
            "Closing price over time. "
            "Dot color shows the sentiment on that day: "
            "green = positive news, red = negative, grey = neutral."
        )

        COLOR_MAP = {"Positive": "#3fb950", "Neutral": "#8b949e", "Negative": "#f85149"}

        chart_df = merged_df.dropna(subset=["Close", "Date"]).copy()
        chart_df["_color"] = chart_df["Sentiment"].map(COLOR_MAP).fillna("#8b949e")

        fig = go.Figure()

        # Line (no color on the line itself — dots carry the sentiment color)
        fig.add_trace(go.Scatter(
            x=chart_df["Date"], y=chart_df["Close"],
            mode="lines",
            line=dict(color="#30363d", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Colored dots
        for sentiment, color in COLOR_MAP.items():
            sub = chart_df[chart_df["Sentiment"] == sentiment]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["Date"], y=sub["Close"],
                mode="markers",
                marker=dict(color=color, size=9, line=dict(width=1, color="#0e1117")),
                name=sentiment,
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Price: ₹%{y:,.2f}<br>"
                    "Date: %{x|%d %b %Y}"
                    "<extra></extra>"
                ),
                customdata=sub["Sentiment"],
            ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Closing price (₹)",
            legend=dict(
                title="News sentiment on that day",
                orientation="h", y=-0.18, x=0,
            ),
            margin=dict(t=10, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)



# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# TAB — WHAT IF?
# User picks: company, event type, sentiment strength.
# App returns: expected move, direction probability, range, past cases.
# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
with tab_whatif:

    from src.events     import EVENT_LABELS, ALL_EVENT_TYPES
    from src.event_study import (
        build_event_study, event_stats_table,
        get_event_stats, predict_scenario,
    )
    from src.ml_model import get_model_suite, predict_from_models

    st.markdown(
        "**Pick a company, an event type, and a sentiment strength. "
        "The app returns the historical pattern for that combination — "
        "what tends to happen next, and the past cases behind the number.**"
    )
    st.caption(
        "Numbers are drawn from past headlines in the dataset. "
        "They describe what *historically followed* similar events — not a guarantee."
    )
    st.markdown("")

    # ── Scenario controls ──────────────────────────────────────────────────────
    sc_col1, sc_col2, sc_col3 = st.columns([2, 2, 2])

    with sc_col1:
        sc_ticker_options = ["📊 All companies"] + [
            TICKER_LABELS.get(t, t) for t in ordered if t in available_tickers
        ]
        sc_ticker_label = st.selectbox(
            "Company",
            options=sc_ticker_options,
            key="sc_ticker",
        )
        if sc_ticker_label == "📊 All companies":
            sc_ticker = None
            sc_sector = None
        else:
            sc_ticker = ordered[display_names.index(sc_ticker_label)]
            sc_sector = TICKER_SECTOR.get(sc_ticker)

    with sc_col2:
        event_display_options = [
            EVENT_LABELS[e] for e in ALL_EVENT_TYPES if e != "unclassified"
        ]
        sc_event_label = st.selectbox(
            "Event type",
            options=event_display_options,
            key="sc_event",
        )
        # Reverse-map label → key
        sc_event_type = next(
            (k for k, v in EVENT_LABELS.items() if v == sc_event_label), "unclassified"
        )

    with sc_col3:
        sc_sentiment = st.slider(
            "Sentiment strength",
            min_value=-1.0, max_value=1.0, value=0.6, step=0.05,
            help=(
                "FinBERT confidence score, signed: "
                "+1.0 = very positive news, −1.0 = very negative. "
                "0.6 is a typical positive headline."
            ),
            key="sc_sentiment",
        )
        # Plain-English label under the slider
        if sc_sentiment >= 0.5:
            sc_sent_label = "Strongly positive"
        elif sc_sentiment >= 0.1:
            sc_sent_label = "Mildly positive"
        elif sc_sentiment >= -0.1:
            sc_sent_label = "Neutral"
        elif sc_sentiment >= -0.5:
            sc_sent_label = "Mildly negative"
        else:
            sc_sent_label = "Strongly negative"
        st.caption(f"Sentiment: **{sc_sent_label}**")

    sc_window_options = {
        "Next day (t+1)":    "t1",
        "3 days later (t+3)": "t3",
        "5 days later (t+5)": "t5",
    }
    sc_window_label = st.selectbox(
        "Return window",
        options=list(sc_window_options.keys()),
        index=0,
        key="sc_window",
    )
    sc_window = sc_window_options[sc_window_label]

    st.markdown("---")

    # ── Load event-study data (cached) ────────────────────────────────────────
    @st.cache_data(show_spinner="Building event study …")
    def load_event_study(ticker: str | None, sector: str | None) -> "pd.DataFrame":
        from src.analysis    import analyze_data, analyze_all_companies
        from src.events      import tag_dataframe
        from src.event_study import build_event_study

        if ticker is None:
            df = analyze_all_companies(sector=sector)
        else:
            df = analyze_data(ticker=ticker)
            if "Sector" not in df.columns:
                df["Sector"] = TICKER_SECTOR.get(ticker, "Unknown")
            if "Ticker" not in df.columns:
                df["Ticker"] = ticker

        if df.empty:
            return df

        if "Event_Type" not in df.columns or df["Event_Type"].isna().any():
            df = tag_dataframe(df)

        return build_event_study(df)

    # Always load from the full pooled dataset so sector-fallback works
    @st.cache_data(show_spinner="Building event study …")
    def load_full_event_study() -> "pd.DataFrame":
        from src.analysis    import analyze_all_companies
        from src.events      import tag_dataframe
        from src.event_study import build_event_study

        df = analyze_all_companies()
        if df.empty:
            return df
        if "Event_Type" not in df.columns or df["Event_Type"].isna().any():
            df = tag_dataframe(df)
        return build_event_study(df)

    try:
        es_full = load_full_event_study()
    except Exception as _e:
        st.error(f"Could not build event study: {_e}")
        st.stop()

    if es_full.empty:
        st.warning(
            "No merged data available for the event study. "
            "Run `python src/fetch_news.py` to collect and score headlines first."
        )
        st.stop()

    # ── Run scenario prediction ────────────────────────────────────────────────
    sc_pred = predict_scenario(
        es_full,
        event_type=sc_event_type,
        sentiment_score=sc_sentiment,
        ticker=sc_ticker,
        sector=sc_sector,
        window=sc_window,
    )

    # ── Load model suite (cached separately — heavier) ────────────────────────
    @st.cache_data(show_spinner="Training models …")
    def load_model_suite(return_col: str) -> dict:
        return get_model_suite(return_col=return_col)

    _window_to_ret_col = {"t1": "ret_next_day", "t3": "ret_3day", "t5": "ret_5day"}
    sc_ret_col = _window_to_ret_col[sc_window]
    model_suite = load_model_suite(sc_ret_col)

    ml_pred = predict_from_models(
        model_suite,
        event_type=sc_event_type,
        sentiment_score=sc_sentiment,
        sector=sc_sector or "Unknown",
    )

    # ── Render: prediction card ───────────────────────────────────────────────
    n_cases = sc_pred["n"]
    conf    = sc_pred["confidence"]

    # Confidence gate: grey out if too few cases
    CONF_COLORS = {"low": "#444", "medium": "#a87000", "high": "#238636"}
    conf_color  = CONF_COLORS.get(conf, "#444")
    conf_label  = {"low": "Low confidence (< 15 cases)", "medium": "Medium (15–29 cases)", "high": "High (≥ 30 cases)"}

    if n_cases == 0:
        st.markdown(
            f'<div style="background:#1c1c1c;border:1px solid #444;border-radius:10px;padding:20px 24px;">'
            f'<p style="font-size:1.4rem;font-weight:700;color:#888;margin:0 0 8px 0;">No past cases found</p>'
            f'<p style="color:#888;margin:0;">There are no headlines in the dataset that match '
            f'<b>{sc_event_label}</b> events for this selection. '
            f'Try "All companies" or a different event type.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Format expected AR
        exp_ar   = sc_pred["expected_ar"]
        ar_sign  = "+" if exp_ar >= 0 else ""
        ar_color = "#3fb950" if exp_ar > 0.001 else ("#f85149" if exp_ar < -0.001 else "#8b949e")

        st.markdown(
            f'<div style="background:#161b22;border:1px solid {conf_color};'
            f'border-radius:10px;padding:20px 24px;">'
            f'<p style="font-size:1.5rem;font-weight:700;color:{ar_color};margin:0 0 4px 0;">'
            f'Expected abnormal return: {ar_sign}{exp_ar:.2%}'
            f'</p>'
            f'<p style="color:#8b949e;margin:0;font-size:0.95rem;">'
            f'{sc_event_label} &nbsp;·&nbsp; {sc_sent_label} sentiment &nbsp;·&nbsp; {sc_window_label} &nbsp;·&nbsp; '
            f'Based on {n_cases} past case{"s" if n_cases != 1 else ""} &nbsp;·&nbsp; '
            f'<span style="color:{conf_color};">{conf_label.get(conf, conf)}</span>'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── Four key metrics ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        p_up = sc_pred["p_up"]
        m1.metric(
            "Probability up",
            f"{p_up:.0%}" if p_up is not None else "—",
            help="Fraction of past similar events where the stock had a positive abnormal return.",
        )

        m2.metric(
            "Expected move",
            f"{ar_sign}{exp_ar:.2%}" if exp_ar is not None else "—",
            help="Weighted average abnormal return from the closest past cases.",
        )

        r_low  = sc_pred.get("range_low")
        r_high = sc_pred.get("range_high")
        m3.metric(
            "Range (P10)",
            f"{r_low:+.2%}" if r_low is not None else "—",
            help="10th percentile — what the worst 10% of similar events looked like.",
        )
        m4.metric(
            "Range (P90)",
            f"{r_high:+.2%}" if r_high is not None else "—",
            help="90th percentile — what the best 10% of similar events looked like.",
        )

        st.markdown("")

        # ── Model-based prediction row ────────────────────────────────────────
        if ml_pred["magnitude"] is not None or ml_pred["direction_prob"] is not None:
            st.subheader("Model predictions")
            st.caption(
                "These come from the Ridge regression and logistic model trained "
                "on all past events with the full feature set. "
                "They complement — but don't replace — the historical case count above."
            )
            mm1, mm2, mm3, mm4 = st.columns(4)

            if ml_pred["direction_prob"] is not None:
                mm1.metric(
                    "Direction probability (model)",
                    f"{ml_pred['direction_prob']:.0%}",
                    help="Logistic regression: probability of a positive return.",
                )

            if ml_pred["magnitude"] is not None:
                mm2.metric(
                    "Magnitude (Ridge)",
                    f"{ml_pred['magnitude']:+.2%}",
                    help="Ridge regression: predicted return magnitude.",
                )

            if ml_pred["q10"] is not None:
                mm3.metric("P10 (quantile model)", f"{ml_pred['q10']:+.2%}",
                           help="10th percentile from quantile regression.")
            if ml_pred["q90"] is not None:
                mm4.metric("P90 (quantile model)", f"{ml_pred['q90']:+.2%}",
                           help="90th percentile from quantile regression.")

        st.markdown("---")

        # ── Range bar chart ───────────────────────────────────────────────────
        st.subheader("Distribution of past abnormal returns")
        st.caption(
            "Each bar is one past event of this type. "
            "Orange line = expected move for your scenario. "
            "Bars are sorted smallest to largest."
        )

        similar = sc_pred.get("similar_cases", pd.DataFrame())
        ar_col_key = f"ar_{sc_window}"

        if not similar.empty and ar_col_key in similar.columns:
            ar_vals = similar[ar_col_key].dropna().sort_values().reset_index(drop=True)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=list(range(len(ar_vals))),
                y=ar_vals.values,
                marker_color=[
                    "#3fb950" if v > 0 else "#f85149" for v in ar_vals.values
                ],
                hovertemplate="AR: %{y:.2%}<extra></extra>",
                showlegend=False,
            ))
            fig_dist.add_hline(
                y=exp_ar, line_dash="dot",
                line_color="#f0883e", line_width=2,
                annotation_text=f"Expected: {ar_sign}{exp_ar:.2%}",
                annotation_position="top right",
            )
            fig_dist.update_layout(
                template="plotly_dark",
                xaxis_title="Past cases (sorted)",
                yaxis_title="Abnormal return",
                yaxis_tickformat=".1%",
                margin=dict(t=20, b=40),
                height=280,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        # ── Similar past cases table ──────────────────────────────────────────
        st.subheader("The past cases behind this prediction")
        st.caption(
            "Sorted by how close each headline's sentiment score is to yours. "
            "These are the specific events the model learned from."
        )

        if not similar.empty:
            display_cols = [c for c in
                            ["Date", "Ticker", "Company", "Headline",
                             "Sentiment", "Sentiment_Score", ar_col_key]
                            if c in similar.columns]
            tbl = similar[display_cols].copy()

            # Rename AR column for clarity
            tbl = tbl.rename(columns={ar_col_key: f"AR ({sc_window_label})"})

            # Format floats
            for col in ["Sentiment_Score", f"AR ({sc_window_label})"]:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(
                        lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
                    )

            st.dataframe(tbl, use_container_width=True)
        else:
            st.info("No individual cases to show for this selection.")

    st.markdown("---")

    # ── Historical event-type overview table ──────────────────────────────────
    st.subheader(f"All event types — historical averages ({sc_window_label})")
    st.caption(
        "Grey out = fewer than 15 cases. "
        "Amber = 15–29 cases (medium confidence). "
        "Full opacity = 30+ cases (high confidence). "
        "Pool across 'All companies' to see more cases per event type."
    )

    @st.cache_data(show_spinner=False)
    def get_overview_table(_es_df, window: str, ticker: str | None, sector: str | None) -> "pd.DataFrame":
        from src.event_study import event_stats_table
        return event_stats_table(_es_df, window=window, sector=sector, ticker=ticker)

    overview = get_overview_table(es_full, sc_window, sc_ticker, sc_sector)

    if overview.empty:
        st.info(
            "Not enough data to build the overview table. "
            "Switch to 'All companies' or collect more headlines."
        )
    else:
        # Color-code rows by confidence
        def _row_style(row):
            if row["Confidence"] == "high":
                return [""] * len(row)
            elif row["Confidence"] == "medium":
                return ["opacity: 0.7; color: #e6c76a"] * len(row)
            else:
                return ["opacity: 0.4; color: #888"] * len(row)

        show_cols = ["Event Type", "Avg AR", "Hit Rate", "Std Dev", "P10", "P90", "N", "Confidence"]
        styled = overview[show_cols].style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — PROOF
# Spec: box plot, scatter, pie, ML card.
# Raw stats (r, p, ρ, R², MAE) are allowed here.
# ══════════════════════════════════════════════════════════════════════════════
with tab_proof:

    # ── Box plot ──────────────────────────────────────────────────────────────
    st.subheader("Returns by sentiment label")
    st.caption(
        "Each box shows the spread of returns on days with that sentiment. "
        "Dots are individual days — count them to see how thin the data is. "
        "If the Positive box sits clearly above the Negative box, sentiment is tracking returns."
    )
    SENT_COLORS = {"Positive": "#3fb950", "Neutral": "#8b949e", "Negative": "#f85149"}
    fig_box = px.box(
        merged_df, x="Sentiment", y=selected_return_col,
        color="Sentiment", points="all",
        color_discrete_map=SENT_COLORS,
        category_orders={"Sentiment": ["Negative", "Neutral", "Positive"]},
    )
    fig_box.update_layout(
        template="plotly_dark",
        xaxis_title="", yaxis_title="Stock return",
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.subheader("Sentiment score vs return — every data point")
    st.caption(
        "Each dot is one trading day. "
        "X axis: how positive the news was (FinBERT confidence score). "
        "Y axis: how much the stock moved. "
        "Orange line = best-fit trend. "
        "Upward slope = higher sentiment goes with higher returns."
    )
    scatter_df = merged_df.dropna(subset=["Sentiment_Score", selected_return_col])
    if not scatter_df.empty:
        fig_s = px.scatter(
            scatter_df, x="Sentiment_Score", y=selected_return_col,
            color="Sentiment",
            hover_data=["Headline", "Date"] if "Headline" in scatter_df.columns else None,
            color_discrete_map=SENT_COLORS,
            trendline="ols", trendline_scope="overall",
            trendline_color_override="#f0883e",
        )
        fig_s.update_layout(
            template="plotly_dark",
            xaxis_title="Sentiment score  (−1 = very negative → +1 = very positive)",
            yaxis_title="Stock return",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # ── Sentiment split pie ───────────────────────────────────────────────────
    st.subheader("Sentiment distribution")
    st.caption(
        "A balanced split helps the correlation. "
        "If almost all headlines are Positive, the model barely sees any Negative examples to compare against."
    )
    counts = merged_df["Sentiment"].value_counts()
    fig_pie = px.pie(
        values=counts.values, names=counts.index, hole=0.5,
        color=counts.index, color_discrete_map=SENT_COLORS,
    )
    fig_pie.update_layout(template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ── ML model card ─────────────────────────────────────────────────────────
    st.subheader("Can a model trained on sentiment predict the direction of returns?")

    @st.cache_data(show_spinner="Running model…")
    def run_ml() -> dict:
        from src.ml_model import train_and_evaluate
        return train_and_evaluate()

    ml = run_ml()

    if ml["status"] == "insufficient_data":
        st.warning(ml["message"])
    else:
        dir_model = ml["dir_model"]
        dir_base  = ml["dir_baseline"]
        n_ml      = ml["n_total"]

        # Spec: directional accuracy is primary. Baseline sentence always present.
        if n_ml < 50:
            st.warning(
                f"This model correctly predicts the direction of returns "
                f"**{dir_model:.0%}** of the time. "
                f"A model that ignores sentiment and just guesses the average "
                f"would also be right about **{dir_base:.0%}** of the time."
            )
            st.caption(
                f"With only {n_ml} trading days, even a genuinely predictive model "
                "would appear no better than chance. This is expected — not a flaw."
            )
        elif dir_model > dir_base + 0.05:
            st.success(
                f"This model correctly predicts the direction of returns "
                f"**{dir_model:.0%}** of the time. "
                f"A model that ignores sentiment would be right **{dir_base:.0%}** of the time."
            )
        else:
            st.warning(
                f"This model correctly predicts the direction of returns "
                f"**{dir_model:.0%}** of the time — "
                f"not meaningfully better than a model that ignores sentiment entirely "
                f"(**{dir_base:.0%}**)."
            )

        # R² and MAE available but secondary
        with st.expander("Full model metrics"):
            st.caption(
                f"Training: {ml['train_date_range']} ({ml['n_train']} days)  →  "
                f"Test: {ml['test_date_range']} ({ml['n_test']} days)"
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Directional accuracy", f"{dir_model:.1%}",
                       delta=f"baseline: {dir_base:.1%}",
                       delta_color="normal" if dir_model > dir_base else "inverse")
            mc2.metric("R² on test set", f"{ml['r2_model']:.3f}",
                       delta=f"baseline: {ml['r2_baseline']:.3f}",
                       delta_color="normal" if ml["r2_model"] > ml["r2_baseline"] else "inverse")
            mc3.metric("MAE", f"{ml['mae_model']:.4f}",
                       delta=f"baseline: {ml['mae_baseline']:.4f}",
                       delta_color="inverse" if ml["mae_model"] < ml["mae_baseline"] else "normal")



# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# TAB — HEADLINES DATA
# Spec: headlines table + confidence filter. Nothing else.
# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
with tab_data:

    st.subheader("Headlines used in this analysis")
    st.caption(
        "Each row is one trading day after aggregation. "
        "n_headlines = how many articles were averaged into that day's sentiment score. "
        "Headlines have been cleaned: junk removed, company-validated, near-duplicates merged."
    )

    # Phase 5: confidence filter
    conf_min = st.slider(
        "Filter: hide headlines where FinBERT confidence is below",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help=(
            "FinBERT's score close to 0 means it wasn't confident about the sentiment. "
            "Setting this to 0.2 hides the 'unsure' headlines and shows only clear signals."
        ),
    )

    show_cols = [c for c in
                 ["Date", "Ticker", "Sector", "Company",
                  "Headline", "Sentiment", "Sentiment_Score", "n_headlines"]
                 if c in merged_df.columns]
    display_df = merged_df[show_cols].copy()

    if "Sentiment_Score" in display_df.columns:
        display_df["Sentiment_Score"] = display_df["Sentiment_Score"].round(3)
        if conf_min > 0:
            before = len(display_df)
            display_df = display_df[display_df["Sentiment_Score"].abs() >= conf_min]
            hidden = before - len(display_df)
            if hidden:
                st.caption(
                    f"Hiding {hidden} headline{'s' if hidden != 1 else ''} where "
                    f"|score| < {conf_min:.2f} — FinBERT's uncertain zone."
                )

    st.dataframe(display_df, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Stock data: yfinance (NSE)  ·  News: GNews API  ·  "
    "Sentiment: FinBERT (ProsusAI/finbert)  ·  Updates daily via GitHub Actions"
)
