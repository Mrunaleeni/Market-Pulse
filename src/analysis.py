"""
analysis.py
-----------
Core data pipeline: loads stock CSVs, runs sentiment scoring, merges on date,
and returns a clean DataFrame ready for the dashboard or ML model.

Logging
-------
Uses the 'analysis' logger. Set LOG_LEVEL=DEBUG in your environment to see
merge row counts, forward-fill details, and ticker skips.
The dashboard sees nothing by default.
"""

import logging
import os
import glob
import pandas as pd
from src.sentiment import get_sentiment, get_sentiment_score, analyze_sentiment as _analyze_sentiment

log = logging.getLogger("analysis")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")

# ── Company ↔ Ticker mapping ─────────────────────────────────────────────────
# Maps ticker keys (filenames like HDFC_2026.csv → "HDFC") to the Company
# labels used in news.csv. Keeps each stock's news matched to its own prices.
TICKER_TO_COMPANY = {
    "RELIANCE":  "Reliance",
    "TCS":       "TCS",
    "INFOSYS":   "Infosys",
    "HDFC":      "HDFC Bank",
    "ICICI":     "ICICI Bank",
    "AXISBANK":  "Axis Bank",
    "HUL":       "HUL",
    "ONGC":      "ONGC",
    "MARUTI":    "Maruti",
    # New companies
    "SBI":       "SBI",
    "ITC":       "ITC",
    "TATAMOTORS":"Tata Motors",
    "SUNPHARMA": "Sun Pharma",
    "DRREDDY":   "Dr Reddys",
    "TATASTEEL": "Tata Steel",
    "LT":        "L&T",
    "AIRTEL":    "Airtel",
}

# Thin wrapper kept for backwards compatibility with ml_model.py
from nltk.sentiment import SentimentIntensityAnalyzer as _SIA
_sia = _SIA()

def get_sentiment(text):   # noqa: F811
    from src.sentiment import get_sentiment as _gs
    return _gs(text)

# ── Stock loader ──────────────────────────────────────────────────────────────
def load_stock_data() -> pd.DataFrame:
    """
    Load all stock CSVs from data/raw/.
    Prefers *_2026.csv files; falls back to the originals if none found.
    Returns columns: Date, Close, Return, Ticker.
    """
    files = glob.glob(os.path.join(RAW_DIR, "*_2026.csv"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
                 if "news" not in os.path.basename(f).lower()]

    frames = []
    for f in files:
        ticker = os.path.basename(f).replace("_2026.csv", "").replace(".csv", "")
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]

        if "Date" not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)

        close_col = next(
            (c for c in df.columns if c.lower() in ("close", "price", "adj close")), None
        )
        if close_col is None:
            log.warning("Skipping %s — no Close column found.", ticker)
            continue

        df = df[["Date", close_col]].rename(columns={close_col: "Close"})
        df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
        df["Return"] = df["Close"].pct_change(fill_method=None)
        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No stock CSV files found in data/raw/")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Date", "Close"])
    log.debug("Loaded stock data: %d rows across %d tickers.", len(combined), len(frames))
    return combined


# ── Weekend / holiday forward-fill ────────────────────────────────────────────
def forward_fill_to_trading_day(news_df: pd.DataFrame,
                                trading_dates: pd.Series) -> pd.DataFrame:
    """
    Advance each headline's date to the next available trading day so that
    weekend and holiday news is matched to the Monday open rather than dropped.

    Adds a 'Merge_Date' column; the original 'Date' (publication date) is kept.
    Headlines with no subsequent trading day in the stock data are dropped.
    """
    trading_set = pd.DatetimeIndex(sorted(trading_dates.unique()))

    def next_trading_day(pub_date):
        future = trading_set[trading_set >= pub_date]
        return future[0] if len(future) else pd.NaT

    df = news_df.copy()
    df["Merge_Date"] = df["Date"].apply(next_trading_day)

    dropped = df["Merge_Date"].isna().sum()
    if dropped:
        log.warning(
            "%d headline(s) dropped — no trading day available after their publication date.",
            dropped,
        )
    df = df.dropna(subset=["Merge_Date"])
    return df


# ── Ticker discovery ─────────────────────────────────────────────────────────
def get_available_tickers() -> list[str]:
    """Return sorted list of ticker names that have stock CSVs on disk."""
    files = glob.glob(os.path.join(RAW_DIR, "*_2026.csv"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
                 if "news" not in os.path.basename(f).lower()]
    return sorted(
        os.path.basename(f).replace("_2026.csv", "").replace(".csv", "")
        for f in files
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_data(ticker: str | None = None,
                 pre_scored_news: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge news sentiment with stock returns on Date.

    Parameters
    ----------
    ticker : str or None
        If given (e.g. "RELIANCE"), filters to that stock's price/return data
        AND to only headlines tagged with that company.
        If None, averages all tickers per date and uses all headlines.
    pre_scored_news : DataFrame or None
        If supplied, skip running FinBERT and use this already-scored DataFrame
        instead. Pass the output of analyze_sentiment() cached at the app level
        so the model only runs once per session, not once per ticker switch.

    Returns
    -------
    DataFrame with columns: Date, Headline, Company, Sentiment,
                            Sentiment_Score, Close, Return
    (plus Published_Date when any headlines were forward-filled)
    """
    # --- News + sentiment scores (one FinBERT pass for the whole file) ---
    # If pre-scored news is supplied (e.g. cached by the dashboard), use it
    # directly instead of re-running FinBERT. This means switching companies
    # only re-filters and re-merges — it never re-runs the model.
    if pre_scored_news is not None:
        news_df = pre_scored_news.copy()
    else:
        news_df = _analyze_sentiment()
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")

    # --- Stock prices ---
    stock_df = load_stock_data()

    if ticker:
        ticker_df = stock_df[stock_df["Ticker"] == ticker].copy()
        if ticker_df.empty:
            raise ValueError(
                f"Ticker '{ticker}' not found. "
                f"Available: {stock_df['Ticker'].unique().tolist()}"
            )
        daily_stock = ticker_df[["Date", "Close", "Return"]]

        # Filter news to only this company's headlines
        company_label = TICKER_TO_COMPANY.get(ticker)
        if company_label and "Company" in news_df.columns:
            filtered_news = news_df[news_df["Company"] == company_label]
            if filtered_news.empty:
                log.warning(
                    "No headlines found for company '%s' — using all news as fallback.",
                    company_label,
                )
                filtered_news = news_df
        else:
            filtered_news = news_df
    else:
        daily_stock = (
            stock_df
            .groupby("Date", as_index=False)
            .agg(Close=("Close", "mean"), Return=("Return", "mean"))
        )
        filtered_news = news_df

    # --- Forward-fill weekends/holidays to next trading day ---
    filtered_news = forward_fill_to_trading_day(filtered_news, daily_stock["Date"])

    # --- Merge on Merge_Date ---
    merged_df = pd.merge(
        filtered_news,
        daily_stock,
        left_on="Merge_Date",
        right_on="Date",
        how="inner",
        suffixes=("_news", ""),
    )
    if "Date_news" in merged_df.columns:
        merged_df = merged_df.rename(columns={"Date_news": "Published_Date"})
    merged_df = merged_df.drop(columns=["Merge_Date"], errors="ignore")
    merged_df = merged_df.dropna(subset=["Return"])
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    log.debug(
        "analyze_data(ticker=%s): %d rows, %s → %s",
        ticker,
        len(merged_df),
        merged_df["Date"].min().date() if len(merged_df) else "—",
        merged_df["Date"].max().date() if len(merged_df) else "—",
    )
    return merged_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else None
    df = analyze_data(ticker=ticker_arg)

    print(f"\nMerged rows: {len(df)}")
    print("Columns:", list(df.columns))
    if "Sentiment_Score" in df.columns and len(df) >= 3:
        r = df["Sentiment_Score"].corr(df["Return"])
        print(f"Pearson r (sentiment vs return): {r:.3f}")
    print("\nAverage return by sentiment:")
    print(df.groupby("Sentiment")["Return"].mean())
