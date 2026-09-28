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

# ── Sector mapping ────────────────────────────────────────────────────────────
# Used for sector-level pooled views on the dashboard.
TICKER_SECTOR = {
    "TCS":        "Technology",
    "INFOSYS":    "Technology",
    "RELIANCE":   "Energy",
    "ONGC":       "Energy",
    "HDFC":       "Banking",
    "ICICI":      "Banking",
    "AXISBANK":   "Banking",
    "SBI":        "Banking",
    "HUL":        "FMCG",
    "ITC":        "FMCG",
    "MARUTI":     "Auto",
    "TATAMOTORS": "Auto",
    "SUNPHARMA":  "Pharma",
    "DRREDDY":    "Pharma",
    "TATASTEEL":  "Metals & Infra",
    "LT":         "Metals & Infra",
    "AIRTEL":     "Telecom",
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
        df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"]     = pd.to_numeric(df["Close"], errors="coerce")
        df["Return"]    = df["Close"].pct_change(fill_method=None)
        # Lagged returns — how much did the price move AFTER today's news?
        # shift(-1) means "tomorrow's return" — the value from the next row
        # shift(-3) means "3 days later" — captures slower market reactions
        df["ret_next_day"] = df["Return"].shift(-1)
        df["ret_3day"]     = df["Close"].pct_change(3).shift(-3)
        df["Ticker"]    = ticker
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No stock CSV files found in data/raw/")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Date", "Close"])
    log.debug("Loaded stock data: %d rows across %d tickers.", len(combined), len(frames))
    return combined


# ── Weekend / holiday + after-hours forward-fill ─────────────────────────────
NSE_CLOSE_UTC = 10   # NSE closes 15:30 IST = 10:00 UTC

def forward_fill_to_trading_day(news_df: pd.DataFrame,
                                trading_dates: pd.Series) -> pd.DataFrame:
    """
    Advance each headline's Merge_Date to the next available trading day when:
      - The publication date falls on a weekend or market holiday, OR
      - The headline was published after NSE close (10:00 UTC = 15:30 IST),
        because post-close news cannot move today's price.

    The original 'Date' (actual publication timestamp) is always preserved.
    Headlines with no subsequent trading day in the stock data are dropped.
    """
    trading_set = pd.DatetimeIndex(sorted(trading_dates.unique()))

    def next_trading_day(pub_date):
        """Return the next trading day >= pub_date (same day counts if it's a trading day)."""
        future = trading_set[trading_set >= pub_date]
        return future[0] if len(future) else pd.NaT

    def effective_trading_date(pub_date):
        """
        If we have a timezone-aware timestamp and the headline came out after
        NSE close, push it to the NEXT trading day. Otherwise use the normal
        next-trading-day logic (handles weekends/holidays).
        """
        try:
            # pub_date is timezone-aware (has UTC offset)
            if pub_date.tzinfo is not None and pub_date.hour >= NSE_CLOSE_UTC:
                # Published after market close — affects next day's open
                next_day = pub_date.normalize() + pd.Timedelta(days=1)
                future = trading_set[trading_set >= next_day.tz_localize(None)]
            else:
                future = trading_set[trading_set >= pub_date.tz_localize(None)
                                     if pub_date.tzinfo else trading_set[trading_set >= pub_date]]
        except Exception:
            future = trading_set[trading_set >= pub_date.replace(tzinfo=None)
                                 if hasattr(pub_date, 'tzinfo') else trading_set[trading_set >= pub_date]]
        return future[0] if len(future) else pd.NaT

    df = news_df.copy()

    # Use after-hours logic only if timestamps are timezone-aware
    # (GNews gives ISO timestamps with timezone; date-only strings are tz-naive)
    sample = df["Date"].dropna().iloc[0] if not df["Date"].dropna().empty else None
    if sample is not None and hasattr(sample, 'tzinfo') and sample.tzinfo is not None:
        df["Merge_Date"] = df["Date"].apply(effective_trading_date)
        log.debug("Using after-hours cutoff (timestamps are tz-aware).")
    else:
        df["Merge_Date"] = df["Date"].apply(next_trading_day)
        log.debug("Using date-only forward-fill (timestamps are tz-naive).")

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


# ── Pooled pipeline ───────────────────────────────────────────────────────────
def analyze_all_companies(pre_scored_news: pd.DataFrame | None = None,
                          sector: str | None = None) -> pd.DataFrame:
    """
    Run analyze_data() for every ticker that has both stock data and headlines,
    then concatenate the results into one DataFrame.

    This is the pooled view — each company's news is still matched only to its
    own stock prices, but all rows are combined so you get a much larger n for
    correlation.

    Parameters
    ----------
    pre_scored_news : DataFrame or None
        Pre-scored headlines from FinBERT — passed through to analyze_data()
        so the model doesn't re-run for each ticker.
    sector : str or None
        If given (e.g. "Banking"), only include tickers from that sector.
        If None, include all available tickers.

    Returns
    -------
    Same column structure as analyze_data(), plus a 'Sector' column.
    """
    tickers = get_available_tickers()

    # Filter by sector if requested
    if sector:
        tickers = [t for t in tickers if TICKER_SECTOR.get(t) == sector]

    frames = []
    for t in tickers:
        company_label = TICKER_TO_COMPANY.get(t)
        if not company_label:
            continue
        # Check this company actually has headlines — skip silently if not
        if pre_scored_news is not None:
            has_news = (pre_scored_news.get("Company", pd.Series(dtype=str)) == company_label).any()
        else:
            import pandas as _pd
            _news = _pd.read_csv(NEWS_PATH)
            has_news = (company_label in _news.get("Company", _pd.Series(dtype=str)).values)
        if not has_news:
            continue

        try:
            df = analyze_data(ticker=t, pre_scored_news=pre_scored_news)
        except Exception as e:
            log.warning("Skipping %s in pooled view: %s", t, e)
            continue

        if df.empty:
            continue

        df["Ticker"] = t
        df["Sector"] = TICKER_SECTOR.get(t, "Other")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    log.debug(
        "analyze_all_companies(sector=%s): %d rows across %d tickers.",
        sector, len(pooled), len(frames)
    )
    return pooled


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
    # --- News + sentiment scores ---
    # Prefer pre-scored news if supplied (backwards compatibility).
    # Otherwise read directly from news.csv — scores are saved there by
    # fetch_news.py at fetch time, so the dashboard never needs to run FinBERT.
    if pre_scored_news is not None:
        news_df = pre_scored_news.copy()
    else:
        news_df = pd.read_csv(NEWS_PATH)
        # If scores are missing (e.g. CSV predates Step 5), fall back to FinBERT
        if "Sentiment_Score" not in news_df.columns or news_df["Sentiment_Score"].isna().all():
            log.warning("Sentiment scores missing from news.csv — running FinBERT now.")
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
        daily_stock = ticker_df[["Date", "Close", "Return", "ret_next_day", "ret_3day"]]

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
            .agg(
                Close=("Close", "mean"),
                Return=("Return", "mean"),
                ret_next_day=("ret_next_day", "mean"),
                ret_3day=("ret_3day", "mean"),
            )
        )
        filtered_news = news_df

    # --- Forward-fill weekends/holidays to next trading day ---
    filtered_news = forward_fill_to_trading_day(filtered_news, daily_stock["Date"])

    # --- Part C: Aggregate to one row per company per trading day -----------
    # If 4 HUL headlines land on the same day, average their sentiment scores
    # into one number before merging. Otherwise one busy news day gets 4x the
    # weight of a quiet day in the correlation, which is not meaningful.
    agg_news = (
        filtered_news
        .groupby("Merge_Date", as_index=False)
        .agg(
            Sentiment_Score=("Sentiment_Score", "mean"),
            Headline=("Headline", lambda x: " | ".join(x)),   # join for display
            Sentiment=("Sentiment", lambda x: x.mode()[0]),   # most common label
            n_headlines=("Headline", "count"),
        )
    )
    # Preserve Company column if present
    if "Company" in filtered_news.columns:
        company_per_date = (
            filtered_news.groupby("Merge_Date")["Company"].first().reset_index()
        )
        agg_news = agg_news.merge(company_per_date, on="Merge_Date", how="left")

    log.debug("After daily aggregation: %d rows (was %d)", len(agg_news), len(filtered_news))

    # --- Merge on Merge_Date ---
    merged_df = pd.merge(
        agg_news,
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
