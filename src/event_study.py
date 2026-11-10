"""
event_study.py
--------------
Computes abnormal returns per event type and returns per-event statistics
that power both the dashboard's Scenario tab and the extended ML model.

What is an "abnormal return"?
------------------------------
  AR = stock return  −  market return (same window)

"Market return" here is the equal-weighted average of ALL tickers in the
dataset on the same day.  This removes broad market moves (e.g. a Nifty
rally on budget day) so what remains is company-specific price reaction.

Return windows
--------------
  t+1  : next trading day   (fast reaction)
  t+3  : 3 days later       (medium drift)
  t+5  : 5 days later       (slow drift / sustained reaction)

Per-event statistics (what the dashboard shows)
------------------------------------------------
  avg_ar        : average abnormal return (mean)
  hit_rate      : fraction of cases where the stock went up (AR > 0)
  std           : standard deviation (spread)
  p10 / p90     : 10th and 90th percentile (best/worst-case range)
  n             : number of past cases
  confidence    : "low" (<15), "medium" (15–29), "high" (≥30)

Public API
----------
  build_event_study(merged_df)      → full event-study DataFrame
  get_event_stats(merged_df)        → per-event stats dict keyed by (event_type, window)
  get_similar_cases(merged_df, ...) → k most similar past events for the scenario tab
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("event_study")

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOWS = ["t1", "t3", "t5"]          # shorthand for t+1, t+3, t+5
WINDOW_COLS = {                        # maps window key → stock return column
    "t1": "ret_next_day",
    "t3": "ret_3day",
    "t5": "ret_5day",
}
MIN_CASES_LOW    = 1    # minimum to even show a row
MIN_CASES_MEDIUM = 15   # minimum for "medium" confidence
MIN_CASES_HIGH   = 30   # minimum for "high" confidence


# ── Market return computation ─────────────────────────────────────────────────

def compute_market_returns(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the equal-weighted market return across all tickers per day,
    for each return window.

    Returns a DataFrame indexed by Date with columns:
      mkt_t1, mkt_t3, mkt_t5
    """
    required_windows = {
        "t1": "ret_next_day",
        "t3": "ret_3day",
        "t5": "ret_5day",
    }

    # If the df has a Ticker column, group by Date across tickers.
    # If it's a single-ticker df, we can't subtract market returns properly —
    # in that case we fall back to using all available rows.
    if "Ticker" not in merged_df.columns:
        # Single ticker: market return = the same ticker's return (AR = 0)
        # This is mathematically neutral and better than crashing.
        log.debug(
            "No Ticker column — using stock return as market return "
            "(single-ticker mode, AR == 0)."
        )
        mkt = merged_df[["Date"]].copy()
        for w_key, ret_col in required_windows.items():
            if ret_col in merged_df.columns:
                mkt[f"mkt_{w_key}"] = merged_df[ret_col].values
            else:
                mkt[f"mkt_{w_key}"] = np.nan
        return mkt.set_index("Date")

    groups = merged_df.groupby("Date")
    rows = []
    for date, grp in groups:
        row = {"Date": date}
        for w_key, ret_col in required_windows.items():
            if ret_col in grp.columns:
                row[f"mkt_{w_key}"] = grp[ret_col].mean()
            else:
                row[f"mkt_{w_key}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")


# ── Abnormal return computation ───────────────────────────────────────────────

def add_abnormal_returns(df: pd.DataFrame,
                         market_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Add AR columns (ar_t1, ar_t3, ar_t5) to df by subtracting the market
    return for the same day.

    Parameters
    ----------
    df             : output of analyze_data() — one row per (ticker, date)
    market_returns : output of compute_market_returns()

    Returns
    -------
    df with extra columns ar_t1, ar_t3, ar_t5.
    """
    df = df.copy()
    df = df.join(market_returns, on="Date", how="left")

    for w_key, ret_col in WINDOW_COLS.items():
        mkt_col = f"mkt_{w_key}"
        ar_col  = f"ar_{w_key}"
        if ret_col in df.columns and mkt_col in df.columns:
            df[ar_col] = df[ret_col] - df[mkt_col]
        else:
            df[ar_col] = np.nan

    return df


# ── ret_5day: derive if missing ───────────────────────────────────────────────

def _ensure_ret5day(df: pd.DataFrame) -> pd.DataFrame:
    """
    analysis.py computes ret_next_day and ret_3day but not ret_5day.
    We derive it here from the Close column when it's present.
    If Close is not available or there are too few rows, we use NaN.
    """
    if "ret_5day" in df.columns:
        return df
    df = df.copy()
    if "Close" in df.columns and "Ticker" in df.columns:
        frames = []
        for ticker, grp in df.groupby("Ticker"):
            grp = grp.sort_values("Date").copy()
            grp["ret_5day"] = grp["Close"].pct_change(5).shift(-5)
            frames.append(grp)
        df = pd.concat(frames).sort_values("Date").reset_index(drop=True)
    elif "Close" in df.columns:
        df_sorted = df.sort_values("Date").copy()
        df_sorted["ret_5day"] = df_sorted["Close"].pct_change(5).shift(-5)
        df = df_sorted
    else:
        df["ret_5day"] = np.nan
    return df


# ── Main builder ──────────────────────────────────────────────────────────────

def build_event_study(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the output of analyze_data() or analyze_all_companies() (which must
    have an Event_Type column), return an enriched DataFrame with:
      - ar_t1, ar_t3, ar_t5  : abnormal returns per window
      - All original columns preserved

    Parameters
    ----------
    merged_df : DataFrame from analysis.py — must contain Date, Event_Type,
                ret_next_day, ret_3day (ret_5day derived here if missing).

    Returns
    -------
    DataFrame with abnormal return columns added.
    """
    if "Event_Type" not in merged_df.columns:
        raise ValueError(
            "merged_df is missing the 'Event_Type' column. "
            "Run analysis.py with event tagging enabled first."
        )

    df = _ensure_ret5day(merged_df)
    market_rets = compute_market_returns(df)
    df = add_abnormal_returns(df, market_rets)

    classified = (df["Event_Type"] != "unclassified").sum()
    log.info(
        "build_event_study: %d rows, %d classified (%.1f%%)",
        len(df), classified, 100 * classified / max(len(df), 1),
    )
    return df


# ── Per-event statistics ──────────────────────────────────────────────────────

def _confidence(n: int) -> str:
    if n >= MIN_CASES_HIGH:
        return "high"
    if n >= MIN_CASES_MEDIUM:
        return "medium"
    return "low"


def get_event_stats(event_study_df: pd.DataFrame,
                    sector: str | None = None,
                    ticker: str | None = None) -> dict:
    """
    Compute per-event-type statistics for each return window.

    Parameters
    ----------
    event_study_df : output of build_event_study()
    sector         : if given, filter to this sector before computing stats
    ticker         : if given, filter to this ticker before computing stats
                     (ticker takes precedence over sector; if too few cases,
                      falls back to sector pool automatically)

    Returns
    -------
    Dict keyed by (event_type, window) → {avg_ar, hit_rate, std, p10, p90, n, confidence}

    Example
    -------
    stats[("company_results", "t1")] = {
        "avg_ar":     0.009,
        "hit_rate":   0.64,
        "std":        0.022,
        "p10":       -0.018,
        "p90":        0.031,
        "n":          22,
        "confidence": "high",
    }
    """
    df = event_study_df.copy()

    # Filter scope
    if ticker and "Ticker" in df.columns:
        ticker_df = df[df["Ticker"] == ticker]
        # Fall back to sector pool if too thin
        if len(ticker_df) < MIN_CASES_LOW and sector and "Sector" in df.columns:
            log.debug(
                "Ticker %s has too few rows (%d) — falling back to sector %s pool.",
                ticker, len(ticker_df), sector,
            )
            df = df[df["Sector"] == sector]
        else:
            df = ticker_df
    elif sector and "Sector" in df.columns:
        df = df[df["Sector"] == sector]

    results: dict = {}

    for event_type in df["Event_Type"].unique():
        if event_type == "unclassified":
            continue
        event_df = df[df["Event_Type"] == event_type]

        for w_key in WINDOWS:
            ar_col = f"ar_{w_key}"
            if ar_col not in event_df.columns:
                continue

            clean = event_df[ar_col].dropna()
            n = len(clean)
            if n < MIN_CASES_LOW:
                continue

            results[(event_type, w_key)] = {
                "avg_ar":     float(clean.mean()),
                "hit_rate":   float((clean > 0).mean()),
                "std":        float(clean.std()),
                "p10":        float(clean.quantile(0.10)),
                "p90":        float(clean.quantile(0.90)),
                "n":          n,
                "confidence": _confidence(n),
            }

    log.info(
        "get_event_stats: %d (event_type, window) combinations computed.",
        len(results),
    )
    return results


# ── Scenario helper: find similar past cases ──────────────────────────────────

def get_similar_cases(event_study_df: pd.DataFrame,
                      event_type: str,
                      sentiment_score: float,
                      ticker: str | None = None,
                      sector: str | None = None,
                      window: str = "t1",
                      k: int = 10) -> pd.DataFrame:
    """
    Return the k most similar past headline events for use in the "Scenario"
    tab — so the user can see *which* past headlines the model is drawing on.

    Similarity is defined as closeness of the sentiment score (|score − target|).
    Filtered first by event_type, optionally by ticker or sector.

    Returns a DataFrame with columns:
      Date, Ticker, Company, Headline, Sentiment, Sentiment_Score,
      Event_Type, ar_{window}, ret_{window_col}
    sorted by ascending |score_diff| (most similar first).
    """
    df = event_study_df.copy()

    # Filter by event type
    df = df[df["Event_Type"] == event_type]

    # Filter by scope (ticker > sector > all)
    if ticker and "Ticker" in df.columns:
        filtered = df[df["Ticker"] == ticker]
        if len(filtered) >= 3:
            df = filtered
        elif sector and "Sector" in df.columns:
            df = df[df["Sector"] == sector]
    elif sector and "Sector" in df.columns:
        df = df[df["Sector"] == sector]

    if df.empty:
        return pd.DataFrame()

    # Score similarity
    df = df.copy()
    df["_score_diff"] = (df["Sentiment_Score"] - sentiment_score).abs()
    df = df.sort_values("_score_diff")

    # Build display columns
    ar_col  = f"ar_{window}"
    ret_col = WINDOW_COLS.get(window, "ret_next_day")

    keep_cols = [c for c in
                 ["Date", "Ticker", "Sector", "Company",
                  "Headline", "Sentiment", "Sentiment_Score",
                  "Event_Type", ar_col, ret_col]
                 if c in df.columns]

    result = df[keep_cols].head(k).copy()
    result = result.drop(columns=["_score_diff"], errors="ignore")

    # Round floats for display
    for col in [ar_col, ret_col, "Sentiment_Score"]:
        if col in result.columns:
            result[col] = result[col].round(4)

    return result.reset_index(drop=True)


# ── Scenario prediction ───────────────────────────────────────────────────────

def predict_scenario(event_study_df: pd.DataFrame,
                     event_type: str,
                     sentiment_score: float,
                     ticker: str | None = None,
                     sector: str | None = None,
                     window: str = "t1") -> dict:
    """
    Given a hypothetical event + sentiment, return a prediction dict:
      expected_ar   : weighted average abnormal return (weights = similarity)
      p_up          : probability of going up (AR > 0)
      range_low     : 10th percentile AR from similar cases
      range_high    : 90th percentile AR from similar cases
      n             : how many past cases inform this prediction
      confidence    : "low" / "medium" / "high"
      similar_cases : DataFrame of the closest past cases

    Falls back to the global stats for this event type if no similar cases found.
    """
    stats = get_event_stats(
        event_study_df, sector=sector, ticker=ticker
    )
    stat_key = (event_type, window)
    global_stat = stats.get(stat_key, {})

    similar = get_similar_cases(
        event_study_df, event_type=event_type,
        sentiment_score=sentiment_score,
        ticker=ticker, sector=sector,
        window=window, k=20,
    )

    ar_col = f"ar_{window}"

    if similar.empty or ar_col not in similar.columns:
        # No similar cases at all — use global stat
        if global_stat:
            return {
                "expected_ar": global_stat["avg_ar"],
                "p_up":        global_stat["hit_rate"],
                "range_low":   global_stat["p10"],
                "range_high":  global_stat["p90"],
                "n":           global_stat["n"],
                "confidence":  global_stat["confidence"],
                "similar_cases": pd.DataFrame(),
                "source":      "global_average",
            }
        return {
            "expected_ar":  None,
            "p_up":         None,
            "range_low":    None,
            "range_high":   None,
            "n":            0,
            "confidence":   "low",
            "similar_cases": pd.DataFrame(),
            "source":       "no_data",
        }

    ar_vals = similar[ar_col].dropna()
    if len(ar_vals) == 0:
        return {
            "expected_ar":  None,
            "p_up":         None,
            "range_low":    None,
            "range_high":   None,
            "n":            0,
            "confidence":   "low",
            "similar_cases": similar,
            "source":       "no_ar_data",
        }

    # Inverse-distance weights on sentiment score similarity
    if "Sentiment_Score" in similar.columns:
        diffs = (similar["Sentiment_Score"] - sentiment_score).abs().values
        weights = 1.0 / (diffs + 0.01)   # add epsilon to avoid div/0
        weights = weights[:len(ar_vals)]  # align with dropna
        weights /= weights.sum()
        expected_ar = float(np.average(ar_vals.values, weights=weights))
    else:
        expected_ar = float(ar_vals.mean())

    n = len(ar_vals)
    return {
        "expected_ar": expected_ar,
        "p_up":        float((ar_vals > 0).mean()),
        "range_low":   float(ar_vals.quantile(0.10)) if n >= 3 else float(ar_vals.min()),
        "range_high":  float(ar_vals.quantile(0.90)) if n >= 3 else float(ar_vals.max()),
        "n":           n,
        "confidence":  _confidence(n),
        "similar_cases": similar.head(10),
        "source":      "similar_cases",
    }


# ── Summary table (for dashboard overview) ───────────────────────────────────

def event_stats_table(event_study_df: pd.DataFrame,
                      window: str = "t1",
                      sector: str | None = None,
                      ticker: str | None = None) -> pd.DataFrame:
    """
    Return a tidy DataFrame suitable for display in the dashboard, one row
    per event type, sorted by average abnormal return descending.

    Columns: Event_Type, Avg AR, Hit Rate, Std Dev, P10, P90, N, Confidence
    """
    from src.events import EVENT_LABELS

    stats = get_event_stats(event_study_df, sector=sector, ticker=ticker)

    rows = []
    for (event_type, w_key), s in stats.items():
        if w_key != window:
            continue
        rows.append({
            "Event Type":  EVENT_LABELS.get(event_type, event_type),
            "_event_type": event_type,
            "Avg AR":      s["avg_ar"],
            "Hit Rate":    s["hit_rate"],
            "Std Dev":     s["std"],
            "P10":         s["p10"],
            "P90":         s["p90"],
            "N":           s["n"],
            "Confidence":  s["confidence"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Avg AR", ascending=False).reset_index(drop=True)

    # Format for display
    df["Avg AR"]   = df["Avg AR"].map(lambda x: f"{x:+.2%}")
    df["Hit Rate"] = df["Hit Rate"].map(lambda x: f"{x:.0%}")
    df["Std Dev"]  = df["Std Dev"].map(lambda x: f"{x:.2%}")
    df["P10"]      = df["P10"].map(lambda x: f"{x:+.2%}")
    df["P90"]      = df["P90"].map(lambda x: f"{x:+.2%}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from src.analysis import analyze_all_companies

    print("Loading data …")
    try:
        df = analyze_all_companies()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if "Event_Type" not in df.columns:
        print("Event_Type column missing — tagging now …")
        from src.events import tag_dataframe
        df = tag_dataframe(df)

    print(f"Loaded {len(df)} rows. Running event study …\n")
    es_df = build_event_study(df)

    for window in WINDOWS:
        print(f"\n{'='*60}")
        print(f"  EVENT STUDY — window: {window}")
        print(f"{'='*60}")
        tbl = event_stats_table(es_df, window=window)
        if tbl.empty:
            print("  No data.")
        else:
            print(tbl[["Event Type", "Avg AR", "Hit Rate", "Std Dev",
                        "P10", "P90", "N", "Confidence"]].to_string(index=False))
