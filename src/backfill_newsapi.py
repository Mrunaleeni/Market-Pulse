"""
backfill_newsapi.py
-------------------
One-off script to backfill up to 30 days of historical headlines from
NewsAPI.org (https://newsapi.org) into data/news_backfill.csv.

Usage
-----
1. Register for a free key at https://newsapi.org (takes 2 minutes).
2. Set the environment variable:
       set NEWSAPI_KEY=your_key_here          (Windows CMD)
       $env:NEWSAPI_KEY="your_key_here"       (PowerShell)
3. Run:
       python src/backfill_newsapi.py

The script writes to data/news_backfill.csv (NOT news.csv).
After you review the summary printed at the end, merge manually:
       python src/backfill_newsapi.py --merge

Design rules
------------
- Reuses passes_alias_check, is_junk, remove_near_duplicates from fetch_news.py
  so backfilled rows are filtered identically to daily rows.
- Deduplicates against existing news.csv by exact headline and by fuzzy match.
- Scores sentiment with FinBERT (or VADER fallback) and tags Event_Type.
- Adds a Source='newsapi' column. Existing news.csv rows get Source='gnews'
  when --merge is run.
- Respects the free plan:  100 requests/day, 100 articles/request.
  Stops cleanly on 429/426 rate-limit responses.
- After-hours rule: headlines published after 15:30 IST are dated to the
  next calendar day (the same rule used in analysis.py's forward_fill logic).
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_CSV     = os.path.join(BASE_DIR, "data", "raw", "news.csv")
BACKFILL_CSV = os.path.join(BASE_DIR, "data", "news_backfill.csv")

sys.path.insert(0, BASE_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_newsapi")

# ── NewsAPI config ────────────────────────────────────────────────────────────
NEWSAPI_BASE   = "https://newsapi.org/v2/everything"
PAGE_SIZE      = 100        # max per request on free plan
DELAY_SECONDS  = 1.2        # polite delay between requests
MAX_PAGES      = 1          # free plan: 1 page × 100 articles per query is enough
                             # (only ~1 month of history on free tier anyway)

# ── IST timezone ──────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
NSE_CLOSE_IST_HOUR = 15
NSE_CLOSE_IST_MIN  = 30

# ── Company search queries for NewsAPI ───────────────────────────────────────
# Using qInTitle so the company name MUST appear in the headline.
# This eliminates the 91% alias-rejection problem where NewsAPI matched
# the company name in the article body but not the headline.
# qInTitle supports simple terms and exact phrases but not OR operators,
# so we run one request per major alias and deduplicate afterwards.
# Each entry is a LIST of qInTitle queries to try for that company.
COMPANY_QUERIES = {
    "TCS":        ["TCS", "Tata Consultancy"],
    "Infosys":    ["Infosys"],
    "Wipro":      ["Wipro"],
    "HDFC Bank":  ["HDFC Bank", "HDFC"],
    "ICICI Bank": ["ICICI Bank", "ICICI"],
    "Axis Bank":  ["Axis Bank"],
    "SBI":        ["SBI", "State Bank India"],
    "Reliance":   ["Reliance Industries", "Jio Platforms", "Jio IPO"],
    "ONGC":       ["ONGC"],
    "HUL":        ["Hindustan Unilever", "HUL"],
    "ITC":        ["ITC Limited", "ITC Ltd"],
    "Maruti":     ["Maruti Suzuki", "Maruti"],
    "Tata Motors":["Tata Motors"],
    "Sun Pharma": ["Sun Pharma", "Sun Pharmaceutical"],
    "Dr Reddys":  ["Dr Reddy"],
    "Tata Steel": ["Tata Steel"],
    "L&T":        ["Larsen Toubro", "L&T"],
    "Airtel":     ["Bharti Airtel", "Airtel"],
}


def _ist_date(pub_str: str) -> str | None:
    """
    Parse a NewsAPI publishedAt string (ISO 8601 UTC, e.g. '2026-09-12T08:30:00Z')
    into a YYYY-MM-DD date in IST, applying the after-15:30 rule:
    news published after NSE close is dated to the next calendar day.

    Returns None if the string can't be parsed.
    """
    if not pub_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        dt_ist = dt_utc.astimezone(IST)
        # After-hours: push to next calendar day
        if (dt_ist.hour > NSE_CLOSE_IST_HOUR or
                (dt_ist.hour == NSE_CLOSE_IST_HOUR and
                 dt_ist.minute >= NSE_CLOSE_IST_MIN)):
            dt_ist = dt_ist + timedelta(days=1)
        return dt_ist.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def _iso_timestamp(pub_str: str) -> str | None:
    """Return a full ISO timestamp in IST for storage (keeps timezone info)."""
    if not pub_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        return dt_utc.astimezone(IST).isoformat()
    except (ValueError, AttributeError):
        return None


def fetch_newsapi(company: str, query: str, api_key: str,
                  from_date: str, to_date: str,
                  request_counter: list) -> list[dict] | None:
    """
    Fetch up to PAGE_SIZE articles for one company from NewsAPI using
    qInTitle so the company name MUST appear in the headline.

    Returns list of raw dicts, empty list on soft failure,
    or None on rate-limit (caller must stop).
    """
    params = {
        "qInTitle": query,       # ← company name must be in the headline
        "language": "en",
        "sortBy":   "publishedAt",
        "from":     from_date,
        "to":       to_date,
        "pageSize": PAGE_SIZE,
        "page":     1,
        "apiKey":   api_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=15)
        request_counter[0] += 1
    except requests.RequestException as e:
        log.error("Request failed for %s: %s", company, e)
        return []

    # Rate limit hit
    if resp.status_code in (426, 429):
        log.warning(
            "Rate limit hit after %d requests (HTTP %d). "
            "Stop and try again tomorrow.",
            request_counter[0], resp.status_code,
        )
        return None   # sentinel: caller must stop

    if resp.status_code == 401:
        log.error(
            "Unauthorised (HTTP 401). Check your NEWSAPI_KEY. "
            "Get a free key at https://newsapi.org"
        )
        return None

    if not resp.ok:
        log.warning("HTTP %d for %s: %s", resp.status_code, company,
                    resp.text[:200])
        return []

    data = resp.json()
    if data.get("status") != "ok":
        code = data.get("code", "")
        msg  = data.get("message", "")
        # Free plan restriction on date range
        if code in ("parameterInvalid", "maximumResultsReached"):
            log.warning("%s — %s (company: %s)", code, msg, company)
            return []
        log.warning("NewsAPI error for %s: %s — %s", company, code, msg)
        return []

    articles = data.get("articles", [])
    rows = []
    for art in articles:
        headline = (art.get("title") or "").strip()
        pub_str  = art.get("publishedAt", "")
        url      = art.get("url", "")
        source   = art.get("source", {}).get("name", "")

        # Skip "removed" placeholders that NewsAPI returns for deleted articles
        if not headline or headline == "[Removed]":
            continue
        if not pub_str:
            continue

        date_str = _ist_date(pub_str)
        if date_str is None:
            continue

        rows.append({
            "Date":         date_str,
            "Headline":     headline,
            "Company":      company,
            "URL":          url,
            "PublishedISO": _iso_timestamp(pub_str),
            "NewsSource":   source,
        })

    return rows


def run_backfill(api_key: str, days_back: int = 30) -> pd.DataFrame:
    """
    Main backfill loop. Fetches articles for all 17 companies and returns
    a filtered, scored, tagged DataFrame ready for review.
    """
    from src.fetch_news import (
        passes_alias_check, is_junk, remove_near_duplicates,
        COMPANY_ALIASES,
    )

    to_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    log.info("Backfill window: %s → %s (%d days)", from_date, to_date, days_back)

    # Load existing news for dedup
    existing_headlines: set[str] = set()
    existing_df = pd.DataFrame()
    if os.path.exists(NEWS_CSV):
        existing_df = pd.read_csv(NEWS_CSV)
        existing_headlines = set(existing_df["Headline"].dropna().str.strip().str.lower())
        log.info("Loaded %d existing headlines from news.csv for dedup.", len(existing_headlines))

    request_counter = [0]
    all_rows: list[dict] = []

    # Per-company stats for summary
    stats: dict[str, dict] = {}

    total = len(COMPANY_QUERIES)
    for i, (company, queries) in enumerate(COMPANY_QUERIES.items(), start=1):
        # queries is now a list — run one request per alias term
        company_rows: list[dict] = []
        seen_in_company: set[str] = set()
        rate_limited = False

        for q in queries:
            print(f"[{i:>2}/{total}] {company:<20} qInTitle={q!r:<28}", end=" ", flush=True)
            rows = fetch_newsapi(company, q, api_key,
                                 from_date, to_date, request_counter)
            if rows is None:
                print("RATE LIMIT — stopping.")
                log.warning(
                    "Stopping early. %d requests used. %d rows collected so far.",
                    request_counter[0], len(all_rows),
                )
                rate_limited = True
                break

            # Deduplicate within the same company across multiple queries
            for r in rows:
                key = r["Headline"].strip().lower()
                if key not in seen_in_company:
                    seen_in_company.add(key)
                    company_rows.append(r)

            print(f"got {len(rows)}", flush=True)
            if len(queries) > 1:
                time.sleep(DELAY_SECONDS)

        if rate_limited:
            break

        stats[company] = {
            "fetched": len(company_rows), "junk": 0, "alias": 0,
            "exact_dup": 0, "kept": 0,
        }
        clean = [r for r in company_rows if not is_junk(r["Headline"])]
        stats[company]["junk"] = len(company_rows) - len(clean)

        # Alias validation (still apply as a safety net)
        validated = [r for r in clean if passes_alias_check(r["Headline"], r["Company"])]
        stats[company]["alias"] = len(clean) - len(validated)

        # Exact dedup against existing news.csv
        novel = []
        for r in validated:
            if r["Headline"].strip().lower() in existing_headlines:
                stats[company]["exact_dup"] += 1
            else:
                existing_headlines.add(r["Headline"].strip().lower())
                novel.append(r)

        stats[company]["kept"] = len(novel)
        all_rows.extend(novel)

        print(
            f"         total kept={stats[company]['kept']:>3}  "
            f"(junk={stats[company]['junk']}  alias={stats[company]['alias']}  "
            f"dup={stats[company]['exact_dup']})"
        )

        if i < total:
            time.sleep(DELAY_SECONDS)

    log.info("Total requests used: %d", request_counter[0])

    if not all_rows:
        log.warning("No new rows after all filters.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Source"] = "newsapi"

    # Near-duplicate removal within the new batch (cross-company dups already
    # can't happen since alias check ensures company match)
    before_nd = len(df)
    df = remove_near_duplicates(df)
    nd_removed = before_nd - len(df)
    if nd_removed:
        log.info("Near-duplicate removal: dropped %d more rows.", nd_removed)

    # Fuzzy dedup against existing news.csv
    if not existing_df.empty and len(df) > 0:
        from rapidfuzz import fuzz
        existing_hl_list = existing_df["Headline"].dropna().tolist()
        keep_mask = []
        fuzzy_dropped = 0
        for _, row in df.iterrows():
            is_dup = any(
                fuzz.token_set_ratio(row["Headline"], ex) >= 90
                for ex in existing_hl_list
            )
            if is_dup:
                fuzzy_dropped += 1
                keep_mask.append(False)
            else:
                keep_mask.append(True)
        df = df[keep_mask].reset_index(drop=True)
        if fuzzy_dropped:
            log.info("Fuzzy dedup against existing news.csv: dropped %d rows.", fuzzy_dropped)

    # ── Sentiment scoring ─────────────────────────────────────────────────────
    if len(df) > 0:
        try:
            from src.sentiment import get_sentiment_score, get_sentiment
            log.info("Scoring %d headlines with FinBERT/VADER …", len(df))
            # Batch via the pipeline directly if FinBERT is available
            try:
                from src.sentiment import _load_finbert, _FINBERT_LABEL_MAP
                pipe = _load_finbert()
                if pipe is not None:
                    headlines = df["Headline"].tolist()
                    results   = pipe(headlines, batch_size=16)
                    df["Sentiment"] = [
                        _FINBERT_LABEL_MAP.get(r["label"].lower(), "Neutral")
                        for r in results
                    ]
                    def _signed(r):
                        label = r["label"].lower()
                        if label == "positive": return  r["score"]
                        if label == "negative": return -r["score"]
                        return 0.0
                    df["Sentiment_Score"] = [_signed(r) for r in results]
                    log.info("FinBERT scoring complete.")
                else:
                    raise ImportError("FinBERT unavailable")
            except Exception:
                log.info("Falling back to VADER scoring.")
                df["Sentiment"]       = df["Headline"].apply(get_sentiment)
                df["Sentiment_Score"] = df["Headline"].apply(get_sentiment_score)
        except ImportError as e:
            log.warning("Sentiment module unavailable: %s — skipping.", e)
            df["Sentiment"]       = None
            df["Sentiment_Score"] = None

    # ── Event tagging ─────────────────────────────────────────────────────────
    if len(df) > 0:
        try:
            from src.events import tag_events_batch
            log.info("Tagging event types …")
            df["Event_Type"] = tag_events_batch(df["Headline"].tolist())
            coverage = (df["Event_Type"] != "unclassified").sum()
            log.info("Event tagging: %d/%d classified.", coverage, len(df))
        except ImportError as e:
            log.warning("Events module unavailable: %s — skipping.", e)
            df["Event_Type"] = None

    # Keep only the columns that match news.csv schema + extras
    final_cols = ["Date", "Headline", "Company", "Sentiment", "Sentiment_Score",
                  "Event_Type", "Source", "URL", "PublishedISO", "NewsSource"]
    for c in final_cols:
        if c not in df.columns:
            df[c] = None
    df = df[final_cols]

    return df, stats, request_counter[0]


def print_summary(df: pd.DataFrame, stats: dict, n_requests: int):
    """Print the final summary table."""
    print()
    print("=" * 70)
    print("  NEWSAPI BACKFILL SUMMARY")
    print("=" * 70)

    total_fetched = sum(s["fetched"] for s in stats.values())
    total_junk    = sum(s["junk"]    for s in stats.values())
    total_alias   = sum(s["alias"]   for s in stats.values())
    total_dup     = sum(s["exact_dup"] for s in stats.values())
    total_kept    = sum(s["kept"]    for s in stats.values())

    print(f"  API requests used  : {n_requests}")
    print(f"  Total fetched      : {total_fetched}")
    print(f"  Dropped — junk     : {total_junk}")
    print(f"  Dropped — alias    : {total_alias}")
    print(f"  Dropped — exact dup: {total_dup}")
    print(f"  After near-dup     : {len(df)}")
    print()

    if df.empty:
        print("  No new rows to add.")
        return

    print(f"  Date range covered : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print()
    print("  Rows per company:")
    for company, cnt in df.groupby("Company").size().sort_values(ascending=False).items():
        print(f"    {company:<22} {cnt:>4}")

    print()
    print("  Rows per event type:")
    if "Event_Type" in df.columns:
        for et, cnt in df["Event_Type"].value_counts().items():
            print(f"    {et:<22} {cnt:>4}")

    print()
    print(f"  Written to: {BACKFILL_CSV}")
    print()
    print("  To merge into news.csv, run:")
    print("    python src/backfill_newsapi.py --merge")
    print("=" * 70)


def check_price_coverage(df: pd.DataFrame):
    """
    Check whether *_2026.csv price files cover the backfill date range.
    Reports gaps. Does NOT fix anything.
    """
    if df.empty:
        return

    import glob
    raw_dir = os.path.join(BASE_DIR, "data", "raw")

    print()
    print("=" * 70)
    print("  PRICE FILE COVERAGE CHECK")
    print("=" * 70)

    backfill_min = df["Date"].min()
    backfill_max = df["Date"].max()
    print(f"  Backfill date range: {backfill_min.date()} → {backfill_max.date()}")
    print()

    price_files = glob.glob(os.path.join(raw_dir, "*_2026.csv"))
    if not price_files:
        print("  WARNING: No *_2026.csv files found in data/raw/")
        return

    from src.analysis import TICKER_TO_COMPANY
    company_to_ticker = {v: k for k, v in TICKER_TO_COMPANY.items()}

    problems = []
    for company in df["Company"].unique():
        ticker = company_to_ticker.get(company)
        if not ticker:
            continue
        price_path = os.path.join(raw_dir, f"{ticker}_2026.csv")
        if not os.path.exists(price_path):
            problems.append(f"  MISSING: {ticker}_2026.csv  (needed for {company})")
            continue
        prices = pd.read_csv(price_path, parse_dates=["Date"])
        price_min = prices["Date"].min()
        price_max = prices["Date"].max()
        if price_min > backfill_min:
            problems.append(
                f"  GAP: {ticker}_2026.csv starts {price_min.date()} "
                f"but backfill has news from {backfill_min.date()} "
                f"({(price_min - backfill_min).days} days uncovered)"
            )
        if price_max < backfill_max:
            problems.append(
                f"  GAP: {ticker}_2026.csv ends {price_max.date()} "
                f"but backfill has news to {backfill_max.date()}"
            )

    if problems:
        for p in problems:
            print(p)
        print()
        print("  Run `python src/fetch_news.py` to update stock prices.")
    else:
        print("  All price files cover the backfill date range. ✓")

    # Nifty index check
    nifty_files = glob.glob(os.path.join(raw_dir, "*ifty*")) + \
                  glob.glob(os.path.join(raw_dir, "*NIFTY*"))
    if not nifty_files:
        print()
        print("  NOTE: No Nifty index file found in data/raw/.")
        print("  Abnormal returns use equal-weighted market average instead.")
        print("  For a proper market benchmark, add a NIFTY50_2026.csv file.")
    print("=" * 70)


def post_merge_event_study_summary():
    """
    After merging, show how many past cases each event type and sector has.
    Uses the existing event_study code.
    """
    print()
    print("=" * 70)
    print("  EVENT STUDY CASE COUNTS (after merge)")
    print("=" * 70)
    try:
        from src.analysis    import analyze_all_companies
        from src.events      import tag_dataframe, EVENT_LABELS
        from src.event_study import build_event_study, get_event_stats

        df = analyze_all_companies()
        if df.empty:
            print("  No merged data available.")
            return
        if "Event_Type" not in df.columns or df["Event_Type"].isna().any():
            df = tag_dataframe(df)

        es_df = build_event_study(df)
        stats = get_event_stats(es_df)

        # Collect t1 window stats
        rows = []
        for (et, w), s in stats.items():
            if w != "t1":
                continue
            rows.append({
                "Event Type": EVENT_LABELS.get(et, et),
                "N cases":    s["n"],
                "Avg AR":     f"{s['avg_ar']:+.2%}",
                "Hit Rate":   f"{s['hit_rate']:.0%}",
                "Confidence": s["confidence"],
            })

        if rows:
            rdf = pd.DataFrame(rows).sort_values("N cases", ascending=False)
            print(rdf.to_string(index=False))
        else:
            print("  No event data available yet.")

    except Exception as e:
        print(f"  Could not run event study: {e}")
    print("=" * 70)


def merge_into_news_csv(backfill_df: pd.DataFrame):
    """
    Merge backfill rows into news.csv.
    - Adds Source='gnews' to all existing rows that lack a Source column.
    - Appends new rows (Source='newsapi') after deduplication.
    - Drops the URL / PublishedISO / NewsSource helper columns before writing
      (they're not in the standard news.csv schema — kept in backfill file only).
    """
    if not os.path.exists(NEWS_CSV):
        log.error("news.csv not found at %s", NEWS_CSV)
        return

    existing = pd.read_csv(NEWS_CSV, parse_dates=["Date"])

    # Backfill Source column on existing rows
    if "Source" not in existing.columns:
        existing["Source"] = "gnews"
    else:
        existing["Source"] = existing["Source"].fillna("gnews")

    # Strip helper columns from backfill before merge
    merge_cols = ["Date", "Headline", "Company", "Sentiment",
                  "Sentiment_Score", "Event_Type", "Source"]
    new_rows = backfill_df[merge_cols].copy()

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = (combined
                .drop_duplicates(subset=["Headline"])
                .sort_values("Date")
                .reset_index(drop=True))

    combined.to_csv(NEWS_CSV, index=False)
    added = len(combined) - len(existing)
    log.info("Merged %d new rows into news.csv (%d total).", added, len(combined))
    print(f"\n  ✓ Merged {added} new rows into news.csv ({len(combined)} total).")

    post_merge_event_study_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill headlines from NewsAPI into data/news_backfill.csv"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge existing news_backfill.csv into news.csv (run after review)",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="How many days back to fetch (default: 30, free plan max: 30)",
    )
    args = parser.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        if not os.path.exists(BACKFILL_CSV):
            print(f"ERROR: {BACKFILL_CSV} not found. Run without --merge first.")
            sys.exit(1)
        backfill_df = pd.read_csv(BACKFILL_CSV, parse_dates=["Date"])
        print(f"Merging {len(backfill_df)} rows from {BACKFILL_CSV} into {NEWS_CSV} …")
        merge_into_news_csv(backfill_df)
        sys.exit(0)

    # ── Fetch mode ────────────────────────────────────────────────────────────
    api_key = os.environ.get("NEWSAPI_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: NEWSAPI_KEY environment variable not set.\n"
            "  1. Register free at https://newsapi.org\n"
            "  2. Then run:\n"
            "       $env:NEWSAPI_KEY='your_key_here'   (PowerShell)\n"
            "       set NEWSAPI_KEY=your_key_here       (CMD)\n"
            "  3. Re-run this script."
        )
        sys.exit(1)

    print("=" * 70)
    print("  NewsAPI Backfill — Market Pulse")
    print("=" * 70)
    print(f"  Fetching last {args.days} days of headlines …")
    print()

    result = run_backfill(api_key, days_back=args.days)

    if isinstance(result, tuple):
        df, stats, n_req = result
    else:
        df = result
        stats = {}
        n_req = 0

    if df is not None and not df.empty:
        os.makedirs(os.path.dirname(BACKFILL_CSV), exist_ok=True)
        df.to_csv(BACKFILL_CSV, index=False)
        print_summary(df, stats, n_req)
        check_price_coverage(df)
    else:
        print("\nNo new headlines fetched. Nothing written.")
