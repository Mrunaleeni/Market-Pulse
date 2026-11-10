"""
backfill_et.py
--------------
Backfill historical headlines from Economic Times RSS feeds into
data/news_backfill_et.csv (separate file from the NewsAPI backfill so
you can review each source independently before merging).

robots.txt check (performed before building this script)
---------------------------------------------------------
  https://economictimes.indiatimes.com/robots.txt
  User-agent: *
  Allow: /
  The company/market/topic article pages are NOT in any Disallow rule.
  General crawling of public article listings is permitted.

Strategy
--------
ET does not have per-company RSS feeds, so this script:
  1. Pulls from multiple ET RSS feeds (markets, stocks, earnings, sector).
  2. Uses the existing COMPANY_ALIASES from fetch_news.py to assign each
     headline to the correct company (first matching company wins).
  3. A headline with no alias match is discarded.

This is honest about what RSS can deliver: ~50 articles per feed per run,
typically covering the last 1-3 days. Running daily accumulates history.
For genuine bulk backfill (months of history), use backfill_newsapi.py.

The script also reads ET's published news sitemaps
(sitemap-today.xml, sitemap-yesterday.xml) to reach articles older than
what the RSS feeds carry. Sitemaps are listed in robots.txt so fetching
them is explicitly sanctioned.

Politeness rules (honoured in full)
------------------------------------
- User-Agent: "MarketPulse-Research/1.0 (+https://github.com; non-commercial)"
- 2–3 second random delay between HTTP requests
- Retry with exponential back-off (2 s, 4 s, 8 s) on 5xx / connection errors
- No parallel requests (single-threaded throughout)
- Raw XML cached in data/raw_cache/ so a re-run never re-downloads

Timestamp rule
--------------
Every headline must have a reliable pubDate from the RSS/sitemap.
Headlines with no parseable timestamp are dropped (the after-15:30 rule
in analysis.py depends on exact IST timestamps).

Usage
-----
  python src/backfill_et.py                  # fetch + write to backfill file
  python src/backfill_et.py --merge          # merge into news.csv after review
  python src/backfill_et.py --no-cache       # ignore cached pages (re-download)
"""

import os
import sys
import time
import random
import logging
import hashlib
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import requests
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_CSV      = os.path.join(BASE_DIR, "data", "raw", "news.csv")
BACKFILL_CSV  = os.path.join(BASE_DIR, "data", "news_backfill_et.csv")
CACHE_DIR     = os.path.join(BASE_DIR, "data", "raw_cache")

sys.path.insert(0, BASE_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_et")

# ── HTTP config ───────────────────────────────────────────────────────────────
USER_AGENT    = "MarketPulse-Research/1.0 (non-commercial academic project)"
DELAY_MIN     = 2.0    # seconds
DELAY_MAX     = 3.0    # seconds
MAX_RETRIES   = 3
BACKOFF_BASE  = 2      # seconds; doubles each retry
REQUEST_TIMEOUT = 20   # seconds

# ── IST timezone ──────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
NSE_CLOSE_IST_HOUR = 15
NSE_CLOSE_IST_MIN  = 30

# ── ET RSS feeds to pull ──────────────────────────────────────────────────────
# These are the feeds confirmed working in the pre-build check.
# Each returns ~50 articles.  We pull all of them to maximise coverage.
ET_RSS_FEEDS = [
    # Markets & stocks (broadest coverage)
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    # Earnings results
    "https://economictimes.indiatimes.com/markets/stocks/earnings/rssfeeds/2146842.cms",
    # Sector feeds — Technology
    "https://economictimes.indiatimes.com/industry/services/consultancy-/-audit/rssfeeds/13358575.cms",
    # Sector feeds — Banking & Finance
    "https://economictimes.indiatimes.com/industry/banking/finance/rssfeeds/13358575.cms",
    "https://economictimes.indiatimes.com/industry/banking/finance/banking/rssfeeds/13358575.cms",
    # Sector feeds — Energy
    "https://economictimes.indiatimes.com/industry/energy/oil-gas/rssfeeds/13358575.cms",
    # Sector feeds — Auto
    "https://economictimes.indiatimes.com/industry/auto/rssfeeds/13358575.cms",
    # Sector feeds — Pharma
    "https://economictimes.indiatimes.com/industry/healthcare/biotech/pharmaceuticals/rssfeeds/13358575.cms",
    # Sector feeds — Telecom
    "https://economictimes.indiatimes.com/industry/telecom/rssfeeds/13358575.cms",
    # Company-specific topic pages via RSS
    "https://economictimes.indiatimes.com/topic/reliance-industries/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/tata-consultancy-services/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/infosys/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/hdfc-bank/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/icici-bank/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/sbi/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/tata-steel/rssfeeds/2143429.cms",
    "https://economictimes.indiatimes.com/topic/sun-pharma/rssfeeds/2143429.cms",
]

# Sitemap URLs that are listed in robots.txt (explicitly sanctioned)
ET_SITEMAPS = [
    "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-today.xml",
    "https://economictimes.indiatimes.com/etstatic/sitemaps/et/news/sitemap-yesterday.xml",
]


# ── Caching ───────────────────────────────────────────────────────────────────

def _cache_path(url: str) -> str:
    """Return local cache path for a URL."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"et_{key}.xml")


def _polite_get(url: str, use_cache: bool = True) -> bytes | None:
    """
    Fetch a URL with caching, politeness delay, and retry on transient errors.
    Returns raw bytes or None on permanent failure.
    """
    cache_path = _cache_path(url)

    # Check cache first
    if use_cache and os.path.exists(cache_path):
        log.debug("Cache hit: %s", url)
        with open(cache_path, "rb") as f:
            return f.read()

    # Polite delay before every real request
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.content
                with open(cache_path, "wb") as f:
                    f.write(data)
                log.debug("Fetched and cached: %s", url)
                return data
            elif resp.status_code in (301, 302, 307, 308):
                # Follow redirects manually (requests does this, but log it)
                log.debug("Redirect %d for %s", resp.status_code, url)
                return None
            elif resp.status_code == 404:
                log.debug("404 — skipping: %s", url)
                return None
            elif resp.status_code in (403, 406):
                log.warning("Access denied (%d) for %s — skipping.", resp.status_code, url)
                return None
            elif resp.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                log.warning("HTTP %d for %s — retry %d/%d in %ds",
                             resp.status_code, url, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d for %s — giving up.", resp.status_code, url)
                return None
        except requests.ConnectionError as e:
            wait = BACKOFF_BASE ** attempt
            log.warning("Connection error for %s (%s) — retry %d/%d in %ds",
                         url, e, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
        except requests.Timeout:
            log.warning("Timeout for %s — retry %d/%d", url, attempt, MAX_RETRIES)
            time.sleep(BACKOFF_BASE ** attempt)

    log.error("Failed to fetch %s after %d attempts.", url, MAX_RETRIES)
    return None


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _parse_pubdate(pub_str: str) -> datetime | None:
    """
    Parse an RSS pubDate string (RFC 2822) into an aware datetime in IST.
    Returns None if unparseable.
    """
    if not pub_str:
        return None
    try:
        dt = parsedate_to_datetime(pub_str.strip())
        return dt.astimezone(IST)
    except Exception:
        # Fallback: try ISO format (sitemaps use this)
        try:
            dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            return dt.astimezone(IST)
        except Exception:
            return None


def _effective_date(dt_ist: datetime) -> str:
    """
    Apply the after-15:30 IST rule: news published after NSE close is
    attributed to the next calendar day (matching analysis.py logic).
    """
    if (dt_ist.hour > NSE_CLOSE_IST_HOUR or
            (dt_ist.hour == NSE_CLOSE_IST_HOUR and
             dt_ist.minute >= NSE_CLOSE_IST_MIN)):
        dt_ist = dt_ist + timedelta(days=1)
    return dt_ist.strftime("%Y-%m-%d")


# ── RSS parser ────────────────────────────────────────────────────────────────

def _parse_rss(data: bytes, feed_url: str) -> list[dict]:
    """
    Parse RSS XML bytes and return list of raw article dicts:
      {headline, url, pubdate_str, dt_ist, date_str}
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        log.warning("XML parse error for %s: %s", feed_url, e)
        return []

    items = root.findall(".//item")
    rows = []
    for item in items:
        headline = (item.findtext("title") or "").strip()
        # Strip CDATA wrapper if present
        headline = headline.replace("<![CDATA[", "").replace("]]>", "").strip()
        url      = (item.findtext("link") or item.findtext("guid") or "").strip()
        pub_str  = (item.findtext("pubDate") or "").strip()

        if not headline or not pub_str:
            continue
        # Skip live-blog and price-ticker items — they have no real content
        if any(skip in url for skip in ["/liveblog/", "/stock-liveblog/", "/slideshow/"]):
            continue
        # Skip empty headlines that are just ticker updates
        if headline.endswith("Share Price Live Updates:") or headline.endswith("Stock Update"):
            continue

        dt_ist = _parse_pubdate(pub_str)
        if dt_ist is None:
            log.debug("Dropping — unparseable date: %s", headline[:60])
            continue

        rows.append({
            "headline":    headline,
            "url":         url,
            "pubdate_str": pub_str,
            "dt_ist":      dt_ist,
            "date_str":    _effective_date(dt_ist),
        })
    return rows


def _parse_sitemap(data: bytes, sitemap_url: str) -> list[dict]:
    """
    Parse a Google News sitemap XML and return list of raw article dicts.
    Sitemaps use <loc> for URL and <news:publication_date> for timestamp.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        log.warning("Sitemap parse error for %s: %s", sitemap_url, e)
        return []

    ns = {
        "sm":   "http://www.sitemaps.org/schemas/sitemap/0.9",
        "news": "http://www.google.com/schemas/sitemap-news/0.9",
    }

    urls  = root.findall(".//sm:url", ns)
    rows  = []
    for url_elem in urls:
        loc      = (url_elem.findtext("sm:loc", namespaces=ns) or "").strip()
        headline = (url_elem.findtext("news:news/news:title", namespaces=ns) or "").strip()
        pub_str  = (url_elem.findtext("news:news/news:publication_date", namespaces=ns) or "").strip()

        if not headline or not pub_str or not loc:
            continue
        if any(skip in loc for skip in ["/liveblog/", "/stock-liveblog/", "/slideshow/"]):
            continue

        dt_ist = _parse_pubdate(pub_str)
        if dt_ist is None:
            continue

        rows.append({
            "headline":    headline,
            "url":         loc,
            "pubdate_str": pub_str,
            "dt_ist":      dt_ist,
            "date_str":    _effective_date(dt_ist),
        })
    return rows


# ── Company assignment ────────────────────────────────────────────────────────

def assign_company(headline: str, url: str,
                   aliases: dict[str, list[str]]) -> str | None:
    """
    Return the first company whose alias appears in the headline (or URL).
    Returns None if no company matches.
    The order of COMPANY_ALIASES matters — more specific companies first.
    """
    text = (headline + " " + url).lower()
    for company, kws in aliases.items():
        if any(kw in text for kw in kws):
            return company
    return None


# ── Main fetch ────────────────────────────────────────────────────────────────

def run_et_backfill(use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Pull all ET feeds + sitemaps, filter, assign companies, deduplicate,
    score and tag.  Returns (result_df, stats_dict).
    """
    from src.fetch_news import (
        passes_alias_check, is_junk, remove_near_duplicates,
        COMPANY_ALIASES,
    )

    # ── 1. Fetch all feeds ────────────────────────────────────────────────────
    all_raw: list[dict] = []
    seen_urls: set[str] = set()

    all_sources = [(url, "rss")     for url in ET_RSS_FEEDS] + \
                  [(url, "sitemap") for url in ET_SITEMAPS]

    print(f"Fetching {len(all_sources)} ET sources …")
    for i, (url, kind) in enumerate(all_sources, 1):
        short = url.split("/")[-1][:50]
        print(f"  [{i:>2}/{len(all_sources)}] {kind:<7} {short}", end=" ", flush=True)

        data = _polite_get(url, use_cache=use_cache)
        if data is None:
            print("SKIP")
            continue

        if kind == "rss":
            rows = _parse_rss(data, url)
        else:
            rows = _parse_sitemap(data, url)

        new_count = 0
        for r in rows:
            if r["url"] and r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])
            all_raw.append(r)
            new_count += 1
        print(f"{new_count} new articles")

    log.info("Total unique raw articles from ET: %d", len(all_raw))

    if not all_raw:
        log.warning("No articles fetched from any ET feed.")
        return pd.DataFrame(), {}

    # ── 2. Assign companies ───────────────────────────────────────────────────
    assigned: list[dict] = []
    stats = {
        "total_raw":      len(all_raw),
        "no_company":     0,
        "junk":           0,
        "alias_fail":     0,
        "no_timestamp":   0,
        "exact_dup":      0,
        "fuzzy_dup":      0,
        "kept":           0,
    }

    for r in all_raw:
        company = assign_company(r["headline"], r["url"], COMPANY_ALIASES)
        if company is None:
            stats["no_company"] += 1
            continue
        r["Company"] = company
        assigned.append(r)

    log.info("Company-assigned: %d / %d", len(assigned), len(all_raw))

    # ── 3. Junk filter ────────────────────────────────────────────────────────
    clean = [r for r in assigned if not is_junk(r["headline"])]
    stats["junk"] = len(assigned) - len(clean)

    # ── 4. Alias double-check (headline must contain company keyword) ─────────
    validated = [r for r in clean if passes_alias_check(r["headline"], r["Company"])]
    stats["alias_fail"] = len(clean) - len(validated)

    # ── 5. Exact dedup against existing news.csv ──────────────────────────────
    existing_headlines: set[str] = set()
    existing_df = pd.DataFrame()
    if os.path.exists(NEWS_CSV):
        existing_df = pd.read_csv(NEWS_CSV)
        existing_headlines = set(
            existing_df["Headline"].dropna().str.strip().str.lower()
        )

    novel: list[dict] = []
    seen_in_batch: set[str] = set()
    for r in validated:
        key = r["headline"].strip().lower()
        if key in existing_headlines or key in seen_in_batch:
            stats["exact_dup"] += 1
            continue
        seen_in_batch.add(key)
        novel.append(r)

    # ── 6. Build DataFrame ────────────────────────────────────────────────────
    if not novel:
        log.info("No new articles after dedup.")
        return pd.DataFrame(), stats

    df = pd.DataFrame({
        "Date":        [r["date_str"]  for r in novel],
        "Headline":    [r["headline"]  for r in novel],
        "Company":     [r["Company"]   for r in novel],
        "URL":         [r["url"]       for r in novel],
        "PublishedISO":[r["dt_ist"].isoformat() for r in novel],
    })
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Source"] = "et"

    # ── 7. Near-duplicate removal within new batch ────────────────────────────
    before_nd = len(df)
    df = remove_near_duplicates(df)
    stats["fuzzy_dup"] = before_nd - len(df)

    # ── 8. Fuzzy dedup against existing news.csv ──────────────────────────────
    if not existing_df.empty and len(df) > 0:
        from rapidfuzz import fuzz
        ex_list  = existing_df["Headline"].dropna().tolist()
        keep     = []
        fz_count = 0
        for _, row in df.iterrows():
            is_dup = any(
                fuzz.token_set_ratio(row["Headline"], ex) >= 90
                for ex in ex_list
            )
            if is_dup:
                fz_count += 1
                keep.append(False)
            else:
                keep.append(True)
        df = df[keep].reset_index(drop=True)
        stats["fuzzy_dup"] += fz_count

    stats["kept"] = len(df)

    # ── 9. Sentiment scoring ──────────────────────────────────────────────────
    if len(df) > 0:
        try:
            from src.sentiment import _load_finbert, _FINBERT_LABEL_MAP, get_sentiment, get_sentiment_score
            pipe = _load_finbert()
            if pipe is not None:
                log.info("Scoring %d headlines with FinBERT …", len(df))
                results = pipe(df["Headline"].tolist(), batch_size=16)
                df["Sentiment"] = [
                    _FINBERT_LABEL_MAP.get(r["label"].lower(), "Neutral")
                    for r in results
                ]
                def _signed(r):
                    lbl = r["label"].lower()
                    if lbl == "positive": return  r["score"]
                    if lbl == "negative": return -r["score"]
                    return 0.0
                df["Sentiment_Score"] = [_signed(r) for r in results]
                log.info("FinBERT scoring complete.")
            else:
                raise ImportError("FinBERT unavailable")
        except Exception:
            log.info("Falling back to VADER scoring.")
            from src.sentiment import get_sentiment, get_sentiment_score
            df["Sentiment"]       = df["Headline"].apply(get_sentiment)
            df["Sentiment_Score"] = df["Headline"].apply(get_sentiment_score)

    # ── 10. Event tagging ─────────────────────────────────────────────────────
    if len(df) > 0:
        try:
            from src.events import tag_events_batch
            log.info("Tagging event types …")
            df["Event_Type"] = tag_events_batch(df["Headline"].tolist())
            coverage = (df["Event_Type"] != "unclassified").sum()
            log.info("Event tagging: %d/%d classified.", coverage, len(df))
        except ImportError as e:
            log.warning("Events module unavailable: %s", e)
            df["Event_Type"] = None

    # Ensure all schema columns present
    for col in ["Sentiment", "Sentiment_Score", "Event_Type"]:
        if col not in df.columns:
            df[col] = None

    final_cols = ["Date", "Headline", "Company", "Sentiment", "Sentiment_Score",
                  "Event_Type", "Source", "URL", "PublishedISO"]
    df = df[[c for c in final_cols if c in df.columns]]

    return df, stats


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, stats: dict):
    print()
    print("=" * 70)
    print("  ECONOMIC TIMES BACKFILL SUMMARY")
    print("=" * 70)
    print(f"  Raw articles fetched      : {stats.get('total_raw', 0)}")
    print(f"  Dropped — no company match: {stats.get('no_company', 0)}")
    print(f"  Dropped — junk filter     : {stats.get('junk', 0)}")
    print(f"  Dropped — alias fail      : {stats.get('alias_fail', 0)}")
    print(f"  Dropped — exact duplicate : {stats.get('exact_dup', 0)}")
    print(f"  Dropped — fuzzy duplicate : {stats.get('fuzzy_dup', 0)}")
    print(f"  Final new rows            : {stats.get('kept', 0)}")
    print()

    if df.empty:
        print("  No new rows to add.")
        print("=" * 70)
        return

    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
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
    print("    python src/backfill_et.py --merge")
    print("=" * 70)


def check_price_coverage(df: pd.DataFrame):
    """Same price-coverage check as in backfill_newsapi.py."""
    if df.empty:
        return
    import glob
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    print()
    print("=" * 70)
    print("  PRICE FILE COVERAGE CHECK")
    print("=" * 70)
    bf_min = df["Date"].min()
    bf_max = df["Date"].max()
    print(f"  Backfill date range: {bf_min.date()} → {bf_max.date()}")
    print()

    from src.analysis import TICKER_TO_COMPANY
    c2t = {v: k for k, v in TICKER_TO_COMPANY.items()}
    problems = []
    for company in df["Company"].unique():
        ticker = c2t.get(company)
        if not ticker:
            continue
        price_path = os.path.join(raw_dir, f"{ticker}_2026.csv")
        if not os.path.exists(price_path):
            problems.append(f"  MISSING: {ticker}_2026.csv  (needed for {company})")
            continue
        prices = pd.read_csv(price_path, parse_dates=["Date"])
        if prices["Date"].min() > bf_min:
            problems.append(
                f"  GAP: {ticker}_2026.csv starts {prices['Date'].min().date()} "
                f"but backfill has news from {bf_min.date()}"
            )
        if prices["Date"].max() < bf_max:
            problems.append(
                f"  GAP: {ticker}_2026.csv ends {prices['Date'].max().date()} "
                f"but backfill has news to {bf_max.date()}"
            )

    if problems:
        for p in problems:
            print(p)
        print()
        print("  Run `python src/fetch_news.py` to update stock prices.")
    else:
        print("  All price files cover the backfill date range. ✓")

    nifty = glob.glob(os.path.join(raw_dir, "*ifty*")) + \
            glob.glob(os.path.join(raw_dir, "*NIFTY*"))
    if not nifty:
        print()
        print("  NOTE: No Nifty index file found in data/raw/.")
        print("  Abnormal returns use equal-weighted market average instead.")
    print("=" * 70)


def post_merge_event_study_summary():
    """Show event study case counts after merge."""
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
    """Merge ET backfill rows into news.csv."""
    if not os.path.exists(NEWS_CSV):
        log.error("news.csv not found at %s", NEWS_CSV)
        return

    existing = pd.read_csv(NEWS_CSV, parse_dates=["Date"])

    if "Source" not in existing.columns:
        existing["Source"] = "gnews"
    else:
        existing["Source"] = existing["Source"].fillna("gnews")

    merge_cols = ["Date", "Headline", "Company", "Sentiment",
                  "Sentiment_Score", "Event_Type", "Source"]
    new_rows = backfill_df[[c for c in merge_cols if c in backfill_df.columns]].copy()

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
        description="Backfill headlines from Economic Times RSS into data/news_backfill_et.csv"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge existing news_backfill_et.csv into news.csv",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached pages and re-download everything",
    )
    args = parser.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        if not os.path.exists(BACKFILL_CSV):
            print(f"ERROR: {BACKFILL_CSV} not found. Run without --merge first.")
            sys.exit(1)
        bf = pd.read_csv(BACKFILL_CSV, parse_dates=["Date"])
        print(f"Merging {len(bf)} rows from {BACKFILL_CSV} into {NEWS_CSV} …")
        merge_into_news_csv(bf)
        sys.exit(0)

    # ── Fetch mode ────────────────────────────────────────────────────────────
    use_cache = not args.no_cache
    if not use_cache:
        log.info("Cache disabled — re-downloading all pages.")

    print("=" * 70)
    print("  Economic Times RSS Backfill — Market Pulse")
    print("=" * 70)
    print(f"  Feeds: {len(ET_RSS_FEEDS)} RSS + {len(ET_SITEMAPS)} sitemaps")
    print(f"  Cache: {'enabled' if use_cache else 'disabled'} ({CACHE_DIR})")
    print()

    df, stats = run_et_backfill(use_cache=use_cache)

    if df is not None and not df.empty:
        os.makedirs(os.path.dirname(BACKFILL_CSV) if os.path.dirname(BACKFILL_CSV) else ".", exist_ok=True)
        df.to_csv(BACKFILL_CSV, index=False)
        print_summary(df, stats)
        check_price_coverage(df)
    else:
        print_summary(pd.DataFrame(), stats)
        print("\nNo new headlines to write.")
