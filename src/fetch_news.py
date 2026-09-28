"""
fetch_news.py
-------------
Fetches real financial news headlines from the GNews API for a diverse
set of 10 Indian companies and saves them to data/raw/news.csv.

Format output: Date, Headline  (matches what the rest of the pipeline expects)

GNews free tier limits:
  - 100 requests / day
  - Up to 10 articles per request
  - ~1 year of historical data
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GNEWS_API_KEY", "e487a418fd484c9fae5cec36d1806c9c")
BASE_URL = "https://gnews.io/api/v4/search"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")

# 17 companies across 8 sectors for diversity
COMPANIES = {
    # Technology
    "TCS":          "TCS Tata Consultancy Services",
    "Infosys":      "Infosys India IT",
    "Wipro":        "Wipro India technology",
    # Banking & Finance
    "HDFC Bank":    "HDFC Bank India",
    "ICICI Bank":   "ICICI Bank India",
    "Axis Bank":    "Axis Bank India",
    "SBI":          "SBI State Bank India",
    # Energy & Conglomerates
    "Reliance":     "Reliance Industries India",
    "ONGC":         "ONGC Oil Natural Gas India",
    # FMCG & Consumer
    "HUL":          "Hindustan Unilever India",
    "ITC":          "ITC India FMCG cigarettes",
    # Auto
    "Maruti":       "Maruti Suzuki India",
    "Tata Motors":  "Tata Motors India EV",
    # Pharma
    "Sun Pharma":   "Sun Pharmaceutical India",
    "Dr Reddys":    "Dr Reddy Laboratories India",
    # Metals & Infrastructure
    "Tata Steel":   "Tata Steel India metals",
    "L&T":          "Larsen Toubro India infrastructure",
    # Telecom
    "Airtel":       "Bharti Airtel India telecom",
}

MAX_ARTICLES_PER_COMPANY = 10   # GNews free tier max per request
DELAY_BETWEEN_REQUESTS   = 2    # seconds — stay well within rate limits

# ── Part A: Alias-based company validation ────────────────────────────────────
# Each company maps to a list of keywords that MUST appear in the headline
# (case-insensitive) for it to be a genuine match.
# If none of the keywords appear, the headline is rejected — even if GNews
# returned it for that company's search query.
# This permanently fixes mismatches like Apple Pay stories tagged as ICICI Bank.
COMPANY_ALIASES = {
    "TCS":         ["tcs", "tata consultancy"],
    "Infosys":     ["infosys"],
    "Wipro":       ["wipro"],
    "HDFC Bank":   ["hdfc"],
    "ICICI Bank":  ["icici"],
    "Axis Bank":   ["axis bank"],
    "SBI":         ["sbi", "state bank"],
    "Reliance":    ["reliance", "jio"],
    "ONGC":        ["ongc"],
    "HUL":         ["hul", "hindustan unilever", "unilever"],
    "ITC":         ["itc"],
    "Maruti":      ["maruti", "suzuki"],
    "Tata Motors": ["tata motors", "tata curvv", "tata punch", "tata.cars",
                    "tata.ev", "tmcv"],
    "Sun Pharma":  ["sun pharma", "sun pharmaceutical"],
    "Dr Reddys":   ["dr. reddy", "dr reddy", "drreddy"],
    "Tata Steel":  ["tata steel"],
    "L&T":         ["l&t", "larsen", "larsen & toubro"],
    "Airtel":      ["airtel", "bharti"],
}

def passes_alias_check(headline: str, company: str) -> bool:
    """
    Return True if the headline contains at least one alias keyword
    for the given company. Headlines that don't mention the company
    are rejected, even if GNews returned them for that search query.
    """
    aliases = COMPANY_ALIASES.get(company, [])
    if not aliases:
        return True   # no aliases defined — let it through
    h = headline.lower()
    return any(alias in h for alias in aliases)


# ── Part B: Near-duplicate removal ───────────────────────────────────────────
# The same story from 5 different outlets counts 5 times without this.
# We use fuzzy token matching — two headlines are duplicates if they share
# 90%+ of the same words regardless of order.
DEDUP_THRESHOLD = 90   # similarity score 0–100

def remove_near_duplicates(df: pd.DataFrame, threshold: int = DEDUP_THRESHOLD) -> pd.DataFrame:
    """
    Remove near-duplicate headlines within the same company using fuzzy matching.
    Keeps the first occurrence; drops subsequent ones that are too similar.
    """
    from rapidfuzz import fuzz

    keep = []
    # Process per company so a Reliance headline can't be a dup of a TCS headline
    for company, group in df.groupby("Company"):
        seen_headlines: list[str] = []
        for _, row in group.iterrows():
            h = row["Headline"]
            is_dup = any(
                fuzz.token_set_ratio(h, seen) >= threshold
                for seen in seen_headlines
            )
            if not is_dup:
                seen_headlines.append(h)
                keep.append(row)

    if not keep:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(keep).reset_index(drop=True)

# ── Junk filter ───────────────────────────────────────────────────────────────
# Keywords that flag a headline as off-topic (case-insensitive substring match).
# Add more here as you spot bad headlines over time.
JUNK_KEYWORDS = [
    "marriage advice",
    "spoke word artist",
    "quit her engineering job",    # human interest / ex-employee stories
    "road accident",
    "car smashed",
    "biker",
    "monsoon deficit",
    "el nino",
    "satcom",
    "trai chairman",
    "chief human resources officer",  # HR appointments at unrelated companies
]

def is_junk(headline: str) -> bool:
    h = headline.lower()
    return any(kw in h for kw in JUNK_KEYWORDS)

# ── Fetcher ───────────────────────────────────────────────────────────────────

def fetch_headlines(company_name: str, query: str) -> list[dict]:
    """Call GNews and return a list of {Date, Headline, Company} dicts."""
    params = {
        "q":        query,
        "lang":     "en",
        "country":  "in",
        "max":      MAX_ARTICLES_PER_COMPANY,
        "apikey":   API_KEY,
        "sortby":   "publishedAt",
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed for {company_name}: {e}")
        return []

    articles = data.get("articles", [])
    rows = []
    for article in articles:
        published = article.get("publishedAt", "")
        headline  = article.get("title", "").strip()
        if not headline or not published:
            continue
        # Parse ISO datetime → date string YYYY-MM-DD
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            date_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        except ValueError:
            date_str = published[:10]

        rows.append({
            "Date":     date_str,
            "Headline": headline,
            "Company":  company_name,
        })

    # Filter junk before returning
    clean = [r for r in rows if not is_junk(r["Headline"])]
    if len(clean) < len(rows):
        print(f"(filtered {len(rows) - len(clean)} junk)", end=" ", flush=True)

    # Filter alias mismatches — headline must actually mention the company
    validated = [r for r in clean if passes_alias_check(r["Headline"], r["Company"])]
    if len(validated) < len(clean):
        print(f"(rejected {len(clean) - len(validated)} mismatches)", end=" ", flush=True)

    return validated


def fetch_all_companies() -> pd.DataFrame:
    all_rows = []
    total = len(COMPANIES)

    for i, (company_name, query) in enumerate(COMPANIES.items(), start=1):
        print(f"[{i}/{total}] Fetching: {company_name} ...", end=" ", flush=True)
        rows = fetch_headlines(company_name, query)
        print(f"{len(rows)} articles")
        all_rows.extend(rows)
        if i < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    if not all_rows:
        print("\n[WARNING] No articles fetched. Check your API key or network.")
        return pd.DataFrame(columns=["Date", "Headline", "Company"])

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Headline"]).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def update_stock_data():
    """
    Download the latest stock prices for all companies up to today
    and append any new rows to the existing *_2026.csv files.
    This keeps stock data in sync with newly fetched news dates.
    """
    import yfinance as yf

    TICKERS = {
        "RELIANCE":   "RELIANCE.NS",
        "TCS":        "TCS.NS",
        "INFOSYS":    "INFY.NS",
        "HDFC":       "HDFCBANK.NS",
        "ICICI":      "ICICIBANK.NS",
        "AXISBANK":   "AXISBANK.NS",
        "ONGC":       "ONGC.NS",
        "HUL":        "HINDUNILVR.NS",
        "MARUTI":     "MARUTI.NS",
        # New companies
        "SBI":        "SBIN.NS",
        "ITC":        "ITC.NS",
        "TATAMOTORS": "TMCV.NS",        # Tata Motors CV (post-demerger Oct 2025)
        "SUNPHARMA":  "SUNPHARMA.NS",
        "DRREDDY":    "DRREDDY.NS",
        "TATASTEEL":  "TATASTEEL.NS",
        "LT":         "LT.NS",
        "AIRTEL":     "BHARTIARTL.NS",
    }

    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("\nUpdating stock data...")
    for name, ticker in TICKERS.items():
        filepath = os.path.join(raw_dir, f"{name}_2026.csv")

        # Find the last date we already have
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, parse_dates=["Date"])
            last_date = existing["Date"].max()
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
            start = "2026-08-01"

        if start >= today:
            print(f"  {name}: already up to date")
            continue

        print(f"  {name}: fetching {start} → {today} ...", end=" ", flush=True)
        try:
            new_df = yf.download(ticker, start=start, end=today,
                                 auto_adjust=True, progress=False)
            if new_df.empty:
                print("no new data")
                continue

            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = [c[0] for c in new_df.columns]

            new_df = new_df.reset_index()

            if not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date")
            else:
                combined = new_df

            combined.to_csv(filepath, index=False)
            print(f"+{len(new_df)} rows")
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    print("=" * 55)
    print("  GNews Fetcher — Real Financial Headlines")
    print("=" * 55)
    print(f"  Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    new_df = fetch_all_companies()

    if new_df.empty:
        print("Nothing fetched. Exiting.")
    else:
        # ── Append to existing news.csv, deduplicate ──────────────────────
        if os.path.exists(OUTPUT_PATH):
            existing = pd.read_csv(OUTPUT_PATH, parse_dates=["Date"])
            before   = len(existing)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = (combined
                        .drop_duplicates(subset=["Headline"])
                        .sort_values("Date")
                        .reset_index(drop=True))
            added = len(combined) - before
        else:
            combined = new_df
            added    = len(combined)

        combined.to_csv(OUTPUT_PATH, index=False)

        # ── Near-duplicate removal across the full CSV ─────────────────────
        before_dedup = len(combined)
        combined = remove_near_duplicates(combined)
        dupes_removed = before_dedup - len(combined)
        if dupes_removed:
            print(f"Removed {dupes_removed} near-duplicate headlines")

        # ── Score new headlines with FinBERT ──────────────────────────────
        # Only score rows that don't already have a Sentiment_Score.
        # Skipped silently on the GitHub Actions runner where torch/transformers
        # are not installed (requirements-fetch.txt excludes them to keep the
        # runner fast). Scoring happens locally when you run this script
        # with a full environment.
        needs_scoring = (
            "Sentiment_Score" not in combined.columns
            or combined["Sentiment_Score"].isna().any()
        )
        if needs_scoring:
            try:
                import sys as _sys
                _sys.path.insert(0, BASE_DIR)
                from src.sentiment import analyze_sentiment as _score
                print("\nScoring headlines with FinBERT...")
                scored    = _score()
                score_map = scored.set_index("Headline")[["Sentiment", "Sentiment_Score"]]
                combined  = combined.copy()
                if "Sentiment" not in combined.columns:
                    combined["Sentiment"] = None
                if "Sentiment_Score" not in combined.columns:
                    combined["Sentiment_Score"] = None
                for idx, row in combined.iterrows():
                    if pd.isna(combined.at[idx, "Sentiment_Score"]):
                        hl = row["Headline"]
                        if hl in score_map.index:
                            combined.at[idx, "Sentiment"]       = score_map.at[hl, "Sentiment"]
                            combined.at[idx, "Sentiment_Score"] = score_map.at[hl, "Sentiment_Score"]
                print(f"Scored {combined['Sentiment_Score'].notna().sum()} / {len(combined)} headlines")
            except ImportError:
                print("FinBERT not available in this environment — skipping scoring.")
                print("Run fetch_news.py locally (with full requirements.txt) to score new headlines.")
        else:
            print("All headlines already scored — skipping FinBERT.")

        combined.to_csv(OUTPUT_PATH, index=False)

        print(f"\nNews CSV: {len(combined)} total headlines  (+{added} new today)")
        print(f"Date range: {combined['Date'].min().date()} → {combined['Date'].max().date()}")
        print(f"\nHeadlines per company:")
        print(combined["Company"].value_counts().to_string())

        # ── Keep stock prices in sync ─────────────────────────────────────
        update_stock_data()

    print("\nDone.")
