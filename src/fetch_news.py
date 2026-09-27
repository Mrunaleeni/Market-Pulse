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

API_KEY = "e487a418fd484c9fae5cec36d1806c9c"
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
    return clean


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

        print(f"\nNews CSV: {len(combined)} total headlines  (+{added} new today)")
        print(f"Date range: {combined['Date'].min().date()} → {combined['Date'].max().date()}")
        print(f"\nHeadlines per company:")
        print(combined["Company"].value_counts().to_string())

        # ── Keep stock prices in sync ─────────────────────────────────────
        update_stock_data()

    print("\nDone.")
