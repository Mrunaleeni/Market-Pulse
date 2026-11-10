# Market Pulse — Indian Stock News Sentiment & Event Study

A data science project that answers two questions:

1. **Does news sentiment predict stock returns** for Indian large-cap companies?
2. **What historically happens after a specific type of event** — earnings beat, CEO change, analyst upgrade, regulatory action?

Built on a live data pipeline that collects headlines daily, scores them with a finance-trained NLP model, tags them by event type, and matches them to NSE stock prices. Results are displayed in a Streamlit dashboard with statistical rigour — always showing sample size, confidence level, and the actual past cases behind every number.

> ⚠️ Not financial advice. Built for learning and portfolio purposes only.

---

## Live Demo

Run locally — see [Setup](#setup) below.

---

## What It Does

### Data collection
- Pulls financial headlines daily from **GNews API**, **NewsAPI**, and **Economic Times RSS feeds**
- Filters each headline through alias validation (company name must appear in headline), junk detection (removes roundups, FD calculators, holiday notices), and near-duplicate removal using fuzzy matching
- Stores everything in `data/raw/news.csv` with no duplicates across runs

### Sentiment scoring
- **FinBERT** (`ProsusAI/finbert`) — a BERT model fine-tuned on financial text. Understands phrases like "beats estimates", "misses guidance", "raises target price" that generic models score incorrectly
- Falls back to **NLTK VADER** when FinBERT is not available
- Each headline gets a label (Positive / Neutral / Negative) and a continuous score from −1.0 to +1.0

### Event tagging
- 8 event categories, 100+ keyword/regex patterns, compiled at import for speed:
  - **Company Results** — earnings, revenue, dividend, sales data
  - **Corporate Action** — M&A, IPO, QIP, stake sale, market listing
  - **Management** — CEO/CFO change, appointment, resignation
  - **Analyst Action** — rating upgrade/downgrade, target price change
  - **Regulatory / Legal** — SEBI, penalty, lawsuit, fraud allegation
  - **Business News** — order win, product launch, tariff hike, partnership
  - **Sector & Policy** — PLI scheme, drug price control, subsidy
  - **Macro** — RBI rate decision, crude oil, rupee, FII flows, budget
- Optional zero-shot fallback using `facebook/bart-large-mnli` (`USE_ZERO_SHOT=true`)

### Analysis pipeline
- Joins headlines to stock prices on the correct trading day, handling weekends, holidays, and after-hours publication (news after 15:30 IST pushed to next trading day)
- Computes same-day return, next-day return, 3-day return, and 5-day return
- Multiple headlines on the same day are aggregated — mean sentiment score, mode event type — before merging, so busy news days do not get disproportionate weight

### Event study (abnormal returns)
- **AR = stock return − equal-weighted market return** (removes broad market moves)
- Computed at t+1, t+3, and t+5 trading days after the event
- Per event type: average AR, hit rate (% of times stock went up), standard deviation, P10/P90 range, case count, confidence level

### Machine learning models
All use a **time-based 70/30 split** — trained on oldest data, tested on newest. Never random split.

| Model | Predicts | Algorithm |
|---|---|---|
| Original | Return magnitude | Linear Regression (sentiment only) |
| Extended | Magnitude + direction | Ridge Regression + Logistic Regression (event type, sector, volatility) |
| Quantile | P10 / P50 / P90 range | QuantileRegressor at three quantiles |

---

## Companies Tracked (17 across 8 sectors)

| Sector | Companies |
|---|---|
| Technology | TCS, Infosys, Wipro |
| Banking | HDFC Bank, ICICI Bank, Axis Bank, SBI |
| Energy | Reliance Industries, ONGC |
| FMCG | Hindustan Unilever, ITC |
| Auto | Maruti Suzuki, Tata Motors |
| Pharma | Sun Pharma, Dr Reddy's |
| Metals & Infra | Tata Steel, L&T |
| Telecom | Bharti Airtel |

---

## Dashboard (4 tabs)

**Verdict** — one-sentence answer: did sentiment predict returns for the selected company? Green = positive signal, red = inverted, amber = no signal, grey = not enough data yet. Raw statistics hidden behind an expander.

**What If?** — scenario simulator. Pick a company, event type, and sentiment strength. Returns:
- Expected abnormal return in large type with a probability-of-going-up gauge
- Distribution dot-strip of every past case
- Timing chart (t+1 vs t+3 vs t+5 — does the reaction fade or build?)
- Event type ranking bar chart
- Event type × sector heatmap coloured by average AR
- Past-case cards (real headlines with actual outcomes)
- Peer spillover chart

**Proof** — box plot, scatter plot with OLS trendline, sentiment distribution pie, ML model card comparing directional accuracy vs naive baseline.

**Headlines Data** — raw merged table of every headline used, with FinBERT confidence filter slider.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit, Plotly |
| Sentiment | FinBERT (ProsusAI/finbert, HuggingFace Transformers), NLTK VADER |
| Data sources | yfinance (NSE prices), GNews API, NewsAPI, Economic Times RSS |
| ML | scikit-learn (LinearRegression, Ridge, LogisticRegression, QuantileRegressor) |
| NLP / dedup | rapidfuzz (fuzzy matching) |
| Stats | scipy (Pearson r, Spearman ρ), statsmodels (OLS trendline) |
| Data | pandas, numpy |
| Automation | GitHub Actions (daily fetch), Windows Task Scheduler |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Mrunaleeni/Market-Pulse.git
cd Market-Pulse

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

FinBERT (~440 MB) downloads automatically on first use and is cached locally.

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

---

## Daily Data Update

```bash
# Set your API keys
$env:GNEWS_API_KEY="your_gnews_key"
$env:NEWSAPI_KEY="your_newsapi_key"

# Fetch new headlines + update stock prices
python src/fetch_news.py

# Optional: backfill from additional sources
python src/backfill_newsapi.py   # NewsAPI (last 30 days)
python src/backfill_et.py        # Economic Times RSS
```

GNews key: free at [gnews.io](https://gnews.io)
NewsAPI key: free at [newsapi.org](https://newsapi.org)

GitHub Actions runs `fetch_news.py` automatically Monday–Friday at 08:00 IST.

---

## Project Structure

```
├── dashboard/
│   └── app.py                  # Streamlit dashboard — 4 tabs, verdict engine, Scenario tab
├── data/
│   └── raw/
│       ├── news.csv            # 791+ headlines with sentiment scores and event tags
│       └── *_2026.csv          # Daily OHLCV price files per company
├── src/
│   ├── analysis.py             # Core pipeline: load → merge → aggregate on trading day
│   ├── sentiment.py            # FinBERT + VADER backends
│   ├── events.py               # Event tagger — 8 categories, 100+ regex rules
│   ├── event_study.py          # Abnormal return computation and per-event statistics
│   ├── fetch_news.py           # GNews fetcher + stock price updater (daily)
│   ├── backfill_newsapi.py     # Historical backfill via NewsAPI (30 days)
│   ├── backfill_et.py          # Economic Times RSS + sitemap backfill
│   ├── ml_model.py             # Linear, Ridge, Logistic, Quantile regression models
│   ├── data_collection.py      # One-time yfinance historical downloader
│   ├── preprocessing.py        # Text cleaning utilities
│   └── visualization.py        # Standalone matplotlib chart
├── tests/
│   └── test_verdict.py         # Verdict engine edge-case tests
├── .github/
│   └── workflows/
│       └── daily_fetch.yml     # GitHub Actions: daily news + price update
├── requirements.txt
├── requirements-fetch.txt      # Lightweight CI deps (no torch/transformers)
└── README.md
```

---

## Key Design Decisions

**Time-based train/test split** — ML models train on the oldest 70% of dates and test on the newest 30%. Random splits would leak future data into training, making results look better than they are.

**After-hours rule** — News published after 15:30 IST (NSE close) is attributed to the next trading day, since it cannot affect that day's closing price.

**Always show n** — Every number on the dashboard is shown alongside the number of past cases behind it. Anything below 15 cases is flagged as low confidence and anything below 5 is hidden entirely.

**Abnormal return, not raw return** — The event study subtracts the equal-weighted market return from each stock's return, removing broad market moves so what remains is the company-specific reaction to the event.

---

## Notes

- **Tata Motors** uses `TMCV.NS` (commercial vehicles entity) — the original ticker was delisted after the October 2025 demerger
- **FinBERT is skipped in GitHub Actions CI** — `requirements-fetch.txt` excludes torch/transformers to keep the runner fast. Scores are pre-computed locally and committed with the data
- The `data/raw_cache/` directory caches ET RSS XML pages locally so re-running the ET backfill does not re-download already-fetched pages
