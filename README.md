# 📊 Market Pulse — Indian Stock Sentiment Analysis

A data science project that tracks whether financial news sentiment predicts stock returns for 17 Indian large-cap companies across 8 sectors.

Headlines are pulled daily from the GNews API, scored using **FinBERT** (a BERT model fine-tuned on financial text), and matched to NSE stock price data from yfinance. The Streamlit dashboard lets you switch between companies, view price trends, sentiment distributions, and a **Pearson correlation score** that shows how strongly sentiment and returns move together.

> ⚠️ Not financial advice. Built for learning and portfolio purposes only.

---

## 🔍 Features

- **17 companies** across Tech, Banking, Energy, FMCG, Auto, Pharma, Metals & Infra, and Telecom
- **Real news headlines** fetched daily from GNews API (replaces dummy data)
- **FinBERT sentiment scoring** — trained on financial text, understands phrases like "beats estimates" and "misses guidance" that VADER gets wrong
- **Per-company news filtering** — Reliance headlines only affect Reliance's chart, not Infosys
- **Weekend forward-fill** — Saturday/Sunday headlines are matched to the next Monday's price instead of being dropped
- **Pearson r correlation** with sample size shown — one clear number for "does sentiment predict returns?"
- **Multi-stock dropdown** — switch between all 17 companies from the sidebar
- **FinBERT cached on startup** — scores all headlines once, company switches take 30–65ms
- **Daily auto-refresh** via Windows Task Scheduler (runs at 8 AM, updates both news and stock prices)
- **Clean logging** — no debug noise in the dashboard

---

## 🏢 Companies Tracked

| Sector | Companies |
|--------|-----------|
| Technology | TCS, Infosys |
| Banking & Finance | HDFC Bank, ICICI Bank, Axis Bank, SBI |
| Energy | Reliance Industries, ONGC |
| FMCG | Hindustan Unilever, ITC |
| Auto | Maruti Suzuki, Tata Motors |
| Pharma | Sun Pharma, Dr Reddy's |
| Metals & Infrastructure | Tata Steel, L&T |
| Telecom | Bharti Airtel |

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Dashboard | Streamlit, Plotly |
| Sentiment | FinBERT (ProsusAI/finbert via HuggingFace Transformers), NLTK VADER (fallback) |
| Data | yfinance (stock prices), GNews API (headlines) |
| ML | scikit-learn (Linear Regression, Pearson r) |
| Data handling | pandas, numpy |
| Stats | statsmodels (OLS trendline) |

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/Mrunaleeni/Stock-Sentimient-Analysis.git
cd Stock-Sentimient-Analysis

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

FinBERT (~440 MB) downloads automatically on first run and is cached locally after that.

---

## 📈 Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

---

## 🔄 Refresh News & Stock Data

```bash
python src/fetch_news.py
```

This appends new headlines to `data/raw/news.csv` (no duplicates) and pulls the latest stock prices for all 17 companies. A Windows Task Scheduler job runs this automatically at 8 AM daily.

---

## 📂 Project Structure

```
├── dashboard/
│   └── app.py              # Streamlit dashboard (stock picker, charts, Pearson r)
├── data/
│   └── raw/
│       ├── news.csv         # Real headlines with Company + Date columns
│       └── *_2026.csv       # Per-company NSE stock price files
├── src/
│   ├── analysis.py          # Core pipeline: load → score → filter → merge
│   ├── sentiment.py         # FinBERT + VADER backends, get_sentiment_score()
│   ├── fetch_news.py        # GNews API fetcher + stock data updater
│   ├── ml_model.py          # Linear regression (sentiment score → return)
│   ├── data_collection.py   # One-time yfinance historical downloader
│   ├── preprocessing.py     # Text cleaning utilities
│   └── visualization.py     # Standalone matplotlib chart
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Tabs

**Price** — line chart of the selected stock's closing price over time

**Analysis** — box plot of returns by sentiment label, scatter plot of sentiment score vs return with OLS trendline, sentiment distribution pie chart

**Headlines** — table of all headlines used for the selected company with their computed sentiment labels

---

## 🔢 Understanding the Correlation Score

The **Sentiment↔Return** metric on the dashboard is the Pearson r between the continuous FinBERT confidence score and the same-day stock return.

| r value | Meaning |
|---------|---------|
| < 0.1 | No relationship |
| 0.1 – 0.3 | Weak |
| 0.3 – 0.5 | Moderate |
| 0.5 – 0.7 | Strong |
| > 0.7 | Very strong |

The sample size (n = X headlines) is shown alongside the score. With fewer than ~30 headlines per company, treat the number as indicative rather than conclusive — it will become more reliable as data accumulates over weeks.

---

## ⚙️ Configuration

**Switch sentiment backend:**
```powershell
$env:SENTIMENT_MODEL = "vader"    # faster, less accurate
$env:SENTIMENT_MODEL = "finbert"  # default, slower on first load
```

**Enable debug logging:**
```powershell
python src/analysis.py RELIANCE   # runs pipeline and prints diagnostics
python src/sentiment.py finbert   # scores all headlines and prints results
```

---

## 📌 Notes

- **Tata Motors** uses `TMCV.NS` (commercial vehicles entity) — the original `TATAMOTORS.NS` was delisted after the October 2025 demerger
- **ITC and Tata Steel** may show `N/A` for correlation initially — GNews has thin English coverage for these; data builds up over time with daily fetches
- The daily scheduler uses Windows Task Scheduler (`StockSentimentNewsFetch` task) — visible in Task Scheduler app
