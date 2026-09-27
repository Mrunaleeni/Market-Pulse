# 📊 Stock Sentiment Analysis

This project analyzes how financial news sentiment affects stock prices.

## 🔍 Features
- Stock data collection using yfinance
- News sentiment analysis using NLP (VADER)
- Data merging and return analysis
- Interactive dashboard using Streamlit

## 🛠 Tech Stack
- Python 3.11
- Pandas
- NLTK (VADER)
- Streamlit
- Plotly
- scikit-learn
- yfinance

## 🚀 Setup

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK sentiment data (one-time)
python -c "import nltk; nltk.download('vader_lexicon')"
```

## 📈 Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## ⚠️ Known Limitation: News Data

`data/raw/news.csv` currently contains **5 dummy headlines** covering only Jan 3–7, 2022.
This means the dashboard and ML model will only show data for those 5 dates.

**Options for adding real news data (not yet implemented — pick one before proceeding):**

| Option | Source | Notes |
|--------|--------|-------|
| NewsAPI | [newsapi.org](https://newsapi.org) | Free tier: 100 req/day, 1 month history |
| GNews API | [gnews.io](https://gnews.io) | Free tier available, good for Indian stocks |
| RSS feeds | Moneycontrol / ET Markets | No API key needed, scraping required |
| Kaggle datasets | [kaggle.com/datasets](https://kaggle.com/datasets) | Pre-collected financial news CSVs available |

Until this is addressed, the sentiment vs return analysis reflects only those 5 synthetic data points and should not be used for any real investment decisions.

## 📂 Project Structure

```
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── data/
│   └── raw/                # Stock CSVs + news.csv
├── src/
│   ├── analysis.py         # Core merge + sentiment logic
│   ├── data_collection.py  # yfinance downloader
│   ├── ml_model.py         # Linear regression model
│   ├── news_data.py        # News loader
│   ├── preprocessing.py    # Text cleaning
│   ├── sentiment.py        # VADER sentiment scoring
│   └── visualization.py    # Matplotlib stock chart
├── requirements.txt
└── README.md
```
