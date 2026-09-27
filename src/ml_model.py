import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from src.sentiment import analyze_sentiment

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")

# Sentiment map
sentiment_map = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

def train_model():
    # Load news data with computed sentiment (scales with any number of rows)
    news_df = analyze_sentiment()
    news_df["Sentiment_Score"] = news_df["Sentiment"].map(sentiment_map)
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # Load all available stock CSVs — prefer *_2026.csv (matches news date range)
    files = glob.glob(os.path.join(RAW_DIR, "*_2026.csv"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
                 if "news" not in os.path.basename(f).lower()]

    frames = []
    for f in files:
        df_s = pd.read_csv(f)
        if isinstance(df_s.columns, pd.core.indexes.multi.MultiIndex):
            df_s.columns = [c[0] for c in df_s.columns]
        if "Date" not in df_s.columns:
            df_s.reset_index(inplace=True)
            df_s.rename(columns={"index": "Date"}, inplace=True)
        close_col = next((c for c in df_s.columns if c.lower() in ("close", "price")), None)
        if close_col is None:
            continue
        df_s = df_s[["Date", close_col]].rename(columns={close_col: "Close"})
        df_s["Date"]  = pd.to_datetime(df_s["Date"], errors="coerce")
        df_s["Close"] = pd.to_numeric(df_s["Close"], errors="coerce")
        df_s["Return"] = df_s["Close"].pct_change(fill_method=None)
        frames.append(df_s)

    stock_df = pd.concat(frames, ignore_index=True)
    # Average return across all tickers per date
    stock_df = stock_df.groupby("Date", as_index=False).agg(
        Close=("Close", "mean"), Return=("Return", "mean")
    )

    # Merge on Date
    df = pd.merge(news_df, stock_df, on="Date", how="inner")
    df = df.dropna(subset=["Sentiment_Score", "Return"])

    if df.empty:
        print("No overlapping dates between news and stock data — cannot train model.")
        return None

    X = df[["Sentiment_Score"]]
    y = df["Return"]

    model = LinearRegression()
    model.fit(X, y)

    print(f"\nModel trained on {len(df)} samples.")
    # Use a DataFrame for prediction to avoid sklearn feature-name warnings
    import pandas as _pd
    pred_df = _pd.DataFrame({"Sentiment_Score": [1, -1]})
    preds = model.predict(pred_df)
    print(f"Predicted return (Positive sentiment): {preds[0]:.4%}")
    print(f"Predicted return (Negative sentiment): {preds[1]:.4%}")

    return model

if __name__ == "__main__":
    train_model()
