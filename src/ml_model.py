"""
ml_model.py
-----------
Trains a linear regression model to predict stock returns from FinBERT
sentiment scores, then evaluates it honestly.

Key design decisions:
  - TIME-BASED split (not random): trains on the earlier 70%, tests on the
    later 30%. Random splits are wrong for time-series — you'd be training on
    future data and testing on the past, which leaks information.
  - BASELINE comparison: always predicting the mean return is the dumbest
    possible model. If we can't beat that, our sentiment model is useless.
  - HONEST metrics: R² and directional accuracy on the TEST set only.
    Train-set performance is meaningless — any model can memorize its training
    data.
  - Continuous FinBERT score as feature (not the Positive/Neutral/Negative
    label bucketed into 1/0/-1), which throws away information.
"""

import pandas as pd
import os
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")


def load_merged_data() -> pd.DataFrame:
    """
    Load news.csv (which already has Sentiment_Score from FinBERT),
    merge with stock prices, aggregate per trading day.
    Reuses the same logic as analysis.py but standalone so ml_model.py
    has no dashboard dependency.
    """
    # --- News (scores already saved by fetch_news.py) ---
    news_df = pd.read_csv(NEWS_PATH, parse_dates=["Date"])

    if "Sentiment_Score" not in news_df.columns:
        raise RuntimeError(
            "news.csv has no Sentiment_Score column. "
            "Run fetch_news.py first to score the headlines."
        )

    # --- Stock prices (all companies averaged per date) ---
    files = glob.glob(os.path.join(RAW_DIR, "*_2026.csv"))
    if not files:
        files = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
                 if "news" not in os.path.basename(f).lower()]

    frames = []
    for f in files:
        df_s = pd.read_csv(f, parse_dates=["Date"])
        close_col = next(
            (c for c in df_s.columns if c.lower() in ("close", "price")), None
        )
        if close_col is None:
            continue
        df_s = df_s[["Date", close_col]].rename(columns={close_col: "Close"})
        df_s["Close"]  = pd.to_numeric(df_s["Close"], errors="coerce")
        df_s["Return"] = df_s["Close"].pct_change(fill_method=None)
        frames.append(df_s)

    stock_df = pd.concat(frames, ignore_index=True)
    daily_stock = stock_df.groupby("Date", as_index=False).agg(
        Return=("Return", "mean")
    )

    # --- Aggregate news per day (mean sentiment score) ---
    daily_news = (
        news_df
        .dropna(subset=["Sentiment_Score"])
        .groupby("Date", as_index=False)
        .agg(Sentiment_Score=("Sentiment_Score", "mean"))
    )

    # --- Merge ---
    df = pd.merge(daily_news, daily_stock, on="Date", how="inner")
    df = df.dropna(subset=["Sentiment_Score", "Return"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def train_and_evaluate(min_samples: int = 6) -> dict:
    """
    Train a linear regression on the earliest 70% of dates,
    evaluate on the remaining 30%.

    Returns a dict with all metrics so callers (dashboard, CLI) can
    display results however they like.
    """
    df = load_merged_data()
    n  = len(df)

    if n < min_samples:
        return {
            "status":  "insufficient_data",
            "n_total": n,
            "message": (
                f"Only {n} data points. Need at least {min_samples} to train. "
                "Run fetch_news.py daily to accumulate more headlines."
            ),
        }

    # ── Time-based split ──────────────────────────────────────────────────────
    # Sort by date (already sorted), take first 70% as train, last 30% as test.
    # This simulates real-world use: you train on historical data and predict
    # on future data you've never seen.
    split_idx  = int(n * 0.70)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    n_train = len(train_df)
    n_test  = len(test_df)

    if n_test < 2:
        # Not enough test data — report on all data with a warning
        train_df = df
        test_df  = df
        split_note = "insufficient test set — reporting on full dataset"
    else:
        split_note = f"train: {n_train} days ({train_df['Date'].min().date()} → {train_df['Date'].max().date()}), test: {n_test} days ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})"

    X_train = train_df[["Sentiment_Score"]].values
    y_train = train_df["Return"].values
    X_test  = test_df[["Sentiment_Score"]].values
    y_test  = test_df["Return"].values

    # ── Train model ───────────────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Baseline: always predict the mean return from the training set ────────
    # This is the simplest possible model. If we can't beat this, our
    # sentiment signal adds no value.
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())

    # ── Metrics ───────────────────────────────────────────────────────────────
    r2_model    = r2_score(y_test, y_pred)
    r2_baseline = r2_score(y_test, baseline_pred)

    # Directional accuracy: did the model predict up when it went up, and down
    # when it went down? (ignores magnitude, just checks the sign)
    # Baseline directional accuracy: always predicting "up" (positive mean return)
    mean_sign      = 1 if y_train.mean() >= 0 else -1
    dir_model      = np.mean(np.sign(y_pred) == np.sign(y_test))
    dir_baseline   = np.mean(mean_sign == np.sign(y_test))

    # MAE — mean absolute error, in return units (e.g. 0.01 = 1% off on average)
    mae_model    = float(np.mean(np.abs(y_pred - y_test)))
    mae_baseline = float(np.mean(np.abs(baseline_pred - y_test)))

    beats_baseline = r2_model > r2_baseline

    return {
        "status":          "ok",
        "n_total":         n,
        "n_train":         n_train,
        "n_test":          n_test,
        "split_note":      split_note,
        "coef":            float(model.coef_[0]),
        "intercept":       float(model.intercept_),
        "r2_model":        float(r2_model),
        "r2_baseline":     float(r2_baseline),
        "dir_model":       float(dir_model),
        "dir_baseline":    float(dir_baseline),
        "mae_model":       mae_model,
        "mae_baseline":    mae_baseline,
        "beats_baseline":  beats_baseline,
        "train_date_range": f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "test_date_range":  f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
    }


if __name__ == "__main__":
    result = train_and_evaluate()

    if result["status"] == "insufficient_data":
        print(f"\n[INFO] {result['message']}")
    else:
        print("\n" + "=" * 60)
        print("  ML MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"Total data points : {result['n_total']}")
        print(f"Split             : {result['split_note']}")
        print(f"Model coefficient : sentiment_score × {result['coef']:.6f} + {result['intercept']:.6f}")
        print()
        print(f"{'Metric':<30} {'Model':>10} {'Baseline':>10} {'Better?':>10}")
        print("-" * 60)
        print(f"{'R² (test set)':<30} {result['r2_model']:>10.3f} {result['r2_baseline']:>10.3f} {'✓' if result['beats_baseline'] else '✗':>10}")
        print(f"{'Directional accuracy':<30} {result['dir_model']:>10.1%} {result['dir_baseline']:>10.1%} {'✓' if result['dir_model'] > result['dir_baseline'] else '✗':>10}")
        print(f"{'MAE':<30} {result['mae_model']:>10.4f} {result['mae_baseline']:>10.4f} {'✓' if result['mae_model'] < result['mae_baseline'] else '✗':>10}")
        print()
        if result["beats_baseline"]:
            print("✓ Model BEATS the naive baseline on the test set.")
        else:
            print("✗ Model does NOT beat the naive baseline on the test set.")
            print("  This is expected with a small dataset — it does not mean")
            print("  the approach is wrong. Collect more data and re-evaluate.")
        print("=" * 60)
