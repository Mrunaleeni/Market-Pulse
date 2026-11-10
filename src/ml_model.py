"""
ml_model.py
-----------
Three models, all time-split, all compared against a naive baseline:

  train_and_evaluate()      — original linear regression (sentiment score only).
                              Called by the existing dashboard Proof tab card.
                              Unchanged so nothing breaks.

  train_extended_model()    — Ridge regression with event type one-hot features
                              + sentiment score + sector + prev-day return +
                              recent volatility + market return that day.
                              Returns direction (logistic) AND magnitude (Ridge).

  train_quantile_model()    — Quantile regression at q=0.10 / 0.50 / 0.90
                              so the Scenario tab can show a best/worst range.

  get_model_suite()         — Trains all three and returns a bundle dict.
                              Cached by the dashboard.

Key design decisions (unchanged from original):
  - TIME-BASED split (not random): trains on the earlier 70%, tests on the
    later 30%. Random splits leak future data into training.
  - BASELINE comparison: always predicting the mean return is the dumbest
    possible model. If we can't beat that, our event features add no value.
  - HONEST metrics on the TEST set only.
  - Continuous FinBERT score as feature (not the bucketed label).
"""

import logging
import pandas as pd
import os
import glob
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing  import OneHotEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.metrics         import r2_score, accuracy_score

log = logging.getLogger("ml_model")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")


# ── Data loader (original, kept for train_and_evaluate) ───────────────────────

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


# ── Extended data loader (adds event + sector features) ───────────────────────

def load_extended_data(return_col: str = "ret_next_day") -> pd.DataFrame:
    """
    Load the full analysis pipeline output (all companies), add event types,
    and engineer the feature set for the extended models.

    Features built here
    -------------------
    sentiment_score   : FinBERT continuous score [-1, +1]
    event_type        : categorical — one-hot encoded downstream
    sector            : categorical — one-hot encoded downstream
    prev_return       : stock return on the previous trading day (momentum)
    volatility_5d     : rolling 5-day std of returns (risk proxy)
    market_return     : equal-weighted market return that day (same as AR baseline)

    Target columns
    --------------
    ret_next_day / ret_3day / ret_5day (caller picks via return_col)
    direction         : 1 if target > 0 else 0  (for logistic model)
    """
    # Use analyze_all_companies() so Event_Type and Sector flow through
    import sys as _sys
    _sys.path.insert(0, BASE_DIR)
    from src.analysis import analyze_all_companies

    df = analyze_all_companies()
    if df.empty:
        raise RuntimeError("analyze_all_companies() returned an empty DataFrame.")

    # On-the-fly event tagging if column missing
    if "Event_Type" not in df.columns or df["Event_Type"].isna().any():
        try:
            from src.events import tag_dataframe
            df = tag_dataframe(df)
        except ImportError:
            df["Event_Type"] = "unclassified"

    # Ensure Sector column
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    # Derive ret_5day if missing
    if "ret_5day" not in df.columns:
        frames = []
        for ticker, grp in df.groupby("Ticker") if "Ticker" in df.columns else [("all", df)]:
            grp = grp.sort_values("Date").copy()
            grp["ret_5day"] = grp["Close"].pct_change(5).shift(-5) if "Close" in grp.columns else np.nan
            frames.append(grp)
        df = pd.concat(frames).sort_values("Date").reset_index(drop=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = df.sort_values(["Ticker", "Date"] if "Ticker" in df.columns else ["Date"])

    # Previous-day return (per ticker)
    if "Ticker" in df.columns:
        df["prev_return"] = df.groupby("Ticker")["Return"].shift(1)
    else:
        df["prev_return"] = df["Return"].shift(1)

    # 5-day rolling volatility of returns (per ticker)
    if "Ticker" in df.columns:
        df["volatility_5d"] = (
            df.groupby("Ticker")["Return"]
              .transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
        )
    else:
        df["volatility_5d"] = df["Return"].shift(1).rolling(5, min_periods=2).std()

    # Market return that day = equal-weighted mean of all tickers' returns
    if "Ticker" in df.columns:
        mkt = df.groupby("Date")["Return"].mean().rename("market_return")
        df = df.join(mkt, on="Date")
    else:
        df["market_return"] = df["Return"]

    # Direction label for logistic regression
    if return_col in df.columns:
        df["direction"] = (df[return_col] > 0).astype(int)

    # Drop rows with missing target or key features
    required = ["Sentiment_Score", "Event_Type", "Sector",
                "prev_return", "volatility_5d", "market_return",
                return_col, "direction"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df = df.sort_values("Date").reset_index(drop=True)

    log.info("load_extended_data: %d rows, target=%s", len(df), return_col)
    return df


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame):
    """
    Build X (feature matrix) and the ColumnTransformer preprocessor.
    Returns (X_raw_df, preprocessor) — fit is done outside so train/test
    stay separate.

    Numerical features  : sentiment_score, prev_return, volatility_5d,
                          market_return
    Categorical features: event_type, sector  (one-hot, drop first for Ridge)
    """
    num_features = ["Sentiment_Score", "prev_return", "volatility_5d", "market_return"]
    cat_features = ["Event_Type", "Sector"]

    # Keep only columns that exist in df
    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]

    X = df[num_features + cat_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  drop="first"), cat_features),
        ],
        remainder="drop",
    )
    return X, preprocessor


# ── Time-based split helper ───────────────────────────────────────────────────

def _time_split(df: pd.DataFrame, train_frac: float = 0.70):
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ── Original model (unchanged, called by existing dashboard Proof tab) ────────────

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

    split_idx  = int(n * 0.70)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    n_train = len(train_df)
    n_test  = len(test_df)

    if n_test < 2:
        train_df   = df
        test_df    = df
        split_note = "insufficient test set — reporting on full dataset"
    else:
        split_note = (
            f"train: {n_train} days "
            f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()}), "
            f"test: {n_test} days "
            f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})"
        )

    X_train = train_df[["Sentiment_Score"]].values
    y_train = train_df["Return"].values
    X_test  = test_df[["Sentiment_Score"]].values
    y_test  = test_df["Return"].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())

    r2_model    = r2_score(y_test, y_pred)
    r2_baseline = r2_score(y_test, baseline_pred)

    mean_sign    = 1 if y_train.mean() >= 0 else -1
    dir_model    = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
    dir_baseline = float(np.mean(mean_sign == np.sign(y_test)))

    mae_model    = float(np.mean(np.abs(y_pred - y_test)))
    mae_baseline = float(np.mean(np.abs(baseline_pred - y_test)))

    return {
        "status":           "ok",
        "n_total":          n,
        "n_train":          n_train,
        "n_test":           n_test,
        "split_note":       split_note,
        "coef":             float(model.coef_[0]),
        "intercept":        float(model.intercept_),
        "r2_model":         float(r2_model),
        "r2_baseline":      float(r2_baseline),
        "dir_model":        dir_model,
        "dir_baseline":     dir_baseline,
        "mae_model":        mae_model,
        "mae_baseline":     mae_baseline,
        "beats_baseline":   r2_model > r2_baseline,
        "train_date_range": f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "test_date_range":  f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
    }


# ── Extended model (Ridge + Logistic, with event features) ────────────────────

def train_extended_model(return_col: str = "ret_next_day",
                         min_samples: int = 20) -> dict:
    """
    Train two models on the extended feature set
    (sentiment + event type + sector + prev return + volatility + market return):

      ridge_model   : Ridge regression  → predicts magnitude of return
      logit_model   : Logistic regression → predicts direction (up/down)

    Both use a time-based 70/30 split and are compared against naive baselines.

    Returns
    -------
    dict with keys:
      status, n_total, n_train, n_test, split_note,
      ridge_r2, ridge_r2_baseline, ridge_mae, ridge_mae_baseline,
      logit_accuracy, logit_accuracy_baseline,
      beats_baseline_ridge, beats_baseline_logit,
      feature_names,
      ridge_model_obj, logit_model_obj, preprocessor_obj,
      train_date_range, test_date_range
    """
    try:
        df = load_extended_data(return_col=return_col)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    n = len(df)
    if n < min_samples:
        return {
            "status":  "insufficient_data",
            "n_total": n,
            "message": (
                f"Only {n} rows with complete features. "
                f"Need at least {min_samples}. "
                "Collect more data and re-run."
            ),
        }

    train_df, test_df = _time_split(df)
    n_train, n_test   = len(train_df), len(test_df)

    if n_test < 5:
        # Not enough test data; use full set with a warning
        train_df   = df
        test_df    = df
        split_note = "insufficient test set — reporting on full dataset"
    else:
        split_note = (
            f"train: {n_train} rows "
            f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()}), "
            f"test: {n_test} rows "
            f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})"
        )

    # Build features
    X_train_raw, preprocessor = _build_feature_matrix(train_df)
    X_test_raw,  _            = _build_feature_matrix(test_df)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    y_train_ret = train_df[return_col].values
    y_test_ret  = test_df[return_col].values
    y_train_dir = train_df["direction"].values
    y_test_dir  = test_df["direction"].values

    # ── Ridge regression ──────────────────────────────────────────────────────
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_ret)
    y_pred_ret = ridge.predict(X_test)

    baseline_ret    = np.full_like(y_test_ret, fill_value=y_train_ret.mean())
    ridge_r2        = float(r2_score(y_test_ret, y_pred_ret))
    ridge_r2_base   = float(r2_score(y_test_ret, baseline_ret))
    ridge_mae       = float(np.mean(np.abs(y_pred_ret - y_test_ret)))
    ridge_mae_base  = float(np.mean(np.abs(baseline_ret - y_test_ret)))

    # ── Logistic regression (direction) ───────────────────────────────────────
    # Guard: need both classes in training set
    beats_logit = False
    logit_acc = logit_acc_base = None
    logit = None
    if len(np.unique(y_train_dir)) >= 2:
        logit = LogisticRegression(max_iter=500, C=1.0)
        logit.fit(X_train, y_train_dir)
        y_pred_dir      = logit.predict(X_test)
        logit_acc       = float(accuracy_score(y_test_dir, y_pred_dir))
        # Baseline: always predict the majority class from training
        majority_class  = int(np.bincount(y_train_dir).argmax())
        baseline_dir    = np.full_like(y_test_dir, fill_value=majority_class)
        logit_acc_base  = float(accuracy_score(y_test_dir, baseline_dir))
        beats_logit     = logit_acc > logit_acc_base
    else:
        log.warning("Only one direction class in training set — skipping logistic model.")

    # ── Feature names (for explainability) ───────────────────────────────────
    try:
        num_names = [t[2] for t in preprocessor.transformers_ if t[0] == "num"][0]
        cat_enc   = preprocessor.named_transformers_.get("cat")
        cat_names = list(cat_enc.get_feature_names_out()) if cat_enc else []
        feature_names = num_names + cat_names
    except Exception:
        feature_names = []

    return {
        "status":                  "ok",
        "n_total":                 n,
        "n_train":                 n_train,
        "n_test":                  n_test,
        "split_note":              split_note,
        "ridge_r2":                ridge_r2,
        "ridge_r2_baseline":       ridge_r2_base,
        "ridge_mae":               ridge_mae,
        "ridge_mae_baseline":      ridge_mae_base,
        "beats_baseline_ridge":    ridge_r2 > ridge_r2_base,
        "logit_accuracy":          logit_acc,
        "logit_accuracy_baseline": logit_acc_base,
        "beats_baseline_logit":    beats_logit,
        "feature_names":           feature_names,
        "ridge_model_obj":         ridge,
        "logit_model_obj":         logit,
        "preprocessor_obj":        preprocessor,
        "return_col":              return_col,
        "train_date_range":        f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "test_date_range":         f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
    }


# ── Quantile model (10th / 50th / 90th percentile range) ──────────────────────

def train_quantile_model(return_col: str = "ret_next_day",
                         min_samples: int = 30) -> dict:
    """
    Train three QuantileRegressor models (q=0.10, 0.50, 0.90) on the extended
    feature set.  These give the Scenario tab its best/median/worst-case range.

    QuantileRegressor minimises the pinball loss — it is a proper quantile
    estimator and will not cross (p10 ≤ p50 ≤ p90) on average.

    Returns
    -------
    dict with keys:
      status, n_total, n_train, n_test, split_note,
      q10_model_obj, q50_model_obj, q90_model_obj,
      preprocessor_obj, return_col,
      train_date_range, test_date_range
    """
    try:
        df = load_extended_data(return_col=return_col)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    n = len(df)
    if n < min_samples:
        return {
            "status":  "insufficient_data",
            "n_total": n,
            "message": (
                f"Only {n} rows with complete features — need {min_samples} "
                "for reliable quantile estimates."
            ),
        }

    train_df, test_df = _time_split(df)

    X_train_raw, preprocessor = _build_feature_matrix(train_df)
    X_test_raw,  _            = _build_feature_matrix(test_df)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    y_train = train_df[return_col].values

    models = {}
    for q in [0.10, 0.50, 0.90]:
        qr = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
        try:
            qr.fit(X_train, y_train)
            models[q] = qr
        except Exception as e:
            log.warning("QuantileRegressor q=%.2f failed: %s", q, e)
            models[q] = None

    return {
        "status":         "ok",
        "n_total":        n,
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "split_note":     (
            f"train: {len(train_df)} rows "
            f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()}), "
            f"test: {len(test_df)} rows "
            f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})"
        ),
        "q10_model_obj":  models.get(0.10),
        "q50_model_obj":  models.get(0.50),
        "q90_model_obj":  models.get(0.90),
        "preprocessor_obj": preprocessor,
        "return_col":     return_col,
        "train_date_range": f"{train_df['Date'].min().date()} → {train_df['Date'].max().date()}",
        "test_date_range":  f"{test_df['Date'].min().date()} → {test_df['Date'].max().date()}",
    }


# ── Scenario inference helper ─────────────────────────────────────────────────

def predict_from_models(model_suite: dict,
                        event_type: str,
                        sentiment_score: float,
                        sector: str = "Unknown",
                        prev_return: float = 0.0,
                        volatility_5d: float = 0.01,
                        market_return: float = 0.0) -> dict:
    """
    Given a trained model_suite (output of get_model_suite()), produce a
    prediction for a hypothetical scenario.

    Returns
    -------
    dict with:
      direction_prob  : probability of positive return (0–1) from logistic model
      magnitude       : expected return from Ridge model
      q10 / q50 / q90 : quantile range from quantile model
      (None for any model that wasn't trained successfully)
    """
    result = {
        "direction_prob": None,
        "magnitude":      None,
        "q10":            None,
        "q50":            None,
        "q90":            None,
    }

    # Build a single-row feature DataFrame matching training schema
    row = pd.DataFrame([{
        "Sentiment_Score": sentiment_score,
        "prev_return":     prev_return,
        "volatility_5d":   volatility_5d,
        "market_return":   market_return,
        "Event_Type":      event_type,
        "Sector":          sector,
    }])

    # Extended model predictions
    ext = model_suite.get("extended", {})
    if ext.get("status") == "ok" and ext.get("preprocessor_obj") is not None:
        prep = ext["preprocessor_obj"]
        X_row_raw, _ = _build_feature_matrix(row)
        try:
            X_row = prep.transform(X_row_raw)
        except Exception:
            return result   # feature mismatch — can't predict

        if ext.get("ridge_model_obj") is not None:
            result["magnitude"] = float(ext["ridge_model_obj"].predict(X_row)[0])

        if ext.get("logit_model_obj") is not None:
            proba = ext["logit_model_obj"].predict_proba(X_row)
            # proba shape: (1, 2), index 1 = probability of class 1 (up)
            result["direction_prob"] = float(proba[0][1])

    # Quantile model predictions
    qnt = model_suite.get("quantile", {})
    if qnt.get("status") == "ok" and qnt.get("preprocessor_obj") is not None:
        prep = qnt["preprocessor_obj"]
        X_row_raw, _ = _build_feature_matrix(row)
        try:
            X_row = prep.transform(X_row_raw)
        except Exception:
            return result

        for q_key, q_model_key in [("q10", "q10_model_obj"),
                                    ("q50", "q50_model_obj"),
                                    ("q90", "q90_model_obj")]:
            m = qnt.get(q_model_key)
            if m is not None:
                result[q_key] = float(m.predict(X_row)[0])

    return result


# ── Bundle: train all models once, return together ────────────────────────────

def get_model_suite(return_col: str = "ret_next_day") -> dict:
    """
    Train (or attempt to train) all three model families and return a bundle.
    The dashboard caches this with @st.cache_data.

    Returns
    -------
    {
      "original":  result of train_and_evaluate(),
      "extended":  result of train_extended_model(return_col),
      "quantile":  result of train_quantile_model(return_col),
    }
    """
    log.info("Training model suite for return_col=%s …", return_col)

    original = train_and_evaluate()

    try:
        extended = train_extended_model(return_col=return_col)
    except Exception as e:
        log.warning("Extended model failed: %s", e)
        extended = {"status": "error", "message": str(e)}

    try:
        quantile = train_quantile_model(return_col=return_col)
    except Exception as e:
        log.warning("Quantile model failed: %s", e)
        quantile = {"status": "error", "message": str(e)}

    return {
        "original": original,
        "extended": extended,
        "quantile": quantile,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    result = train_and_evaluate()

    if result["status"] == "insufficient_data":
        print(f"\n[INFO] {result['message']}")
    else:
        print("\n" + "=" * 60)
        print("  ORIGINAL MODEL (sentiment score only)")
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
    print("=" * 60)
    print("  EXTENDED MODEL (event + sector + risk features)")
    print("=" * 60)
    ext = train_extended_model()
    if ext["status"] != "ok":
        print(f"  Status: {ext['status']} — {ext.get('message', '')}")
    else:
        print(f"  n={ext['n_total']}, split: {ext['split_note']}")
        print(f"  Ridge R²: {ext['ridge_r2']:.3f}  (baseline: {ext['ridge_r2_baseline']:.3f})  {'✓' if ext['beats_baseline_ridge'] else '✗'}")
        if ext["logit_accuracy"] is not None:
            print(f"  Logit accuracy: {ext['logit_accuracy']:.1%}  (baseline: {ext['logit_accuracy_baseline']:.1%})  {'✓' if ext['beats_baseline_logit'] else '✗'}")

    print()
    print("=" * 60)
    print("  QUANTILE MODEL (P10 / P50 / P90 range)")
    print("=" * 60)
    qnt = train_quantile_model()
    if qnt["status"] != "ok":
        print(f"  Status: {qnt['status']} — {qnt.get('message', '')}")
    else:
        print(f"  n={qnt['n_total']}, trained. Models ready for scenario prediction.")
    print("=" * 60)


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
