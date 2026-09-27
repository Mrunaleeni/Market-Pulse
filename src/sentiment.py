"""
sentiment.py
------------
Provides sentiment scoring for financial headlines using two backends:

  "vader"   — NLTK VADER (fast, no GPU needed, works offline)
  "finbert" — ProsusAI/finbert (BERT model fine-tuned on financial text,
               ~440 MB one-time download, significantly better on financial
               phrases like "beats estimates", "misses guidance", etc.)

The active backend is controlled by the SENTIMENT_MODEL environment variable
or by passing model="finbert" directly to get_sentiment() / analyze_sentiment().
Default is "finbert" if transformers+torch are available, otherwise "vader".

Logging
-------
Uses the standard 'sentiment' logger (child of the root logger).
Set LOG_LEVEL=DEBUG in your environment to see per-headline scores and
model-load messages. The dashboard sees nothing by default.
"""

import logging
import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

log = logging.getLogger("sentiment")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")

# ── VADER (always available) ──────────────────────────────────────────────────
_sia = SentimentIntensityAnalyzer()

def _vader_sentiment(text: str) -> str:
    compound = _sia.polarity_scores(text)["compound"]
    if compound > 0.05:
        return "Positive"
    elif compound < -0.05:
        return "Negative"
    return "Neutral"

# ── FinBERT (lazy-loaded on first use) ────────────────────────────────────────
_finbert_pipeline = None

def _load_finbert():
    """Download (first time) and cache the FinBERT pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        import transformers as _tf
        _tf.logging.set_verbosity_error()   # suppress HF model-load chatter
        import warnings as _w
        _w.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
        log.info("Loading FinBERT model (ProsusAI/finbert) — "
                 "first run downloads ~440 MB, subsequent runs are instant...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        log.info("FinBERT ready.")
    except Exception as e:
        log.warning("Could not load FinBERT (%s). Falling back to VADER.", e)
        _finbert_pipeline = None

    return _finbert_pipeline

# FinBERT returns labels: "positive", "negative", "neutral"
_FINBERT_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
}

def _finbert_sentiment(text: str) -> str:
    pipe = _load_finbert()
    if pipe is None:
        return _vader_sentiment(text)
    result = pipe(text)[0]
    label  = result["label"].lower()
    return _FINBERT_LABEL_MAP.get(label, "Neutral")

# ── Auto-detect best available backend ───────────────────────────────────────
def _default_model() -> str:
    """Use FinBERT if transformers+torch are installed, else VADER."""
    env_override = os.environ.get("SENTIMENT_MODEL", "").lower()
    if env_override in ("vader", "finbert"):
        return env_override
    try:
        import transformers  # noqa: F401
        import torch          # noqa: F401
        return "finbert"
    except ImportError:
        return "vader"

# ── Public API ────────────────────────────────────────────────────────────────
def get_sentiment(text: str, model: str | None = None) -> str:
    """
    Return "Positive", "Negative", or "Neutral" for a given text.

    Parameters
    ----------
    text  : the headline string
    model : "vader" | "finbert" | None
            None → auto-detect (finbert if available, else vader)
    """
    chosen = (model or _default_model()).lower()
    if chosen == "finbert":
        return _finbert_sentiment(text)
    return _vader_sentiment(text)


def get_sentiment_score(text: str, model: str | None = None) -> float:
    """
    Return a continuous sentiment score in [-1.0, +1.0].

    VADER   → raw compound score directly
    FinBERT → confidence score, signed: +score for positive, -score for negative,
              0.0 for neutral
    """
    chosen = (model or _default_model()).lower()

    if chosen == "finbert":
        pipe = _load_finbert()
        if pipe is not None:
            result = pipe(text)[0]
            label  = result["label"].lower()
            score  = result["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0.0
    return _sia.polarity_scores(text)["compound"]


def analyze_sentiment(model: str | None = None) -> pd.DataFrame:
    """
    Load news.csv, score every headline, return the DataFrame with
    'Sentiment' and 'Sentiment_Score' columns added.

    Parameters
    ----------
    model : "vader" | "finbert" | None  (see get_sentiment)
    """
    chosen = model or _default_model()
    df = pd.read_csv(NEWS_PATH)

    log.info("Running sentiment analysis with: %s  (%d headlines)", chosen.upper(), len(df))

    if chosen == "finbert":
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
                if label == "positive":  return  r["score"]
                if label == "negative":  return -r["score"]
                return 0.0
            df["Sentiment_Score"] = [_signed(r) for r in results]
        else:
            df["Sentiment"]       = df["Headline"].apply(_vader_sentiment)
            df["Sentiment_Score"] = df["Headline"].apply(
                lambda t: _sia.polarity_scores(t)["compound"]
            )
    else:
        df["Sentiment"]       = df["Headline"].apply(_vader_sentiment)
        df["Sentiment_Score"] = df["Headline"].apply(
            lambda t: _sia.polarity_scores(t)["compound"]
        )

    log.debug("Sentiment counts:\n%s", df["Sentiment"].value_counts().to_string())
    return df


if __name__ == "__main__":
    import sys

    # When run directly, show INFO+ so the user sees model-load progress
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s  %(message)s",
    )

    m = sys.argv[1] if len(sys.argv) > 1 else None
    result = analyze_sentiment(model=m)

    # Print results to stdout — intentional for CLI use
    print(f"\nSentiment Analysis Result ({m or _default_model()}):\n")
    try:
        print(result[["Headline", "Sentiment", "Sentiment_Score"]].to_string())
    except UnicodeEncodeError:
        print(
            result[["Headline", "Sentiment", "Sentiment_Score"]]
            .to_string()
            .encode("ascii", errors="replace")
            .decode("ascii")
        )
