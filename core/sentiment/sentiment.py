import torch
import pandas as pd
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger(__name__)


# ---------------------------------------------------
# 🔥 SINGLETON MODEL LOADER
# ---------------------------------------------------

class FinBERTSingleton:
    """
    Loads FinBERT only ONCE per process.

    Prevents:
    - duplicate memory usage
    - slow container startup
    - worker crashes
    """

    _tokenizer = None
    _model = None

    MODEL_NAME = "ProsusAI/finbert"

    @classmethod
    def load(cls):

        if cls._model is None:

            logger.info("Loading FinBERT model (one-time initialization)...")

            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_NAME
            )

            cls._model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME
            )

            cls._model.eval()

        return cls._tokenizer, cls._model


# ---------------------------------------------------
# SENTIMENT ANALYZER
# ---------------------------------------------------

class SentimentAnalyzer:
    """
    Financial news sentiment analyzer using FinBERT.
    Production optimized.
    """

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def __init__(self):

        # 🔥 loads instantly after first call
        self.tokenizer, self.model = FinBERTSingleton.load()

    # ---------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment for a single text string.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]

        label_id = torch.argmax(scores).item()
        label = self.label_map[label_id]

        sentiment_score = scores[2].item() - scores[0].item()

        return {
            "label": label,
            "score": round(sentiment_score, 4)
        }

    # ---------------------------------------------------

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of news.
        Expects column: 'headline'
        """

        if "headline" not in df.columns:
            raise ValueError("DataFrame must contain 'headline' column")

        results = []

        for _, row in df.iterrows():
            sentiment = self.analyze_text(str(row["headline"]))
            results.append(sentiment)

        sentiment_df = pd.DataFrame(results)

        return pd.concat(
            [df.reset_index(drop=True), sentiment_df],
            axis=1
        )

    # ---------------------------------------------------

    def aggregate_daily_sentiment(self, df: pd.DataFrame):

        if "score" not in df.columns:
            raise ValueError("DataFrame must contain 'score' column")

        temp_df = df.copy()

        # Robust date handling
        if "published_at" in temp_df.columns:
            temp_df["date"] = pd.to_datetime(
                temp_df["published_at"],
                errors="coerce"
            ).dt.date
        else:
            temp_df["date"] = None

        # fallback to today
        temp_df["date"] = temp_df["date"].fillna(
            pd.Timestamp.today().date()
        )

        aggregated = (
            temp_df
            .groupby("date")
            .agg(
                avg_sentiment=("score", "mean"),
                news_count=("score", "count"),
                sentiment_std=("score", "std")
            )
            .reset_index()
        )

        aggregated["sentiment_std"] = aggregated["sentiment_std"].fillna(0)

        return aggregated
