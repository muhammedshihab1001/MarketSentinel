import torch
import pandas as pd
import logging
import time
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger("marketsentinel.sentiment")


class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cpu"

    MODEL_NAME = "ProsusAI/finbert"

    @classmethod
    def load(cls):

        # -----------------------------
        # CI / TEST MODE
        # -----------------------------
        if os.getenv("CI") == "true" or os.getenv("TEST_MODE") == "true":
            logger.info("Running in TEST_MODE — FinBERT disabled.")
            return None, None, "cpu"

        if cls._model is None:

            start = time.time()

            logger.info("Loading FinBERT model")

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.set_grad_enabled(False)

            cls._device = "cpu"

            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_NAME
            )

            cls._model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME
            ).to(cls._device)

            cls._model.eval()

            logger.info(
                "FinBERT loaded in %.2f seconds",
                time.time() - start
            )

        return cls._tokenizer, cls._model, cls._device


class SentimentAnalyzer:

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    BATCH_SIZE = 16

    def __init__(self):

        (
            self.tokenizer,
            self.model,
            self.device
        ) = FinBERTSingleton.load()

        self.test_mode = self.model is None

        if not self.test_mode:
            self._warmup()

    # ---------------------------------------------------

    def _warmup(self):
        try:
            self.analyze_batch(["market is stable"])
        except Exception:
            logger.exception("FinBERT warmup failed")

    # ---------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ---------------------------------------------------

    def analyze_text(self, text: str):

        if not text:
            return {"label": "neutral", "score": 0.0}

        if self.test_mode:
            return self._fake_sentiment(text)

        return self.analyze_batch([text])[0]

    # ---------------------------------------------------

    def _fake_sentiment(self, text: str):
        """
        Deterministic fake sentiment for CI.

        No randomness.
        No ML.
        Always stable.
        """

        text = text.lower()

        if any(word in text for word in ["record", "profit", "growth", "beat"]):
            return {"label": "positive", "score": 0.6}

        if any(word in text for word in ["fall", "loss", "weak", "drop"]):
            return {"label": "negative", "score": -0.6}

        return {"label": "neutral", "score": 0.0}

    # ---------------------------------------------------

    def analyze_batch(self, texts):

        if self.test_mode:
            return [self._fake_sentiment(t) for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=1)

        scores = scores.cpu().numpy()

        results = []

        for s in scores:

            label_id = int(s.argmax())
            label = self.label_map[label_id]

            sentiment_score = float(s[2] - s[0])

            results.append({
                "label": label,
                "score": round(sentiment_score, 4)
            })

        return results

    # ---------------------------------------------------

    def analyze_dataframe(self, df: pd.DataFrame):

        if df.empty:
            return df

        if "headline" not in df.columns:
            raise ValueError("Missing headline column")

        headlines = (
            df["headline"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        return pd.concat(
            [df.reset_index(drop=True), sentiment_df],
            axis=1
        )

    # ---------------------------------------------------

    def aggregate_daily_sentiment(self, df: pd.DataFrame):

        if df.empty:
            return pd.DataFrame({
                "date": [],
                "avg_sentiment": [],
                "news_count": [],
                "sentiment_std": []
            })

        temp_df = df.copy()

        temp_df["date"] = pd.to_datetime(
            temp_df.get("published_at"),
            errors="coerce"
        ).dt.date

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

        aggregated["sentiment_std"] = aggregated[
            "sentiment_std"
        ].fillna(0)

        return aggregated
