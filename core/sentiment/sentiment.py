import torch
import pandas as pd
import logging
import time
import os
import threading
import hashlib

from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger("marketsentinel.sentiment")


class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cpu"
    _lock = threading.Lock()

    MODEL_NAME = "ProsusAI/finbert"
    CACHE_DIR = "artifacts/huggingface"

    @classmethod
    def load(cls):

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            start = time.time()

            logger.info("Loading FinBERT model")

            torch.set_grad_enabled(False)
            torch.set_num_threads(1)

            os.makedirs(cls.CACHE_DIR, exist_ok=True)

            try:

                cls._tokenizer = AutoTokenizer.from_pretrained(
                    cls.MODEL_NAME,
                    cache_dir=cls.CACHE_DIR
                )

                cls._model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_NAME,
                    cache_dir=cls.CACHE_DIR
                ).to("cpu")

            except Exception as e:

                raise RuntimeError(
                    "Failed to load FinBERT. Internet required on first run."
                ) from e

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
    MAX_HEADLINE_CHARS = 512

    MIN_CONFIDENCE = 0.55

    MAX_FAILURE_RATE = 0.40

    def __init__(self):

        self.tokenizer, self.model, self.device = FinBERTSingleton.load()

        self._warmup()

    def _warmup(self):
        try:
            self.analyze_batch(["market is stable"])
        except Exception:
            logger.exception("FinBERT warmup failed")

    def _clamp_text(self, text: str) -> str:
        return text[:self.MAX_HEADLINE_CHARS]

    def analyze_batch(self, texts):

        results = []
        failures = 0

        for i in range(0, len(texts), self.BATCH_SIZE):

            batch = texts[i:i+self.BATCH_SIZE]

            batch = [
                self._clamp_text(str(t).strip())
                for t in batch
            ]

            try:

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)

                probs = probs.cpu().numpy()

                for p in probs:

                    confidence = float(p.max())

                    if confidence < self.MIN_CONFIDENCE:

                        results.append({
                            "label": "neutral",
                            "score": 0.0
                        })

                        continue

                    label_id = int(p.argmax())
                    label = self.label_map[label_id]

                    sentiment_score = float(p[2] - p[0])

                    results.append({
                        "label": label,
                        "score": round(sentiment_score, 4)
                    })

            except Exception:

                failures += len(batch)

                logger.exception("FinBERT inference failure")

                results.extend({
                    "label": "neutral",
                    "score": 0.0
                } for _ in batch)

        if failures / max(len(texts), 1) > self.MAX_FAILURE_RATE:

            raise RuntimeError(
                "FinBERT failure rate exceeded safety threshold."
            )

        return results

    def analyze_dataframe(self, df: pd.DataFrame):

        if df.empty:
            return df

        if "headline" not in df.columns:
            raise ValueError("Missing headline column")

        df = df.copy()

        df["hash"] = df["headline"].apply(
            lambda x: hashlib.sha256(
                str(x).encode()
            ).hexdigest()
        )

        df = df.drop_duplicates("hash")

        headlines = (
            df["headline"]
            .astype(str)
            .str.strip()
            .tolist()
        )

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        df = df.reset_index(drop=True)

        return pd.concat(
            [df.drop(columns=["hash"]), sentiment_df],
            axis=1
        )

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
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        temp_df = temp_df.dropna(subset=["date"])

        temp_df["date"] = (
            temp_df["date"].dt.floor("D")
            + pd.Timedelta(days=1)
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
