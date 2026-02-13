import torch
import pandas as pd
import numpy as np
import logging
import time
import os
import threading
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.config.env_loader import init_env

init_env()

logger = logging.getLogger("marketsentinel.sentiment")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


############################################################
# FINBERT SINGLETON
############################################################

class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _lock = threading.Lock()

    MODEL_NAME = "ProsusAI/finbert"
    CACHE_DIR = "artifacts/huggingface"

    USE_FP16 = torch.cuda.is_available()

    @classmethod
    def load(cls):

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            os.makedirs(cls.CACHE_DIR, exist_ok=True)

            logger.info("Loading FinBERT on %s", cls._device)

            torch.set_grad_enabled(False)
            torch.set_num_threads(min(4, os.cpu_count()))

            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_NAME,
                cache_dir=cls.CACHE_DIR
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME,
                cache_dir=cls.CACHE_DIR
            )

            model = model.to(cls._device)

            if cls.USE_FP16:
                model = model.half()

            model.eval()

            if model.config.num_labels != 3:
                raise RuntimeError("FinBERT label mismatch.")

            cls._model = model

            logger.info("FinBERT ready.")

            return cls._tokenizer, cls._model, cls._device


############################################################
# SENTIMENT ANALYZER
############################################################

class SentimentAnalyzer:

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    BATCH_SIZE = 32 if torch.cuda.is_available() else 16
    MAX_HEADLINE_CHARS = 300

    MIN_CONFIDENCE = 0.55
    MAX_FAILURE_RATE = 0.40

    MIN_NEWS_PER_DAY = 3
    STD_FLOOR = 0.05
    SENTIMENT_EMBARGO_HOURS = 2

    ############################################################
    # NEW — NEUTRAL FALLBACK
    ############################################################

    def _neutral_sentiment_frame(self, start, end):

        dates = pd.date_range(start=start, end=end, freq="D")

        df = pd.DataFrame({
            "date": dates,
            "avg_sentiment": np.zeros(len(dates), dtype="float32"),
            "news_count": np.zeros(len(dates), dtype="int16"),
            "sentiment_std": np.full(len(dates), 0.05, dtype="float32")
        })

        logger.warning(
            "Sentiment fallback activated — using neutral signal."
        )

        return df

    ############################################################

    def __init__(self):

        self.tokenizer, self.model, self.device = (
            FinBERTSingleton.load()
        )

        self._warmup()

    ############################################################

    def _warmup(self):

        try:
            dummy = ["market stable"] * self.BATCH_SIZE
            self.analyze_batch(dummy)
        except Exception:
            logger.exception("FinBERT warmup failed")

    ############################################################
    # FIXED — FAULT TOLERANT AGGREGATION
    ############################################################

    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:

        if df is None or df.empty:
            return self._neutral_sentiment_frame(
                pd.Timestamp.utcnow() - pd.Timedelta(days=30),
                pd.Timestamp.utcnow()
            )

        working = df.copy()

        working["published_at"] = pd.to_datetime(
            working["published_at"],
            utc=True,
            errors="coerce"
        )

        working = working.dropna(subset=["published_at"])

        embargo_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(
            hours=self.SENTIMENT_EMBARGO_HOURS
        )

        working = working[
            working["published_at"] <= embargo_cutoff
        ]

        if working.empty:
            return self._neutral_sentiment_frame(
                pd.Timestamp.utcnow() - pd.Timedelta(days=30),
                pd.Timestamp.utcnow()
            )

        working["score"] = pd.to_numeric(
            working["score"],
            errors="coerce"
        )

        working = working.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna(subset=["score"])

        working["score"] = working["score"].clip(-1.0, 1.0)

        working["date"] = working["published_at"].dt.floor("D")

        aggregated = working.groupby("date")["score"].agg(
            avg_sentiment="mean",
            sentiment_std="std",
            news_count="count"
        ).reset_index()

        aggregated["sentiment_std"] = aggregated["sentiment_std"].fillna(0.05)

        if aggregated.empty:
            return self._neutral_sentiment_frame(
                working["date"].min(),
                working["date"].max()
            )

        aggregated.sort_values("date", inplace=True)
        aggregated.reset_index(drop=True, inplace=True)

        return aggregated

    ############################################################

    def analyze_batch(self, texts):

        results = []
        failures = 0

        for i in range(0, len(texts), self.BATCH_SIZE):

            raw_batch = texts[i:i+self.BATCH_SIZE]

            batch = [
                str(t).replace("\n", " ")[:self.MAX_HEADLINE_CHARS]
                for t in raw_batch
                if t and not re.match(r"^\s*$|http[s]?://|^\W+$", str(t))
            ]

            if not batch:
                results.extend(
                    {"label": "neutral", "score": 0.0}
                    for _ in raw_batch
                )
                continue

            try:

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                }

                with torch.inference_mode():

                    logits = self.model(**inputs).logits
                    probs = torch.softmax(
                        logits,
                        dim=1
                    ).float().cpu().numpy()

                idx = 0

                for original in raw_batch:

                    if original is None:
                        results.append(
                            {"label": "neutral", "score": 0.0}
                        )
                        continue

                    p = probs[idx]
                    idx += 1

                    confidence = float(p.max())

                    if confidence < self.MIN_CONFIDENCE:
                        results.append(
                            {"label": "neutral", "score": 0.0}
                        )
                        continue

                    label_id = int(p.argmax())

                    score = float(
                        np.clip(p[2] - p[0], -1.0, 1.0)
                    )

                    results.append({
                        "label": self.label_map[label_id],
                        "score": round(score, 4)
                    })

            except Exception:

                failures += len(raw_batch)

                logger.exception("FinBERT inference failure")

                results.extend(
                    {"label": "neutral", "score": 0.0}
                    for _ in raw_batch
                )

        if failures / max(len(texts), 1) > self.MAX_FAILURE_RATE:
            raise RuntimeError(
                "FinBERT failure rate exceeded threshold."
            )

        return results

    ############################################################

    def analyze_dataframe(self, df):

        if df.empty:
            return df

        df = df.copy()

        headlines = df["headline"].astype(str).tolist()

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        df = df.reset_index(drop=True)

        final = pd.concat(
            [df, sentiment_df],
            axis=1
        )

        return final
