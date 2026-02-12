import torch
import pandas as pd
import numpy as np
import logging
import time
import os
import threading
import hashlib
import re
import tempfile

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

            start = time.time()

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

            logger.info(
                "FinBERT loaded in %.2fs",
                time.time() - start
            )

            return cls._tokenizer, cls._model, cls._device


############################################################
# SENTIMENT ANALYZER
############################################################

class SentimentAnalyzer:

    CACHE_SCHEMA_VERSION = "v3"

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

    CACHE_DIR = "data/sentiment_cache"
    CACHE_TTL_MIN = 60

    TEXT_GARBAGE_REGEX = re.compile(
        r"^\s*$|http[s]?://|^\W+$"
    )

    def __init__(self):

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.tokenizer, self.model, self.device = (
            FinBERTSingleton.load()
        )

        self._warmup()

    ############################################################
    # GPU WARMUP
    ############################################################

    def _warmup(self):

        try:
            dummy = ["market stable"] * self.BATCH_SIZE
            self.analyze_batch(dummy)
        except Exception:
            logger.exception("FinBERT warmup failed")

    ############################################################
    # DAILY AGGREGATION (CRITICAL FIX)
    ############################################################

    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "avg_sentiment", "news_count", "sentiment_std"]
            )

        required_cols = {"published_at", "score"}

        if not required_cols.issubset(df.columns):
            raise RuntimeError(
                f"Sentiment dataframe missing columns: {required_cols}"
            )

        working = df.copy()

        ##################################################
        # TIMESTAMP SAFETY
        ##################################################

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
            return pd.DataFrame(
                columns=["date", "avg_sentiment", "news_count", "sentiment_std"]
            )

        ##################################################
        # NUMERIC SAFETY
        ##################################################

        working["score"] = pd.to_numeric(
            working["score"],
            errors="coerce"
        )

        working = working.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna(subset=["score"])

        working["score"] = working["score"].clip(-1.0, 1.0)

        ##################################################
        # DAILY BUCKET
        ##################################################

        working["date"] = working["published_at"].dt.floor("D")

        grouped = working.groupby("date")

        aggregated = grouped["score"].agg(
            avg_sentiment="mean",
            sentiment_std="std",
            news_count="count"
        ).reset_index()

        ##################################################
        # QUALITY FILTERS
        ##################################################

        aggregated = aggregated[
            aggregated["news_count"] >= self.MIN_NEWS_PER_DAY
        ]

        if aggregated.empty:
            return pd.DataFrame(
                columns=["date", "avg_sentiment", "news_count", "sentiment_std"]
            )

        aggregated["sentiment_std"] = aggregated["sentiment_std"].fillna(0)

        aggregated = aggregated[
            aggregated["sentiment_std"] >= self.STD_FLOOR
        ]

        ##################################################
        # FINAL NUMERIC GUARD
        ##################################################

        if not np.isfinite(
            aggregated[["avg_sentiment", "sentiment_std"]].to_numpy()
        ).all():
            raise RuntimeError("Non-finite sentiment detected.")

        aggregated.sort_values("date", inplace=True)
        aggregated.reset_index(drop=True, inplace=True)

        return aggregated

    ############################################################
    # EXISTING METHODS (UNCHANGED)
    ############################################################

    def analyze_batch(self, texts):

        results = []
        failures = 0

        for i in range(0, len(texts), self.BATCH_SIZE):

            raw_batch = texts[i:i+self.BATCH_SIZE]

            batch = [
                str(t).replace("\n", " ")[:self.MAX_HEADLINE_CHARS]
                for t in raw_batch
                if t and not self.TEXT_GARBAGE_REGEX.search(str(t))
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

                    if original is None or self.TEXT_GARBAGE_REGEX.search(str(original)):
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
                        np.clip(
                            p[2] - p[0],
                            -1.0,
                            1.0
                        )
                    )

                    results.append({
                        "label": self.label_map[label_id],
                        "score": round(score, 4)
                    })

            except Exception:

                failures += len(raw_batch)

                logger.exception(
                    "FinBERT inference failure"
                )

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
