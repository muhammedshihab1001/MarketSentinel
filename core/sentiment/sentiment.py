import torch
import pandas as pd
import numpy as np
import logging
import time
import os
import threading
import hashlib
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config.env_loader import init_env, get_bool

init_env()

logger = logging.getLogger("marketsentinel.sentiment")


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

    MAX_LOAD_RETRIES = 3

    @classmethod
    def load(cls):

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            os.makedirs(cls.CACHE_DIR, exist_ok=True)

            for attempt in range(cls.MAX_LOAD_RETRIES):

                try:

                    start = time.time()

                    logger.info(
                        "Loading FinBERT on %s",
                        cls._device
                    )

                    torch.set_grad_enabled(False)
                    torch.set_num_threads(
                        min(4, os.cpu_count())
                    )

                    cls._tokenizer = AutoTokenizer.from_pretrained(
                        cls.MODEL_NAME,
                        cache_dir=cls.CACHE_DIR
                    )

                    cls._model = (
                        AutoModelForSequenceClassification
                        .from_pretrained(
                            cls.MODEL_NAME,
                            cache_dir=cls.CACHE_DIR
                        )
                        .to(cls._device)
                    )

                    cls._model.eval()

                    if cls._model.config.num_labels != 3:
                        raise RuntimeError(
                            "FinBERT label mismatch."
                        )

                    logger.info(
                        "FinBERT loaded in %.2fs",
                        time.time() - start
                    )

                    return cls._tokenizer, cls._model, cls._device

                except Exception as e:

                    if attempt == cls.MAX_LOAD_RETRIES - 1:
                        raise RuntimeError(
                            "FinBERT failed to load."
                        ) from e

                    sleep = 2 * (attempt + 1)

                    logger.warning(
                        "FinBERT load retry %s",
                        attempt + 1
                    )

                    time.sleep(sleep)


############################################################
# SENTIMENT ANALYZER
############################################################

class SentimentAnalyzer:

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    BATCH_SIZE = 16
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

    ############################################################

    def __init__(self):

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.tokenizer, self.model, self.device = (
            FinBERTSingleton.load()
        )

        self._warmup()

    ############################################################

    def _warmup(self):
        try:
            self.analyze_batch(["market stable"])
        except Exception:
            logger.exception("FinBERT warmup failed")

    ############################################################
    # CACHE
    ############################################################

    def _cache_path(self, key):

        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"{self.CACHE_DIR}/{h}.parquet"

    def _load_cache(self, path):

        if not os.path.exists(path):
            return None

        modified = pd.Timestamp(
            os.path.getmtime(path),
            unit="s"
        )

        if pd.Timestamp.utcnow() - modified > pd.Timedelta(
            minutes=self.CACHE_TTL_MIN
        ):
            return None

        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _write_cache(self, df, path):

        try:
            tmp = path + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except Exception:
            logger.warning("Sentiment cache write failed.")

    ############################################################

    def _is_garbage(self, text):

        if text is None:
            return True

        if self.TEXT_GARBAGE_REGEX.search(str(text)):
            return True

        if len(str(text).strip()) < 6:
            return True

        return False

    def _clamp_text(self, text):

        return str(text).replace(
            "\n",
            " "
        )[:self.MAX_HEADLINE_CHARS]

    ############################################################
    # BATCH INFERENCE
    ############################################################

    def analyze_batch(self, texts):

        results = []
        failures = 0

        for i in range(0, len(texts), self.BATCH_SIZE):

            raw_batch = texts[i:i+self.BATCH_SIZE]

            batch = [
                self._clamp_text(t)
                for t in raw_batch
                if not self._is_garbage(t)
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
                    ).cpu().numpy()

                idx = 0

                for original in raw_batch:

                    if self._is_garbage(original):
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

            finally:

                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if failures / max(len(texts), 1) > self.MAX_FAILURE_RATE:
            raise RuntimeError(
                "FinBERT failure rate exceeded threshold."
            )

        return results

    ############################################################
    # DATAFRAME
    ############################################################

    def analyze_dataframe(self, df):

        if df.empty:
            return df

        key = hashlib.sha256(
            pd.util.hash_pandas_object(
                df[["headline"]],
                index=False
            ).values.tobytes()
        ).hexdigest()

        cache_path = self._cache_path(key)

        cached = self._load_cache(cache_path)

        if cached is not None:
            logger.info("Sentiment cache hit.")
            return cached

        df = df.copy()

        hash_input = (
            df["headline"].astype(str)
            + df.get("source", "").astype(str)
            + df.get("published_at", "").astype(str)
        )

        df["hash"] = hash_input.map(
            lambda x: hashlib.sha256(
                x.encode()
            ).hexdigest()
        )

        df = df.drop_duplicates("hash")

        headlines = df["headline"].astype(str).tolist()

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        df = df.reset_index(drop=True)

        final = pd.concat(
            [df.drop(columns=["hash"]), sentiment_df],
            axis=1
        )

        self._write_cache(final, cache_path)

        return final

    ############################################################
    # DAILY AGGREGATION
    ############################################################

    def aggregate_daily_sentiment(self, df):

        if df.empty:
            return pd.DataFrame({
                "date": [],
                "avg_sentiment": [],
                "news_count": [],
                "sentiment_std": []
            })

        temp_df = df.copy()

        temp_df["date"] = pd.to_datetime(
            temp_df["published_at"],
            errors="coerce",
            utc=True
        )

        temp_df = temp_df.dropna(subset=["date"])

        temp_df["date"] = temp_df["date"].dt.tz_convert(None)

        embargo_hour = 24 - self.SENTIMENT_EMBARGO_HOURS

        late_mask = temp_df["date"].dt.hour >= embargo_hour

        temp_df["effective_date"] = (
            temp_df["date"].dt.floor("D")
            + pd.to_timedelta(
                late_mask.astype(int),
                unit="D"
            )
        )

        aggregated = (
            temp_df
            .groupby("effective_date", sort=True)
            .agg(
                avg_sentiment=("score", "mean"),
                news_count=("score", "count"),
                sentiment_std=("score", "std")
            )
            .reset_index()
            .rename(columns={"effective_date": "date"})
        )

        low_news = aggregated["news_count"] < self.MIN_NEWS_PER_DAY

        aggregated.loc[
            low_news,
            "avg_sentiment"
        ] *= 0.5

        aggregated["sentiment_std"] = (
            aggregated["sentiment_std"]
            .fillna(self.STD_FLOOR)
            .clip(lower=self.STD_FLOOR)
        )

        aggregated["avg_sentiment"] = (
            aggregated["avg_sentiment"]
            .clip(-1, 1)
        )

        return aggregated
