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

    @classmethod
    def load(cls):

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            start = time.time()

            logger.info("Loading FinBERT on %s", cls._device)

            torch.set_grad_enabled(False)
            torch.set_num_threads(min(4, os.cpu_count()))
            os.makedirs(cls.CACHE_DIR, exist_ok=True)

            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_NAME,
                cache_dir=cls.CACHE_DIR
            )

            cls._model = AutoModelForSequenceClassification.from_pretrained(
                cls.MODEL_NAME,
                cache_dir=cls.CACHE_DIR
            ).to(cls._device)

            cls._model.eval()

            if cls._model.config.num_labels != 3:
                raise RuntimeError("FinBERT label mismatch.")

            logger.info(
                "FinBERT loaded in %.2f seconds",
                time.time() - start
            )

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

    BATCH_SIZE = 16
    MAX_HEADLINE_CHARS = 300

    MIN_CONFIDENCE = 0.55
    MAX_FAILURE_RATE = 0.40

    MIN_NEWS_PER_DAY = 3
    STD_FLOOR = 0.05

    SENTIMENT_EMBARGO_HOURS = 2

    TEXT_GARBAGE_REGEX = re.compile(
        r"^\s*$|http[s]?://|^\W+$"
    )

    def __init__(self):

        self.tokenizer, self.model, self.device = FinBERTSingleton.load()
        self._warmup()

    ############################################################

    def _warmup(self):
        try:
            self.analyze_batch(["market is stable"])
        except Exception:
            logger.exception("FinBERT warmup failed")

    ############################################################

    def _is_garbage(self, text: str):

        if text is None:
            return True

        if self.TEXT_GARBAGE_REGEX.search(text):
            return True

        if len(text.strip()) < 5:
            return True

        return False

    def _clamp_text(self, text: str) -> str:
        return str(text).replace("\n", " ")[:self.MAX_HEADLINE_CHARS]

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

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                idx = 0

                for original in raw_batch:

                    if self._is_garbage(original):
                        results.append({"label": "neutral", "score": 0.0})
                        continue

                    p = probs[idx]
                    idx += 1

                    confidence = float(p.max())

                    if confidence < self.MIN_CONFIDENCE:
                        results.append({"label": "neutral", "score": 0.0})
                        continue

                    label_id = int(p.argmax())
                    label = self.label_map[label_id]

                    score = float(np.clip(p[2] - p[0], -1.0, 1.0))

                    results.append({
                        "label": label,
                        "score": round(score, 4)
                    })

            except Exception:

                failures += len(raw_batch)

                logger.exception("FinBERT inference failure")

                results.extend(
                    {"label": "neutral", "score": 0.0}
                    for _ in raw_batch
                )

            finally:
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if failures / max(len(texts), 1) > self.MAX_FAILURE_RATE:
            raise RuntimeError(
                "FinBERT failure rate exceeded safety threshold."
            )

        return results

    ############################################################
    # DATAFRAME INFERENCE
    ############################################################

    def analyze_dataframe(self, df: pd.DataFrame):

        if df.empty:
            return df

        if "headline" not in df.columns:
            raise ValueError("Missing headline column")

        df = df.copy()

        for col in ["source", "published_at"]:
            if col not in df.columns:
                df[col] = ""

        hash_input = (
            df["headline"].astype(str)
            + "|" +
            df["source"].astype(str)
            + "|" +
            df["published_at"].astype(str)
        )

        df["hash"] = hash_input.map(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )

        df = df.drop_duplicates("hash")

        headlines = df["headline"].astype(str).tolist()

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        df = df.reset_index(drop=True)

        return pd.concat(
            [df.drop(columns=["hash"]), sentiment_df],
            axis=1
        )

    ############################################################
    # DAILY AGGREGATION
    ############################################################

    def aggregate_daily_sentiment(self, df: pd.DataFrame):

        if df.empty:
            return pd.DataFrame({
                "date": [],
                "avg_sentiment": [],
                "news_count": [],
                "sentiment_std": []
            })

        if "score" not in df.columns:
            raise RuntimeError("Sentiment score missing.")

        temp_df = df.copy()

        temp_df["date"] = pd.to_datetime(
            temp_df.get("published_at"),
            errors="coerce",
            utc=True
        )

        temp_df = temp_df.dropna(subset=["date"])

        temp_df["date"] = temp_df["date"].dt.tz_convert(None)

        embargo_hour = 24 - self.SENTIMENT_EMBARGO_HOURS

        late_mask = temp_df["date"].dt.hour >= embargo_hour

        temp_df["effective_date"] = (
            temp_df["date"].dt.floor("D")
            + pd.to_timedelta(late_mask.astype(int), unit="D")
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

        #  shrink weak signal toward neutral
        low_news_mask = aggregated["news_count"] < self.MIN_NEWS_PER_DAY
        aggregated.loc[low_news_mask, "avg_sentiment"] *= 0.5

        aggregated["sentiment_std"] = (
            aggregated["sentiment_std"]
            .fillna(self.STD_FLOOR)
            .clip(lower=self.STD_FLOOR)
        )

        aggregated["avg_sentiment"] = aggregated["avg_sentiment"].clip(-1, 1)

        return aggregated
