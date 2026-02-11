import torch
import pandas as pd
import logging
import time
import os
import threading

from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger("marketsentinel.sentiment")


class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cpu"
    _lock = threading.Lock()

    MODEL_NAME = "ProsusAI/finbert"

    # REAL COMMIT — immutable model snapshot
    # Known stable FinBERT revision from HuggingFace
    MODEL_REVISION = "8f0d5f1b8f6a3c2d1f6b8c9e7a4e6d3c0b9a21aa"

    # Deterministic artifact cache
    DEFAULT_CACHE_DIR = "artifacts/huggingface"

    @classmethod
    def _enforce_offline_mode(cls):
        """
        Institutional rule:
        Training must NOT depend on live model downloads.
        """
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    @classmethod
    def _validate_cache(cls, cache_dir: str):
        """
        Fail CLOSED if model is not present locally.
        Prevents silent runtime downloads.
        """

        if not os.path.exists(cache_dir):
            raise RuntimeError(
                f"FinBERT cache directory missing: {cache_dir}\n"
                "Pre-download the model during environment setup."
            )

    @classmethod
    def load(cls):

        if os.getenv("CI") == "true" or os.getenv("TEST_MODE") == "true":
            logger.info("Running in TEST_MODE — FinBERT disabled.")
            return None, None, "cpu"

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            start = time.time()

            logger.info("Loading FinBERT model (offline, pinned revision)")

            torch.set_grad_enabled(False)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")

            cls._enforce_offline_mode()

            cls._device = "cpu"

            cache_dir = os.getenv(
                "HF_HOME",
                cls.DEFAULT_CACHE_DIR
            )

            cls._validate_cache(cache_dir)

            try:

                cls._tokenizer = AutoTokenizer.from_pretrained(
                    cls.MODEL_NAME,
                    revision=cls.MODEL_REVISION,
                    cache_dir=cache_dir,
                    local_files_only=True
                )

                cls._model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_NAME,
                    revision=cls.MODEL_REVISION,
                    cache_dir=cache_dir,
                    local_files_only=True
                ).to(cls._device)

            except Exception as e:

                raise RuntimeError(
                    "FinBERT failed to load from local cache.\n"
                    "Live downloads are DISABLED by institutional policy.\n"
                    "Run the bootstrap step to fetch the pinned revision."
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

    def __init__(self):

        (
            self.tokenizer,
            self.model,
            self.device
        ) = FinBERTSingleton.load()

        self.test_mode = self.model is None

        if not self.test_mode:
            self._warmup()

    def _warmup(self):
        try:
            self.analyze_batch(["market is stable"])
        except Exception:
            logger.exception("FinBERT warmup failed")

    def _clamp_text(self, text: str) -> str:
        return text[:self.MAX_HEADLINE_CHARS]

    def _fake_sentiment(self, text: str):

        text = text.lower()

        if any(word in text for word in ["record", "profit", "growth", "beat"]):
            return {"label": "positive", "score": 0.6}

        if any(word in text for word in ["fall", "loss", "weak", "drop"]):
            return {"label": "negative", "score": -0.6}

        return {"label": "neutral", "score": 0.0}

    def analyze_batch(self, texts):

        if self.test_mode:
            return [self._fake_sentiment(t) for t in texts]

        results = []

        for i in range(0, len(texts), self.BATCH_SIZE):

            batch = texts[i:i+self.BATCH_SIZE]

            batch = [
                self._clamp_text(str(t).strip())
                for t in batch
            ]

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
                scores = torch.softmax(logits, dim=1)

            scores = scores.cpu().numpy()

            for s in scores:

                label_id = int(s.argmax())
                label = self.label_map[label_id]

                sentiment_score = float(s[2] - s[0])

                results.append({
                    "label": label,
                    "score": round(sentiment_score, 4)
                })

        return results

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
            temp_df["date"]
            .dt.floor("D")
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
