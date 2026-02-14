import torch
import pandas as pd
import numpy as np
import logging
import os
import threading

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.config.env_loader import init_env, get_bool

init_env()

logger = logging.getLogger("marketsentinel.sentiment")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


############################################################
# FINBERT SINGLETON (AUTO DOWNLOAD + OFFLINE SAFE)
############################################################

class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _lock = threading.Lock()
    _failed = False

    MODEL_PATH = os.getenv(
        "FINBERT_PATH",
        "artifacts/nlp/finbert"
    )

    HF_MODEL = "ProsusAI/finbert"

    USE_FP16 = torch.cuda.is_available() and get_bool(
        "ENABLE_FP16",
        True
    )

    ########################################################

    @classmethod
    def _download_once(cls):

        logger.warning("FinBERT not found locally → downloading from HuggingFace.")

        os.makedirs(cls.MODEL_PATH, exist_ok=True)

        try:

            tokenizer = AutoTokenizer.from_pretrained(
                cls.HF_MODEL
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                cls.HF_MODEL
            )

            tokenizer.save_pretrained(cls.MODEL_PATH)
            model.save_pretrained(cls.MODEL_PATH)

            logger.warning("FinBERT downloaded and cached locally.")

        except Exception as e:

            cls._failed = True

            raise RuntimeError(
                "FinBERT auto-download failed. "
                "Check internet or HuggingFace access."
            ) from e

    ########################################################

    @classmethod
    def load(cls):

        if cls._failed:
            raise RuntimeError("FinBERT previously failed to load.")

        if cls._model is not None:
            return cls._tokenizer, cls._model, cls._device

        with cls._lock:

            if cls._model is not None:
                return cls._tokenizer, cls._model, cls._device

            ###################################################
            # AUTO DOWNLOAD
            ###################################################

            if not os.path.exists(cls.MODEL_PATH):
                cls._download_once()

            logger.info("Loading FinBERT → %s", cls.MODEL_PATH)
            logger.info("Device → %s", cls._device)

            torch.set_grad_enabled(False)

            torch.set_num_threads(min(4, os.cpu_count()))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            try:

                tokenizer = AutoTokenizer.from_pretrained(
                    cls.MODEL_PATH,
                    local_files_only=True
                )

                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.MODEL_PATH,
                    local_files_only=True
                )

            except Exception as e:

                cls._failed = True

                raise RuntimeError(
                    "FinBERT failed to load from disk."
                ) from e

            model = model.to(cls._device)

            ###################################################
            # SAFE FP16
            ###################################################

            if cls.USE_FP16:
                try:
                    model = model.half()
                except Exception:
                    logger.warning("FP16 conversion failed — using FP32.")

            model.eval()

            if model.config.num_labels != 3:
                raise RuntimeError("FinBERT label mismatch.")

            cls._tokenizer = tokenizer
            cls._model = model

            logger.info("FinBERT READY.")

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

    RNG = np.random.default_rng(42)

    ############################################################

    def __init__(self):

        try:
            self.tokenizer, self.model, self.device = (
                FinBERTSingleton.load()
            )
            self.available = True
            self._warmup()

        except Exception:

            logger.warning(
                "FinBERT unavailable → neutral fallback enabled."
            )

            self.available = False
            self.tokenizer = None
            self.model = None
            self.device = None

    ############################################################

    def _warmup(self):

        if not self.available:
            return

        try:
            dummy = ["market stable"] * self.BATCH_SIZE
            self.analyze_batch(dummy)
        except Exception:
            logger.warning("FinBERT warmup skipped.")

    ############################################################

    def _neutral_sentiment_frame(self, start, end):

        dates = pd.date_range(start=start, end=end, freq="D")

        n = len(dates)

        return pd.DataFrame({
            "date": dates,
            "avg_sentiment": self.RNG.normal(
                0.0, 0.03, n
            ).clip(-0.08, 0.08).astype("float32"),
            "news_count": self.RNG.integers(0, 4, n).astype("int16"),
            "sentiment_std": self.RNG.uniform(
                0.03, 0.08, n
            ).astype("float32")
        })

    ############################################################

    def analyze_batch(self, texts):

        if not self.available:
            return [
                {"label": "neutral", "score": 0.0}
                for _ in texts
            ]

        results = []

        for i in range(0, len(texts), self.BATCH_SIZE):

            batch = [
                str(t).replace("\n", " ")[:self.MAX_HEADLINE_CHARS]
                for t in texts[i:i+self.BATCH_SIZE]
            ]

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
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for p in probs:

                label_id = int(p.argmax())
                score = float(np.clip(p[2] - p[0], -1, 1))

                results.append({
                    "label": self.label_map[label_id],
                    "score": round(score, 4)
                })

        return results

    ############################################################

    def analyze_dataframe(self, df):

        if df.empty:
            return df

        if not self.available:
            df["label"] = "neutral"
            df["score"] = 0.0
            return df

        headlines = df["headline"].astype(str).tolist()

        results = self.analyze_batch(headlines)

        sentiment_df = pd.DataFrame(results)

        return pd.concat(
            [df.reset_index(drop=True), sentiment_df],
            axis=1
        )
