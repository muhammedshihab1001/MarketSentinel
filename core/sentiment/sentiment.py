import torch
import pandas as pd
import logging
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger("marketsentinel.sentiment")


# ---------------------------------------------------
# SINGLETON
# ---------------------------------------------------

class FinBERTSingleton:

    _tokenizer = None
    _model = None
    _device = "cpu"

    MODEL_NAME = "ProsusAI/finbert"

    @classmethod
    def load(cls):

        if cls._model is None:

            start = time.time()

            logger.info("Loading FinBERT...")

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
                f"FinBERT loaded in {round(time.time()-start,2)}s"
            )

        return cls._tokenizer, cls._model, cls._device


# ---------------------------------------------------
# ANALYZER
# ---------------------------------------------------

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

        # warmup
        self._warmup()

    # ---------------------------------------------------

    def _warmup(self):

        try:
            self.analyze_batch(["market is stable"])
        except Exception:
            logger.exception("FinBERT warmup failed.")

    # ---------------------------------------------------

    def analyze_batch(self, texts):

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

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
            .str.lower()
            .tolist()
        )

        all_results = []

        try:

            for i in range(0, len(headlines), self.BATCH_SIZE):

                batch = headlines[i:i+self.BATCH_SIZE]

                results = self.analyze_batch(batch)

                all_results.extend(results)

        except Exception:

            logger.exception(
                "Sentiment inference failed — returning neutral fallback."
            )

            # fail soft
            fallback = [{
                "label": "neutral",
                "score": 0.0
            }] * len(headlines)

            all_results = fallback

        sentiment_df = pd.DataFrame(all_results)

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
