import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List


class SentimentAnalyzer:
    """
    Financial news sentiment analyzer using FinBERT.
    """

    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

        # FinBERT label mapping
        self.label_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

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

        # Convert to numeric score
        sentiment_score = (
            scores[2].item() - scores[0].item()
        )  # positive - negative

        return {
            "label": label,
            "score": round(sentiment_score, 4)
        }

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of news.
        Expects column: 'headline'
        """

        results = []

        for _, row in df.iterrows():
            sentiment = self.analyze_text(row["headline"])
            results.append(sentiment)

        sentiment_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
