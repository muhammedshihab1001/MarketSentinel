from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np


class SentimentEngine:

    def __init__(self, half_life: int = 7):
        self.analyzer = SentimentIntensityAnalyzer()
        self.half_life = half_life

    def score_text(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)["compound"]

    def time_decay(self, days_old: int):
        return np.exp(-days_old / self.half_life)

    def aggregate(self, news_items):
        """
        news_items = [
            {"text": "...", "days_old": 2}
        ]
        """

        weighted_scores = []

        for item in news_items:
            score = self.score_text(item["text"])
            weight = self.time_decay(item["days_old"])

            weighted_scores.append(score * weight)

        if not weighted_scores:
            return 0.0

        return float(np.mean(weighted_scores))
