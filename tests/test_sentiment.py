from app.services.sentiment import SentimentAnalyzer

def test_sentiment_score_range():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text("Apple stock falls after weak guidance")
    assert -1.0 <= result["score"] <= 1.0


def test_sentiment_label():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text("Apple announces record profits")
    assert result["label"] in ["positive", "neutral", "negative"]
