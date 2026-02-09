from core.sentiment.sentiment import SentimentAnalyzer

def test_sentiment_score_range():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text("Apple stock falls after weak guidance")
    assert -1.0 <= result["score"] <= 1.0


def test_sentiment_label():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text("Apple announces record profits")
    assert result["label"] in ["positive", "neutral", "negative"]

def test_aggregation_not_empty():
    from core.sentiment.sentiment import SentimentAnalyzer
    import pandas as pd

    analyzer = SentimentAnalyzer()

    df = pd.DataFrame({
        "published_at": [None, None],
        "score": [0.2, -0.1]
    })

    agg = analyzer.aggregate_daily_sentiment(df)
    assert len(agg) == 1
