from core.sentiment.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

text = "Apple shares surge after strong quarterly earnings"

result = analyzer.analyze_text(text)
print(result)
