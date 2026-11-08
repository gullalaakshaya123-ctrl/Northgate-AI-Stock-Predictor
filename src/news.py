from __future__ import annotations
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_news_sentiment(ticker: str, limit: int = 15):
    """Read public Google News RSS headlines; this is a lightweight sentiment proxy."""
    query = f"{ticker} stock market"
    url = "https://news.google.com/rss/search?q=" + query.replace(" ", "+") + "&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for entry in feed.entries[:limit]:
        title = getattr(entry, "title", "")
        if title:
            score = analyzer.polarity_scores(title)["compound"]
            rows.append({"Headline": title, "Sentiment": score,
                         "Label": "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral"),
                         "Published": getattr(entry, "published", "Not provided")})
    return rows
