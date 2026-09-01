"""
Builds / refreshes the knowledge vector store from news articles (via
src/tools/market_data.fetch_stock_news_raw, which itself falls back from
yfinance to Google News RSS).

Idempotent re-ingestion (PRD Section 6.3): the Chroma id is
'<ticker>:<article-url>'. Before embedding, we ask the store which of those
ids it already has and skip them — so a daily/cron re-run only embeds new
articles, it doesn't re-embed the whole store.

Run:
    python -m src.rag.ingest_news --tickers KPITTECH.NS,RELIANCE.NS
    python -m src.rag.ingest_news --tickers KPITTECH.NS --limit 15
"""
import argparse

from src.rag.vectorstore import VectorStore
from src.tools.market_data import fetch_stock_news_raw

COLLECTION = "knowledge"


def _clean_meta(article: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool — never None."""
    return {
        "ticker": article.get("ticker") or "",
        "title": article.get("title") or "",
        "publisher": article.get("publisher") or "unknown",
        "published": str(article.get("published") or ""),
        "url": article.get("url") or "",
    }


def ingest(tickers: list[str], limit_per_ticker: int = 10) -> dict[str, int]:
    store = VectorStore(collection_name=COLLECTION)
    added_by_ticker: dict[str, int] = {}

    for ticker in tickers:
        articles = fetch_stock_news_raw(ticker, limit=limit_per_ticker)
        # Build candidate ids; drop articles with no url (no stable id).
        candidates = [
            (f'{ticker}:{a["url"]}', a) for a in articles if a.get("url")
        ]
        if not candidates:
            print(f"{ticker}: no usable articles returned.")
            added_by_ticker[ticker] = 0
            continue

        already = store.existing_ids([cid for cid, _ in candidates])
        fresh = [(cid, a) for cid, a in candidates if cid not in already]

        if fresh:
            store.add(
                ids=[cid for cid, _ in fresh],
                documents=[
                    f'{a["title"]}. {a.get("summary", "")}'.strip() for _, a in fresh
                ],
                metadatas=[_clean_meta(a) for _, a in fresh],
            )
        added_by_ticker[ticker] = len(fresh)
        print(
            f"{ticker}: {len(fresh)} new article(s) embedded, "
            f"{len(candidates) - len(fresh)} already present."
        )

    print(f"Knowledge store now holds {store.count()} passages total.")
    return added_by_ticker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated .NS tickers")
    parser.add_argument("--limit", type=int, default=10, help="Max articles per ticker")
    args = parser.parse_args()
    ingest([t.strip() for t in args.tickers.split(",")], limit_per_ticker=args.limit)
