"""
Offline smoke test — exercises everything that does NOT need an LLM:
  - ChromaDB vector store + ONNX embeddings
  - ticker resolution RAG (hits + graceful-failure misses)
  - yfinance market-data tools
  - news fetch (yfinance -> Google News RSS fallback)

Run after `pip install -r requirements.txt`:
    python -m scripts.smoke_offline
"""
from src.rag.ingest_tickers import ingest as ingest_tickers
from src.rag.ingest_news import ingest as ingest_news
from src.tools.ticker_resolver import resolve
from src.tools.knowledge_retriever import retrieve
from src.tools import market_data


def main() -> None:
    print("== 1. Ingest ticker store ==")
    ingest_tickers()

    print("\n== 2. Ticker resolution ==")
    for q in ["KPIT", "KPIT Technologies", "Reliance", "State Bank of India", "Zomato", "Xyzcorp"]:
        print(f"  {q!r:35} -> {resolve(q)}")

    print("\n== 3. Market-data tools (KPITTECH.NS) ==")
    for fn in (
        market_data.get_basic_stock_info,
        market_data.get_fundamental_analysis,
        market_data.get_stock_risk_assessment,
        market_data.get_technical_analysis,
    ):
        print(f"\n--- {fn.name} ---")
        print(fn.run("KPITTECH.NS") if hasattr(fn, "run") else fn("KPITTECH.NS"))

    print("\n== 4. News fetch + ingest (KPITTECH.NS) ==")
    raw = market_data.fetch_stock_news_raw("KPITTECH.NS", limit=5)
    print(f"  fetched {len(raw)} articles; first: {raw[0]['title'] if raw else '(none)'}")
    ingest_news(["KPITTECH.NS"], limit_per_ticker=5)

    print("\n== 5. Knowledge retrieval ==")
    print(retrieve("recent news and outlook", "KPITTECH.NS", top_k=3))

    print("\nOK — offline layers working.")


if __name__ == "__main__":
    main()
