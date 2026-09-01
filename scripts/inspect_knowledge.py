"""Inspect the knowledge/news vector store for one ticker.

Shows how big each retrieved passage is — a large TOTAL (say >8000 chars) is
what stalls a local model right after a retrieve_knowledge call.

    python -m scripts.inspect_knowledge RELIANCE.NS
    python -m scripts.inspect_knowledge KPITTECH.NS "recent margin commentary"
"""
import sys

from src.rag.text_utils import clean_passage
from src.rag.vectorstore import VectorStore


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    query = sys.argv[2] if len(sys.argv) > 2 else "recent news and events"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    store = VectorStore(collection_name="knowledge")
    print(f"knowledge store holds {store.count()} passages total\n")

    results = store.query(query, top_k=top_k, where={"ticker": ticker})
    if not results:
        print(f"no passages for ticker={ticker!r}")
        return

    raw_total = clean_total = 0
    for i, r in enumerate(results, 1):
        raw = r["document"]
        cleaned = clean_passage(raw)
        raw_total += len(raw)
        clean_total += len(cleaned)
        meta = r["metadata"]
        print(
            f"[{i}] sim={r['similarity']:.3f}  raw={len(raw)} chars  "
            f"cleaned={len(cleaned)} chars"
        )
        print(f"    publisher={meta.get('publisher')!r}  published={meta.get('published')!r}")
        print(f"    raw    : {raw[:160]!r}")
        print(f"    cleaned: {cleaned[:160]!r}\n")

    print(f"TOTAL raw chars     : {raw_total}")
    print(f"TOTAL cleaned chars : {clean_total}")
    if raw_total > 8000:
        print("\n>>> raw total is large — this is what stalls the analyst after "
              "retrieve_knowledge. Rebuild the store (see below).")
        print(">>> python -c \"import chromadb; from src.config import "
              "CHROMA_PERSIST_DIR; chromadb.PersistentClient("
              "path=str(CHROMA_PERSIST_DIR)).delete_collection('knowledge')\"")
        print(">>> python -m src.rag.ingest_news --tickers RELIANCE.NS,KPITTECH.NS")


if __name__ == "__main__":
    main()
