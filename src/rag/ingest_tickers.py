"""
Builds the ticker-resolution vector store from data/tickers/nse_tickers.csv.

Each row becomes one embedded document: the company name plus its common
aliases, so that "KPIT", "KPIT Tech", and "KPIT Technologies" all land near
the same vector. The .NS ticker is stored as metadata and returned by
src/tools/ticker_resolver.py.

Idempotent: the ticker symbol is the Chroma id, and VectorStore.add() upserts,
so re-running after editing the CSV updates in place instead of duplicating.

Run once (and re-run whenever the CSV changes):
    python -m src.rag.ingest_tickers
"""
import csv

from src.config import TICKER_DATA_CSV
from src.rag.vectorstore import VectorStore

COLLECTION = "tickers"


def load_ticker_rows() -> list[dict]:
    with open(TICKER_DATA_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _document_for(row: dict) -> str:
    """Text that gets embedded. Company name first (highest signal), then
    aliases split on ';'. e.g. 'Larsen & Toubro. Also known as: L&T, LT'."""
    name = row["company_name"].strip()
    aliases = [a.strip() for a in row.get("aliases", "").split(";") if a.strip()]
    aliases = [a for a in aliases if a.lower() != name.lower()]
    if aliases:
        return f"{name}. Also known as: {', '.join(aliases)}"
    return name


def ingest() -> int:
    rows = load_ticker_rows()
    store = VectorStore(collection_name=COLLECTION)
    store.add(
        ids=[row["ticker"].strip() for row in rows],
        documents=[_document_for(row) for row in rows],
        metadatas=[
            {"ticker": row["ticker"].strip(), "company_name": row["company_name"].strip()}
            for row in rows
        ],
    )
    print(f"Ingested {len(rows)} tickers into '{COLLECTION}' (store now holds {store.count()}).")
    return len(rows)


if __name__ == "__main__":
    ingest()
