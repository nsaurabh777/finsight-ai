"""
RAG retrieval tool over the knowledge/news vector store (built by
src/rag/ingest_news.py). Agents call this instead of reading a raw dumped
news list, so qualitative claims in the final report are retrieval-grounded
and citable back to a source URL.
"""
from src.rag.text_utils import clean_passage
from src.rag.vectorstore import VectorStore

try:
    from crewai.tools import tool
except ImportError:
    from crewai_tools import tool

COLLECTION = "knowledge"

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(collection_name=COLLECTION)
    return _store


def retrieve(query: str, ticker: str, top_k: int = 4) -> str:
    """Plain function (no CrewAI wrapper) for eval/tests.

    Scopes retrieval to one ticker and formats each hit with source
    attribution so downstream text can cite it. Passage text is cleaned and
    truncated (clean_passage) so a handful of hits can't balloon a local
    model's context — the source URL is the pointer to the full article.
    """
    store = _get_store()
    if store.count() == 0:
        return (
            "KNOWLEDGE_STORE_EMPTY: no passages ingested yet. Run "
            "`python -m src.rag.ingest_news --tickers <TICKER>` first."
        )
    results = store.query(query, top_k=top_k, where={"ticker": ticker})
    if not results:
        return f"No grounding passages found for ticker={ticker!r}, query={query!r}."

    lines = []
    for r in results:
        meta = r["metadata"]
        passage = clean_passage(r["document"]) or clean_passage(meta.get("title", ""))
        lines.append(
            f'- "{passage}"\n'
            f'  (source: {meta.get("publisher", "unknown")}, '
            f'{meta.get("published", "unknown date")}, '
            f'{meta.get("url", "no url")})'
        )
    return "\n".join(lines)


@tool("retrieve_knowledge")
def retrieve_knowledge(query: str, ticker: str, top_k: int = 4) -> str:
    """Retrieves the most relevant grounding passages (news now; company
    filings / investor presentations later) for a question, scoped to one
    ticker.

    Params:
    - query: what you're trying to find out (e.g. "recent margin commentary",
      "why did the stock move this week").
    - ticker: the resolved '.NS' ticker to scope retrieval to (e.g. 'KPITTECH.NS').
    - top_k: how many passages to retrieve (default 4).

    Returns passages each with (publisher, date, url) attribution. Cite these
    in your output rather than asserting claims from general knowledge. If it
    returns KNOWLEDGE_STORE_EMPTY or "No grounding passages found", say the
    news evidence is unavailable — do not substitute your own recollection.
    """
    return retrieve(query, ticker, top_k)
