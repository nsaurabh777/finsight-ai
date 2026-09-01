"""
Replaces the prototype's trial-and-error ticker guessing
(KPIT -> KPITTECH -> KPIT-tech -> KPITTECH.NS, 4 wasted LLM calls, see the
prototype notebook transcript) with one deterministic vector lookup.

Graceful failure (PRD Section 10 DoD + rubric criterion 4): if the best match
is below MATCH_THRESHOLD cosine similarity, return TICKER_NOT_FOUND instead of
handing back a low-confidence guess. The agent prompts instruct the crew to
stop and report "not found" rather than hallucinate analysis on a wrong or
nonexistent security.
"""
from src.rag.vectorstore import VectorStore

try:
    from crewai.tools import tool
except ImportError:  # older crewai packaging
    from crewai_tools import tool

# Cosine similarity below which we treat the lookup as a miss. Tuned against
# the 50-ticker seed store: real name/alias matches score ~0.55-0.95, while
# companies not in the store (e.g. "Zomato", "Xyzcorp") score well under 0.4.
MATCH_THRESHOLD = 0.42

NOT_FOUND_PREFIX = "TICKER_NOT_FOUND"

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(collection_name="tickers")
    return _store


def resolve(company_query: str) -> str:
    """Plain function (no CrewAI wrapper) so eval/tests can call it directly.

    Returns either a resolved '.NS' ticker string, or
    'TICKER_NOT_FOUND: <query>' when nothing matches confidently.
    """
    results = _get_store().query(company_query, top_k=1)
    if not results:
        return f"{NOT_FOUND_PREFIX}: {company_query}"

    best = results[0]
    if best["similarity"] < MATCH_THRESHOLD:
        return f"{NOT_FOUND_PREFIX}: {company_query}"
    return best["metadata"]["ticker"]


@tool("resolve_ticker")
def resolve_ticker(company_query: str) -> str:
    """Resolves a company name (or partial name / common abbreviation) to its
    exact NSE ticker symbol with the '.NS' suffix, via semantic search over a
    known-tickers vector store. Use this once, first, instead of guessing
    ticker suffixes.

    Params:
    - company_query: the company as the user referred to it (e.g. "KPIT",
      "KPIT Technologies", "Reliance").

    Returns the best-matching '.NS' ticker (e.g. 'KPITTECH.NS'), OR a string
    starting 'TICKER_NOT_FOUND:' if no known ticker matches confidently. If you
    get TICKER_NOT_FOUND, do NOT invent a ticker — report that the company
    isn't in the coverage universe and stop.
    """
    return resolve(company_query)
