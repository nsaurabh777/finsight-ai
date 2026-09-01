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
import re

from src.rag.vectorstore import VectorStore

try:
    from crewai.tools import tool
except ImportError:  # older crewai packaging
    from crewai_tools import tool

# Cosine similarity below which we treat the lookup as a miss. Tuned against
# the 50-ticker seed store: real name/alias matches score ~0.55-0.95, while
# most companies not in the store score under 0.4.
MATCH_THRESHOLD = 0.42

# MiniLM embeddings of a short out-of-vocabulary token latch onto its subwords
# (e.g. "Xyzcorp" -> "Hero MotoCorp" at 0.55, well above MATCH_THRESHOLD — it
# shares the "corp" subword). A pure similarity threshold can't catch that
# without also rejecting legit short queries like "TCS" (0.59). So below this
# similarity we additionally require a lexical anchor: at least one word from
# the query (>=3 chars) must literally appear in the matched name/aliases.
# Above it we trust the vector match outright.
STRONG_MATCH_SIMILARITY = 0.75

# Generic words that are no evidence of a real match if they're the only
# lexical overlap — "a company called Xyzcorp" must not anchor to "Titan
# Company" on the word "company".
_ANCHOR_STOPWORDS = frozenset({
    "company", "companies", "corp", "corporation", "limited", "ltd", "inc",
    "industries", "enterprises", "group", "holdings", "the", "and", "for",
    "stock", "stocks", "share", "shares", "invest", "investing", "buy",
    "sell", "good", "called", "about", "some", "any",
})

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
    if best["similarity"] < STRONG_MATCH_SIMILARITY and not _has_lexical_anchor(
        company_query, best["document"]
    ):
        return f"{NOT_FOUND_PREFIX}: {company_query}"
    return best["metadata"]["ticker"]


def _has_lexical_anchor(query: str, document: str) -> bool:
    """True if any >=3-char alphanumeric token from the query appears verbatim
    in the matched document (company name + aliases). Guards against MiniLM
    matching a nonexistent company to a real one purely on shared subwords."""
    doc = document.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", query.lower())
              if t not in _ANCHOR_STOPWORDS]
    return any(tok in doc for tok in tokens)


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
