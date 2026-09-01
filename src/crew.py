"""
Assembles the agents into the research pipeline and exposes a single
run_research(query) function — this is what api.py, the Streamlit app, and the
eval harness all call, so the pipeline logic exists in exactly one place
(PRD Section 1: one backend, many callers).

Two stages, deliberately separate (PRD Section 10 "fails gracefully"):
  1. resolve_company()  — a one-agent crew that maps the query to an NSE
     ticker, or decides the company is outside coverage.
  2. build_analysis_crew() — the 3-agent analyst/news/writer pipeline, which
     only runs once a ticker is confirmed.

Keeping the analysis agents from ever running on a not-found company is what
stops a small local model from "passing the not-found note through" in theory
but analysing the company from parametric knowledge in practice (e.g. it will
happily write a Tesla report), and stops a downstream agent looping to
max_iter and crashing the whole run on a query that was never answerable.

CLI usage:
    python -m src.crew "Is KPIT Technologies a good long-term stock?"
"""
import re
import sys
from dataclasses import dataclass

from crewai import Crew, Process, Task

from src.agents.financial_analyst import build_financial_analyst
from src.agents.news_analyst import build_news_analyst
from src.agents.report_writer import build_report_writer
from src.agents.stock_researcher import build_stock_researcher
from src.config import CREW_MAX_RPM, DISCLAIMER, LLM_PROVIDER
from src.tools.ticker_resolver import NOT_FOUND_PREFIX, resolve

_NS_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9.&\-]*\.NS)\b")

# Only Groq has a request-rate ceiling worth throttling for. Ollama is local
# and rate-limiting it just adds dead time between already-slow calls.
_CREW_MAX_RPM = CREW_MAX_RPM if LLM_PROVIDER == "groq" else None


def build_resolver_crew() -> Crew:
    researcher = build_stock_researcher()
    resolve_task = Task(
        description=(
            "The user asked: {query}\n\n"
            "Identify the company the question is about and call resolve_ticker "
            "with JUST the company name (e.g. 'KPIT Technologies', not the whole "
            "sentence). Call resolve_ticker exactly once.\n"
            "Your final answer must be EXACTLY the string resolve_ticker "
            "returned — either the '.NS' ticker symbol on its own, or the "
            "'TICKER_NOT_FOUND: ...' string on its own. No other words, no "
            "analysis, no explanation."
        ),
        expected_output=(
            "Exactly one line: either a '.NS' ticker symbol, or a string "
            "starting 'TICKER_NOT_FOUND:'."
        ),
        agent=researcher,
    )
    return Crew(
        agents=[researcher],
        tasks=[resolve_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=_CREW_MAX_RPM,
    )


def build_analysis_crew() -> Crew:
    analyst = build_financial_analyst()
    news_analyst = build_news_analyst()
    writer = build_report_writer()

    analysis_task = Task(
        description=(
            "The resolved ticker is {ticker}. The user's question is: {query}\n\n"
            "Call get_basic_stock_info once for {ticker}, then produce "
            "fundamental, technical and risk analysis weighted toward what the "
            "question actually asks (e.g. a valuation question -> lead with "
            "fundamentals). Use retrieve_knowledge to ground any qualitative "
            "claim (management commentary, strategy, sector dynamics) in a "
            "retrieved passage — do not assert qualitative claims from general "
            "knowledge."
        ),
        expected_output=(
            "A structured analysis with tool-computed numbers and source-cited "
            "qualitative points."
        ),
        agent=analyst,
    )

    news_task = Task(
        description=(
            "For ticker {ticker}, use retrieve_knowledge to pull recent news "
            "relevant to: {query}. Summarise what the sources actually say "
            "(with attribution) and separate that from any inference you draw. "
            "If retrieval returns nothing or KNOWLEDGE_STORE_EMPTY, say the "
            "news evidence is unavailable — do not fill the gap from memory."
        ),
        expected_output=(
            "A source-attributed summary of recent news and its likely "
            "relevance to the question; or a note that news evidence is "
            "unavailable."
        ),
        agent=news_analyst,
    )

    report_task = Task(
        description=(
            "Synthesise the previous two steps into one markdown report that "
            "answers: {query}\n\n"
            "Structure: '## Executive Summary', then sections matching the "
            "question's focus, then '## Investment Considerations' (balanced, "
            "not advice). Every number must come from a tool output and every "
            "qualitative claim from a cited source — if something isn't "
            "supported, say so rather than asserting it.\n"
            f"End the report with this disclaimer verbatim:{DISCLAIMER}"
        ),
        expected_output=(
            "A complete markdown research report answering the user's question, "
            "ending with the verbatim disclaimer."
        ),
        agent=writer,
        context=[analysis_task, news_task],
    )

    return Crew(
        agents=[analyst, news_analyst, writer],
        tasks=[analysis_task, news_task, report_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=_CREW_MAX_RPM,
    )


def resolve_company(query: str) -> str | None:
    """Map the query to an NSE '.NS' ticker, or None if the company is outside
    FinSight's coverage universe.

    Runs the one-agent resolver crew. If that crew itself errors (a small local
    model can loop to max_iter and then return empty), falls back to resolving
    the raw query text directly against the vector store — a not-found query is
    then still handled gracefully instead of crashing the run.
    """
    try:
        out = str(build_resolver_crew().kickoff(inputs={"query": query})).strip()
    except Exception as exc:  # noqa: BLE001 — resolver flakiness must not crash
        print(f"  !! resolver crew failed ({type(exc).__name__}: {exc}); "
              f"falling back to direct vector lookup")
        out = resolve(query)

    if NOT_FOUND_PREFIX in out:
        return None
    match = _NS_TICKER_RE.search(out)
    if match:
        return match.group(1)

    # Resolver returned something unexpected — try the deterministic path once.
    fallback = resolve(query)
    return fallback if fallback.endswith(".NS") else None


def _not_in_coverage_report(query: str) -> str:
    return (
        "## Outside Coverage Universe\n\n"
        f'The company referenced in your question — "{query.strip()}" — could '
        "not be matched to any security in FinSight's NSE coverage universe "
        "with sufficient confidence. No analysis has been produced.\n\n"
        "FinSight only covers a defined set of NSE-listed companies and will "
        "not guess at ambiguous names or analyse companies outside that set."
        f"{DISCLAIMER}"
    )


@dataclass
class ResearchResult:
    """Structured pipeline output. api.py returns this as JSON; run_research()
    unwraps it to the report string for the CLI and eval harness."""

    query: str
    report: str
    ticker: str | None       # resolved '.NS' symbol, or None if out of coverage
    in_coverage: bool


def research(query: str) -> ResearchResult:
    """Run the full pipeline and return the report plus resolution metadata.

    Safety nets (PRD Section 10): a not-in-coverage company returns a fixed
    refusal without ever invoking the analysis agents; the disclaimer is
    enforced here even if the writer forgets it.
    """
    ticker = resolve_company(query)
    if ticker is None:
        return ResearchResult(query, _not_in_coverage_report(query), None, False)

    result = build_analysis_crew().kickoff(inputs={"query": query, "ticker": ticker})
    report = str(result).strip()
    if "not financial advice" not in report.lower():
        report += DISCLAIMER
    return ResearchResult(query, report, ticker, True)


def run_research(query: str) -> str:
    """Report-string-only wrapper over research(), for the CLI and eval."""
    return research(query).report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.crew "your question about a stock"')
        sys.exit(1)
    print(run_research(" ".join(sys.argv[1:])))
