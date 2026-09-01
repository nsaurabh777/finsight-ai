"""
Assembles the four agents into a sequential CrewAI pipeline and exposes a
single run_research(query) function — this is what api.py, the Streamlit
app, and the eval harness all call, so the pipeline logic exists in exactly
one place (PRD Section 1: one backend, many callers).

CLI usage (Phase 1 checkpoint):
    python -m src.crew "Is KPIT Technologies a good long-term stock?"
"""
import sys

from crewai import Crew, Process, Task

from src.agents.financial_analyst import build_financial_analyst
from src.agents.news_analyst import build_news_analyst
from src.agents.report_writer import build_report_writer
from src.agents.stock_researcher import build_stock_researcher
from src.config import DISCLAIMER


def build_crew() -> Crew:
    researcher = build_stock_researcher()
    analyst = build_financial_analyst()
    news_analyst = build_news_analyst()
    writer = build_report_writer()

    resolve_task = Task(
        description=(
            "The user asked: {query}\n\n"
            "Identify the company being asked about and call resolve_ticker "
            "with that company name. Do NOT guess a ticker or its suffix.\n"
            "- If resolve_ticker returns a '.NS' symbol, call get_basic_stock_info "
            "on it and report the ticker plus the basic info.\n"
            "- If resolve_ticker returns a string starting 'TICKER_NOT_FOUND', "
            "state clearly that the company is outside FinSight's coverage "
            "universe and that no analysis can be produced. Do not invent a ticker."
        ),
        expected_output=(
            "Either: the resolved '.NS' ticker and a basic stock-info summary; "
            "or a clear statement that the company is not in the coverage universe."
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "For the ticker resolved in the previous step, and the user's "
            "question ({query}), produce fundamental, technical and risk "
            "analysis weighted toward what the question actually asks (e.g. a "
            "valuation question -> lead with fundamentals). Use retrieve_knowledge "
            "to ground any qualitative claim (management commentary, strategy, "
            "sector dynamics) in a retrieved passage.\n"
            "If the previous step reported TICKER_NOT_FOUND, do no analysis — "
            "just pass that finding through."
        ),
        expected_output=(
            "A structured analysis with tool-computed numbers and source-cited "
            "qualitative points; or a pass-through of the not-found finding."
        ),
        agent=analyst,
        context=[resolve_task],
    )

    news_task = Task(
        description=(
            "For the resolved ticker, use retrieve_knowledge to pull recent "
            "news relevant to: {query}. Summarise what the sources actually "
            "say (with attribution) and separate that from any inference you "
            "draw. If retrieval returns nothing or KNOWLEDGE_STORE_EMPTY, say "
            "the news evidence is unavailable — do not fill the gap from memory.\n"
            "If the ticker was not found, pass that through."
        ),
        expected_output=(
            "A source-attributed summary of recent news and its likely "
            "relevance to the question; or a note that news evidence is "
            "unavailable; or a pass-through of the not-found finding."
        ),
        agent=news_analyst,
        context=[resolve_task],
    )

    report_task = Task(
        description=(
            "Synthesise the previous three steps into one markdown report that "
            "answers: {query}\n\n"
            "Structure: '## Executive Summary', then sections matching the "
            "question's focus, then '## Investment Considerations' (balanced, "
            "not advice). Every number must come from a tool output and every "
            "qualitative claim from a cited source — if something isn't "
            "supported, say so rather than asserting it.\n"
            "If the company was not in the coverage universe, produce a short "
            "report saying exactly that and stop.\n"
            f"End the report with this disclaimer verbatim:{DISCLAIMER}"
        ),
        expected_output=(
            "A complete markdown research report answering the user's question, "
            "ending with the verbatim disclaimer."
        ),
        agent=writer,
        context=[resolve_task, analysis_task, news_task],
    )

    return Crew(
        agents=[researcher, analyst, news_analyst, writer],
        tasks=[resolve_task, analysis_task, news_task, report_task],
        process=Process.sequential,
        verbose=True,
    )


def run_research(query: str) -> str:
    """Run the full pipeline and return the final markdown report.

    Safety net: the Report Writer is prompted to append the disclaimer, but we
    also enforce it here so PRD Section 10's "every report includes the
    disclaimer" holds even if the model forgets.
    """
    result = build_crew().kickoff(inputs={"query": query})
    report = str(result).strip()
    if "not financial advice" not in report.lower():
        report += DISCLAIMER
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.crew "your question about a stock"')
        sys.exit(1)
    print(run_research(" ".join(sys.argv[1:])))
