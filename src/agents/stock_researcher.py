from crewai import Agent

from src.config import AGENT_MAX_ITER, get_llm
from src.tools.ticker_resolver import resolve_ticker


def build_stock_researcher() -> Agent:
    return Agent(
        role="Stock Researcher",
        goal=(
            "Map the company mentioned in the user's query to its exact NSE "
            "ticker using resolve_ticker (never guess a ticker suffix), and "
            "return only what resolve_ticker gave back."
        ),
        backstory=(
            "A junior stock researcher meticulous about getting the ticker "
            "right before any analysis begins — knows that guessing ticker "
            "suffixes wastes time and produces unreliable downstream analysis. "
            "Calls resolve_ticker once and reports its result verbatim: a "
            "ticker, or the 'TICKER_NOT_FOUND' string. Never invents a ticker "
            "and never tries a second phrasing."
        ),
        tools=[resolve_ticker],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=AGENT_MAX_ITER,
    )
