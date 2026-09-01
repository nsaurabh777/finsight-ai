from crewai import Agent

from src.config import get_llm
from src.tools.market_data import get_basic_stock_info
from src.tools.ticker_resolver import resolve_ticker


def build_stock_researcher() -> Agent:
    return Agent(
        role="Stock Researcher",
        goal=(
            "Resolve the exact ticker for the company mentioned in the user's "
            "query using resolve_ticker (never guess a ticker suffix), then "
            "fetch basic stock info about it."
        ),
        backstory=(
            "A junior stock researcher meticulous about getting the ticker "
            "right before any analysis begins — knows that guessing ticker "
            "suffixes wastes time and produces unreliable downstream analysis."
        ),
        tools=[resolve_ticker, get_basic_stock_info],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
