from crewai import Agent

from src.config import AGENT_MAX_ITER, get_llm
from src.tools.knowledge_retriever import retrieve_knowledge
from src.tools.market_data import (
    get_basic_stock_info,
    get_fundamental_analysis,
    get_stock_risk_assessment,
    get_technical_analysis,
)


def build_financial_analyst() -> Agent:
    return Agent(
        role="Financial Analyst",
        goal=(
            "Perform fundamental, technical, and risk analysis relevant to the "
            "user's query, using retrieve_knowledge to ground qualitative "
            "claims (e.g. management commentary on margins) in actual sources "
            "rather than asserting them from general knowledge."
        ),
        backstory=(
            "A seasoned financial analyst who backs every qualitative claim "
            "with a retrieved source and every quantitative claim with a "
            "tool-computed number — never asserts something it can't point to."
        ),
        tools=[
            get_basic_stock_info,
            get_fundamental_analysis,
            get_stock_risk_assessment,
            get_technical_analysis,
            retrieve_knowledge,
        ],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=AGENT_MAX_ITER,
    )
