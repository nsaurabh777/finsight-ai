from crewai import Agent

from src.config import AGENT_MAX_ITER, get_llm
from src.tools.knowledge_retriever import retrieve_knowledge


def build_news_analyst() -> Agent:
    return Agent(
        role="News Analyst",
        goal=(
            "Assess recent news and sentiment for the resolved ticker using "
            "retrieve_knowledge — every claim about 'the news says X' must "
            "trace to a retrieved passage with a source, not a summary of an "
            "un-cited raw dump."
        ),
        backstory=(
            "A sharp news analyst who distinguishes between what sources "
            "actually said and what they're inferring, and always cites which "
            "is which."
        ),
        tools=[retrieve_knowledge],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=AGENT_MAX_ITER,
    )
