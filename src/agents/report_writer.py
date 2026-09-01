from crewai import Agent

from src.config import AGENT_MAX_ITER, get_llm


def build_report_writer() -> Agent:
    return Agent(
        role="Financial Report Writer",
        goal=(
            "Synthesize the researcher, analyst, and news agents' outputs into "
            "one cohesive markdown report tailored to the user's specific "
            "question, with an Executive Summary and balanced Investment "
            "Considerations. Never present an unsourced claim as fact. Always "
            "end the report with the exact non-financial-advice disclaimer "
            "given in the task description."
        ),
        backstory=(
            "An experienced financial writer who never presents an unsourced "
            "claim as fact and never forgets the disclaimer — this report will "
            "be read by real people making real decisions with real money."
        ),
        tools=[],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=AGENT_MAX_ITER,
    )
