"""
Single entry point used by Streamlit, n8n (scheduled), and OpenClaw
(on-demand chat). This is the actual architecture point of the project —
one implementation of the pipeline, three callers. See PRD.md Section 1.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT, LLM_PROVIDER, active_model_name
from src.crew import research

app = FastAPI(title="FinSight AI", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)


class ResearchResponse(BaseModel):
    query: str
    report: str
    ticker: str | None = None       # resolved '.NS' symbol, null if out of coverage
    in_coverage: bool
    provider: str
    model: str
    elapsed_seconds: float


@app.post("/research", response_model=ResearchResponse)
def do_research(req: ResearchRequest) -> ResearchResponse:
    started = time.monotonic()
    try:
        result = research(req.query)
    except Exception as exc:  # noqa: BLE001 — surface pipeline failures as 502
        raise HTTPException(status_code=502, detail=f"pipeline error: {exc}") from exc
    return ResearchResponse(
        query=result.query,
        report=result.report,
        ticker=result.ticker,
        in_coverage=result.in_coverage,
        provider=LLM_PROVIDER,
        model=active_model_name(),
        elapsed_seconds=round(time.monotonic() - started, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "provider": LLM_PROVIDER, "model": active_model_name()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
