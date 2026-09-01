"""
Single entry point used by Streamlit, n8n (scheduled), and OpenClaw
(on-demand chat). This is the actual architecture point of the project —
one implementation of the pipeline, three callers. See PRD.md Section 1.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import API_HOST, API_PORT
from src.crew import run_research

app = FastAPI(title="FinSight AI", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    report: str


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    report = run_research(req.query)
    return ResearchResponse(query=req.query, report=report)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
