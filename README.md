# FinSight AI — Autonomous Financial Research Assistant

A multi-agent, RAG-grounded stock research system built on 100% free/open-source
infrastructure: CrewAI + a free-tier or local LLM, ChromaDB, yfinance, FastAPI,
Streamlit, self-hosted n8n, and self-hosted OpenClaw.

**Full spec:** see [`PRD.md`](./PRD.md) — read that first, it's the source of
truth for architecture, scope, and build order. Implementation deviations from
the PRD are recorded in [PRD.md § 12](./PRD.md#12-implementation-not--deviations-log).

## Quickstart

```bash
python -m virtualenv .venv          # or: python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # then edit .env — see below

# Build the RAG stores (one-time; re-run when the ticker CSV changes)
python -m src.rag.ingest_tickers
python -m src.rag.ingest_news --tickers KPITTECH.NS,RELIANCE.NS

# Phase 1/2 checkpoint — a CLI report
python -m src.crew "Is KPIT Technologies a good long-term stock?"
```

### Choosing an LLM (`.env`)

| `LLM_PROVIDER` | Needs | When |
|---|---|---|
| `groq` (default) | free `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys) (no card) | no local GPU / low-RAM host |
| `ollama` | an Ollama server at `OLLAMA_BASE_URL` running `OLLAMA_MODEL` | you have the hardware; want fully offline |

The rest of the stack (embeddings, vector store, market data) is local and free
regardless of which LLM you pick.

### Offline smoke test (no LLM needed)

```bash
python -m scripts.smoke_offline
```

Exercises the vector store, ticker resolution (hits + graceful-failure misses),
the yfinance tools, and news ingest/retrieval.

## Why n8n *and* OpenClaw?

They are not redundant. **n8n** owns scheduled automation — a cron job that runs
the pipeline against a watchlist and pushes a daily brief. **OpenClaw** owns
on-demand conversational access — ask a question in chat, get an answer right
now. Same backend (`src/api.py` → `src/crew.py`), two different triggers.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Core CrewAI pipeline + yfinance tools | ✅ implemented |
| 2 | RAG: ticker resolver + knowledge/news store | ✅ implemented |
| 3 | LLM-as-judge eval harness | ✅ implemented (baseline run pending) |
| 4 | FastAPI + Streamlit | 🟡 skeleton |
| 5 | n8n scheduled brief | 🟡 skeleton |
| 6 | OpenClaw on-demand chat | 🟡 skeleton (schema unverified) |
| 7 | Polish, Docker Compose | ⬜ not started |
