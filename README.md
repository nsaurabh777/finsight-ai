# FinSight AI — Autonomous Financial Research Assistant

A multi-agent, RAG-grounded stock research system built on 100% free/open-source
infrastructure: CrewAI + a free-tier or local LLM, ChromaDB, yfinance, FastAPI,
Streamlit, self-hosted n8n, and self-hosted OpenClaw.

**Full spec:** see [`PRD.md`](./PRD.md) — read that first, it's the source of
truth for architecture, scope, and build order. Implementation deviations from
the PRD are recorded in [PRD.md § 12](./PRD.md#12-implementation-not--deviations-log).

## Quickstart

```bash
python -m venv .venv          # or: python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # then edit .env — see below

# Build the RAG stores (one-time; re-run when the ticker CSV changes)
python -m src.rag.ingest_tickers
python -m src.rag.ingest_news --tickers KPITTECH.NS,RELIANCE.NS

# Phase 1/2 checkpoint — a CLI report
python -m src.crew "Is KPIT Technologies a good long-term stock?"
```

### API + web UI (Phase 4)

```bash
# terminal 1 — the backend (all interfaces call this, never the crew directly)
uvicorn src.api:app --port 8000
#   POST /research  {"query": "..."}  -> {report, ticker, in_coverage, elapsed_seconds, ...}
#   GET  /health

# terminal 2 — the demo UI
streamlit run app/streamlit_app.py
```

Point the UI at a non-local API with `FINSIGHT_API_URL` in `.env`.

### Choosing an LLM (`.env`)

| `LLM_PROVIDER` | Needs | When |
|---|---|---|
| `groq` (default) | free `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys) (no card) | no local GPU / low-RAM host |
| `ollama` | an Ollama server at `OLLAMA_BASE_URL` running `OLLAMA_MODEL` (a tool-calling model — `qwen2.5:7b`; start the server with `OLLAMA_CONTEXT_LENGTH=16384` or the crew stalls on context summarisation) | you have the hardware; want fully offline |

The rest of the stack (embeddings, vector store, market data) is local and free
regardless of which LLM you pick.

### Offline smoke test (no LLM needed)

```bash
python -m scripts.smoke_offline
```

Exercises the vector store, ticker resolution (hits + graceful-failure misses),
the yfinance tools, and news ingest/retrieval.

### Evaluation

```bash
python -m eval.run_eval                 # 17 queries through the full crew + LLM judge
python -m eval.run_eval --limit 3       # smoke test
python -m scripts.inspect_eval          # per-query breakdown of the latest run
python -m scripts.inspect_knowledge RELIANCE.NS   # what the news store returns for a ticker
```

Each query runs the full pipeline; a judge LLM scores the report against
[`eval/rubric.md`](./eval/rubric.md). Results land in `eval/results/`
(gitignored). Graceful-failure and ticker-accuracy on out-of-universe queries
are checked deterministically, not left to the judge.

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
| 3 | LLM-as-judge eval harness | ✅ baseline captured (below) |
| 4 | FastAPI + Streamlit | ✅ implemented |
| 5 | n8n scheduled brief | 🟡 skeleton |
| 6 | OpenClaw on-demand chat | 🟡 skeleton (schema unverified) |
| 7 | Polish, Docker Compose | ⬜ not started |

### Eval baseline

17 test queries, full pipeline, `qwen2.5:7b` local via Ollama for both
generation and judging (`eval/run_eval.py`, run 2026-09-02):

| Metric | Score |
|---|---|
| Faithfulness (mean, 1–5) | 4.41 |
| Relevance (mean, 1–5) | 4.88 |
| Ticker accuracy (pass) | 94.1% |
| Graceful failure on out-of-universe queries (pass) | 100% |
| Disclaimer present (pass) | 100% |

The single ticker-accuracy miss was a test-data error (a query asked about a
non-NSE listing under a persona category; the pipeline correctly refused).
Fixed in `eval/test_queries.json` for the next run.
