# PRD: FinSight AI — Autonomous Financial Research Assistant

**Owner:** Saurabh Nair
**Purpose of this document:** Hand this entire file to Claude Code as project context. It should read this before writing any code. It is both the spec and the source of truth for scope — if something isn't in here, ask before building it.

---

## 1. What this project is

A multi-layer AI system that researches a stock and produces a grounded, cited research report, then makes that capability available two ways: on a schedule (automated daily brief) and on demand (chat-triggered). This is a **portfolio project** for a Senior ML Engineer resume — code quality, architectural clarity, and evaluation rigor matter more than feature count. Depth over breadth.

**One system, four layers, not four disconnected scripts:**

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Interfaces                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Streamlit    │  │  n8n         │  │  OpenClaw          │  │
│  │  (manual /    │  │  (scheduled  │  │  (on-demand chat   │  │
│  │   demo UI)    │  │   cron brief)│  │   via WhatsApp/    │  │
│  │               │  │              │  │   Telegram)         │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬───────────┘  │
│         └──────────────────┴────────────────────┘             │
│                            │                                   │
│                   ┌────────▼────────┐                          │
│                   │  FastAPI (single │  ← ALL interfaces call  │
│                   │  entry point)    │    this one endpoint     │
│                   └────────┬────────┘                          │
└────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│  Layer 3: Reasoning — CrewAI multi-agent pipeline                │
│  Stock Researcher → Financial Analyst → News Analyst → Writer    │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│  Layer 2: Retrieval (RAG) — two separate vector stores           │
│  (a) Ticker Resolver: company name → NSE ticker, deterministic  │
│  (b) Knowledge Store: news/filings passages for grounded claims │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│  Layer 1: Data sources                                           │
│  yfinance (prices, fundamentals, news) + local vector stores    │
└───────────────────────────────────────────────────────────────┘

  Cross-cutting: Layer 0 — Evaluation (LLM-as-judge scoring every
  report for faithfulness/relevance, logged for the eval/ harness)
```

**Why this design, explicitly, so Claude Code doesn't "simplify" it away:**
- One FastAPI endpoint, three callers (Streamlit/n8n/OpenClaw) — not three separate implementations of the same logic. This is the actual engineering point of the project.
- n8n and OpenClaw are NOT redundant: n8n owns *scheduled* automation (cron → daily brief → Slack/Telegram push), OpenClaw owns *on-demand conversational* access (user asks a question, gets an answer). Different triggers, same backend. Document this distinction in the README — a reviewer who knows both tools will ask "why both?" and the answer needs to be right there.

---

## 2. Non-goals (explicitly out of scope for v1)

- **This is not investment advice software.** Every report must include a disclaimer: "For educational/portfolio purposes only. Not financial advice." Do not remove this.
- No real-money trading integration, no brokerage API, no order execution.
- No support for options/derivatives/crypto in v1 — equities only (NSE-listed, since that's the existing yfinance-based tooling).
- No user auth / multi-tenancy. Single-user local/personal deployment.
- No mobile app. Streamlit + chat interfaces only.
- Don't try to build all four layers simultaneously — follow the phased plan in Section 7.

---

## 3. Tech stack — every item must be free/open-source, no paid tier required

| Layer | Tool | License / Cost | Notes |
|---|---|---|---|
| LLM (agent reasoning) | **Ollama**, local model (`llama3.1:8b` or `qwen2.5:7b`) | Free, runs fully local, no API key, no internet needed at inference time | Default choice. |
| LLM (optional, faster) | **Groq API**, free tier (`llama-3.3-70b-versatile`) | Free tier, generous rate limits, requires a free signup for an API key | Use if local hardware is slow; same `crewai` code, just swap the LLM client. |
| Agent orchestration | **CrewAI** | Open source (MIT), free | Already used in the prototype notebooks. |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Open source, runs locally via HuggingFace, free | No OpenAI embeddings — avoid any paid embedding API. |
| Vector store | **ChromaDB** | Open source, local persistence to disk, free | No Pinecone/Weaviate Cloud — self-hosted only. |
| Market data | **yfinance** | Open source Python package, free (unofficial Yahoo Finance wrapper) | Already used in prototype; note reliability caveat — no SLA. |
| News source | **yfinance** `.news` + Google News RSS as backup | Free, no API key | RSS needs no signup. |
| API layer | **FastAPI** + **uvicorn** | Open source, free | Single entry point for all interfaces. |
| Frontend | **Streamlit** | Open source, free; deployable on Streamlit Community Cloud (free tier) | |
| Scheduled automation | **n8n**, self-hosted via Docker | Open source (fair-code), free when self-hosted | Do NOT use n8n Cloud — self-host with `docker-compose`. |
| Conversational interface | **OpenClaw**, self-hosted | Open source, free | Bring-your-own LLM key (Ollama local or Groq free tier) — no OpenClaw subscription needed. |
| Eval / LLM-as-judge | Same Ollama/Groq model, custom harness | Free | No LangSmith/paid eval platform — keep it self-contained. |
| Containerization (stretch) | **Docker Compose** | Free | Bundles n8n + API + app together. |

**No OpenAI API usage anywhere in this project.** If Claude Code defaults to `langchain_openai` or similar out of habit, stop and use the Ollama/Groq equivalents instead.

---

## 4. Repository structure

```
finsight-ai/
├── PRD.md                      # this file
├── README.md                   # short overview + quickstart, links to PRD
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py                # central config: model choice, paths, env vars
│   ├── crew.py                  # assembles agents + tasks into the Crew
│   ├── api.py                   # FastAPI app, single /research endpoint
│   ├── agents/
│   │   ├── stock_researcher.py
│   │   ├── financial_analyst.py
│   │   ├── news_analyst.py
│   │   └── report_writer.py
│   ├── tools/
│   │   ├── market_data.py       # yfinance-backed tools (ported from prototype)
│   │   ├── ticker_resolver.py   # RAG-based ticker lookup (replaces guessing)
│   │   └── knowledge_retriever.py  # RAG tool for qualitative grounding
│   └── rag/
│       ├── vectorstore.py       # ChromaDB wrapper, shared by both stores
│       ├── ingest_tickers.py    # builds the ticker resolution vector store
│       └── ingest_news.py       # builds the knowledge/news vector store
├── eval/
│   ├── test_queries.json        # ~15 fixed test queries covering edge cases
│   ├── rubric.md                # scoring rubric for the LLM judge
│   └── run_eval.py              # runs test_queries through the pipeline + judge
├── app/
│   └── streamlit_app.py         # manual/demo UI, calls the FastAPI endpoint
├── automation/
│   ├── n8n/
│   │   └── daily_brief_workflow.json   # importable n8n workflow
│   └── openclaw/
│       └── finsight_skill/
│           └── README.md         # skill design notes (see Section 6.3 — verify against current OpenClaw docs)
└── data/
    └── tickers/
        └── nse_tickers.csv       # seed reference data for ticker resolution RAG
```

---

## 5. Agent specs (ported and refactored from the existing prototype)

The prototype notebooks already have working agent definitions using `yfinance`-backed tools on CrewAI + Groq/Llama3-70B. Reuse that logic, refactored as follows:

### 5.1 Stock Researcher
- **Change from prototype:** currently guesses tickers by trial and error (`KPIT` → `KPITTECH` → `KPIT-tech` → `KPITTECH.NS`, 4 wasted LLM calls). Replace with the `ticker_resolver` RAG tool (Section 6.1) — one deterministic lookup.
- Goal: resolve the company/ticker from the user's query and fetch basic stock info.
- Tools: `resolve_ticker` (new), `get_basic_stock_info` (existing, ported).

### 5.2 Financial Analyst
- Goal: fundamental + technical + risk analysis, tailored to query focus.
- Tools: `get_fundamental_analysis`, `get_stock_risk_assessment`, `get_technical_analysis` (all existing, ported as-is from the prototype — this code already works), plus the new `retrieve_knowledge` RAG tool (Section 6.2) for qualitative grounding (e.g., "what has management said about margins recently").

### 5.3 News Analyst
- **Change from prototype:** currently dumps raw `yfinance` news list and summarizes inline. Route through the `knowledge_retriever` RAG store instead so claims are retrieval-grounded and citable, not just "the LLM read a list and vibed a sentiment."
- Tools: `retrieve_knowledge` (news-scoped query).

### 5.4 Report Writer
- Goal: synthesize into a markdown report with Executive Summary, sections per query focus, Investment Recommendation, disclaimer.
- **Must append the non-goals disclaimer from Section 2 to every report.**
- No new tools — synthesizes prior agents' outputs.

---

## 6. RAG design (this is the centerpiece — build carefully)

### 6.1 Ticker Resolution Store
- **Problem it solves:** the prototype's LLM-guesses-the-ticker-suffix approach (documented failure in the notebook transcript) — non-deterministic, wastes tokens/latency.
- **Source data:** `data/tickers/nse_tickers.csv` — seed with the ~50 tickers already hardcoded in the old prototype (Section 8 has the list), expand later from a public NSE equity listing CSV if desired (NSE publishes this; verify current URL, don't hardcode a possibly-stale one).
- **Embedding:** company name (and common aliases, e.g. "KPIT" / "KPIT Tech" / "KPIT Technologies") → `all-MiniLM-L6-v2` vector.
- **Retrieval:** top-1 nearest neighbor on company name query → returns the exact `.NS` ticker. Deterministic, single call, no guessing.
- **Tool wrapper:** `src/tools/ticker_resolver.py` exposes this as a CrewAI tool the Stock Researcher agent calls directly.

### 6.2 Knowledge/News Store
- **Problem it solves:** makes qualitative claims ("the news suggests a positive outlook") traceable to actual retrieved passages instead of an LLM summarizing a raw dumped list.
- **Source data (v1, keep simple):** `yfinance` `.news` articles, ingested into Chroma with metadata (ticker, publish date, source, URL).
- **Source data (stretch, v2):** if scope allows, add annual reports / investor presentations for a handful of NSE-listed companies (these are freely downloadable PDFs from company investor-relations pages or NSE/BSE announcement archives) — use the `pdf` extraction approach to chunk and ingest. This is what makes the "RAG" claim resume-defensible instead of just news-summarization.
- **Retrieval:** semantic search scoped by ticker + optional date range, top-k (start with k=5) passages returned with source attribution.
- **Tool wrapper:** `src/tools/knowledge_retriever.py`.

### 6.3 Chunking / re-ingestion
- News store should be refreshed incrementally (new articles added, not re-embedding everything each run) — document the ingestion script's idempotency behavior clearly (e.g. dedupe by article URL/id before embedding).

---

## 7. Evaluation harness (this is your strongest differentiator — don't skip it)

This directly mirrors the LLM-as-judge pattern already used professionally (Jaccard Index improvement work), applied to this project. Prioritize this over polish elsewhere.

- `eval/test_queries.json`: ~15 fixed queries covering: (a) simple ticker lookups, (b) ambiguous company names, (c) fundamentals-focused questions, (d) technicals-focused questions, (e) a query about a company NOT in the ticker store (should fail gracefully, not hallucinate a ticker).
- `eval/rubric.md`: scoring criteria for the judge — e.g., **Faithfulness** (does every claim trace to a retrieved source or tool output, 1–5), **Relevance** (does the report answer what was asked, 1–5), **Ticker accuracy** (correct ticker resolved, pass/fail), **Disclaimer present** (pass/fail).
- `eval/run_eval.py`: runs each test query through the full pipeline, then has the judge LLM score the output against the rubric, logs results to a simple CSV/JSON for tracking over time (this becomes the "before/after" evidence when you improve prompts or retrieval — same pattern as the 0.87→0.93 Jaccard bullet).

---

## 8. Existing code to port (don't rewrite from scratch — this already works)

The tool functions below are already implemented and tested in the prototype notebook — port them into `src/tools/market_data.py` largely as-is, just cleaned up (proper CrewAI tool decorator syntax for whatever version gets installed — **verify current syntax against the installed `crewai` version's docs, the decorator API has changed across versions**):

- `get_basic_stock_info(ticker)` — name, sector, industry, market cap, price, 52wk range, volume
- `get_fundamental_analysis(ticker, period)` — PE, forward PE, PEG, price/book, dividend yield, EPS, growth, margins, FCF, D/E, ROE, ratios
- `get_stock_risk_assessment(ticker, period)` — annualized volatility, beta, VaR, max drawdown, Sharpe, Sortino
- `get_technical_analysis(ticker, period)` — SMA 50/200, RSI, MACD, trend/signal classification
- `get_stock_news(ticker, limit)` — recent news list (this becomes the **source data feed** for the knowledge RAG store, not a tool the agent calls directly anymore)

Seed ticker list for `data/tickers/nse_tickers.csv` (from the older prototype — expand as needed):
RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, HDFC, KOTAKBANK, LT, ITC, BAJFINANCE, SBIN, BHARTIARTL, HINDUNILVR, AXISBANK, ASIANPAINT, MARUTI, WIPRO, HCLTECH, ULTRACEMCO, TITAN, SUNPHARMA, ONGC, ADANIENT, TATASTEEL, JSWSTEEL, BAJAJFINSV, NTPC, POWERGRID, TECHM, GRASIM, DIVISLAB, HINDALCO, BRITANNIA, EICHERMOT, M&M, BPCL, TATAMOTORS, INDUSINDBK, UPL, HEROMOTOCO, CIPLA, DRREDDY, COALINDIA, SHREECEM, SBILIFE, APOLLOHOSP, BAJAJ-AUTO, NESTLEIND, TATACONSUM, HDFCLIFE, JIOFIN, KPITTECH, APARINDS (all `.NS` suffix).

**Do NOT port:** the `human` tool pattern from the older prototype notebook (blocks on stdin, incompatible with API/scheduled/chat triggers). Do NOT port the hardcoded ticker dict embedded in a task prompt string — that's what the RAG ticker resolver replaces.

**Known issue to fix while porting:** the older prototype notebook has a broken task reference (`write_report` task description mentions "the news analyst" but no news analyst agent exists in that crew). This is a symptom of copy-pasting between draft versions — double check all `context=[...]` task dependencies reference agents/tasks that actually exist in the final crew definition.

---

## 9. Phased build plan — build and verify each phase before moving on

**Phase 1 — Core pipeline, no RAG yet (get something running end-to-end fast)**
1. Set up `src/config.py`, `.env.example`, `requirements.txt`
2. Port the 4 agents + 5 tools from Section 8 into `src/agents/` and `src/tools/market_data.py`
3. Assemble in `src/crew.py`, verify `crew.kickoff()` works via a simple script or notebook cell, using Ollama or Groq free tier
4. **Checkpoint: a working CLI script that takes a query string and prints a report.**

**Phase 2 — RAG layers**
1. Build `src/rag/vectorstore.py` (Chroma wrapper)
2. Build ticker resolution store + `ticker_resolver.py` tool, swap into Stock Researcher agent, verify it resolves tickers in one call instead of guessing
3. Build news/knowledge store + `knowledge_retriever.py` tool, swap into News Analyst (and optionally Financial Analyst)
4. **Checkpoint: same CLI script, now grounded and deterministic on ticker resolution.**

**Phase 3 — Evaluation**
1. Write `eval/test_queries.json` and `eval/rubric.md`
2. Write `eval/run_eval.py`, run it against Phase 2's pipeline, save baseline scores
3. **Checkpoint: a baseline eval report you can cite in the resume bullet ("evaluated across N test queries, X% faithfulness pass rate" or similar — use real numbers from the actual run, don't estimate).**

**Phase 4 — API + Frontend**
1. Wrap the crew in `src/api.py` (single FastAPI `/research` POST endpoint, takes a query, returns the report + metadata)
2. Finish `app/streamlit_app.py` (the prototype already has this ~80% commented out — complete it, point it at the FastAPI endpoint rather than calling the crew directly)
3. **Checkpoint: working local demo via `streamlit run`.**

**Phase 5 — Automation (n8n)**
1. Self-host n8n via Docker Compose
2. Build `automation/n8n/daily_brief_workflow.json`: cron trigger → HTTP call to the FastAPI `/research` endpoint for a watchlist of tickers → format → push to Slack/Telegram/email (pick one delivery channel to start)
3. **Checkpoint: a scheduled run actually delivers a message.**

**Phase 6 — Conversational interface (OpenClaw)**
1. Self-host OpenClaw
2. **Before writing the skill: fetch OpenClaw's current AgentSkill documentation (search its GitHub repo/docs site) to confirm the manifest schema — do not assume the format in `automation/openclaw/finsight_skill/README.md` is final, it's a design placeholder, not a verified spec.**
3. Build a minimal AgentSkill that accepts a natural-language query via chat (WhatsApp/Telegram) and calls the FastAPI `/research` endpoint
4. **Checkpoint: ask a question via chat, get the report back.**

**Phase 7 — Polish**
1. README with architecture diagram, setup instructions, example output
2. Clean up `requirements.txt` with pinned versions
3. (Stretch) `docker-compose.yml` bundling API + n8n + app together

Do not skip ahead to Phase 5/6 before Phases 1–3 are solid — the automation/interface layers are only as good as the pipeline they call, and a flaky pipeline wrapped in a chat interface just produces confidently wrong answers faster.

---

## 10. Definition of done (v1)

- [x] A query like "Is KPIT Technologies a good long-term stock?" resolves the correct ticker in one deterministic call (no guessing loop)
- [ ] The resulting report cites at least one retrieved knowledge-store passage, not just raw tool output — *partial: works when the store holds relevant news; the seed store is thin so retrieval often (correctly) reports "evidence unavailable". Better news ingestion is Phase 7 polish.*
- [x] `eval/run_eval.py` runs clean and produces a scored report across all test queries — *17 queries, baseline recorded in README (faithfulness 4.41 / relevance 4.88 / graceful-failure 100%)*
- [x] Streamlit app runs locally and produces a report end-to-end — *`app/streamlit_app.py` → FastAPI `/research`, confirmed 2026-09-02*
- [ ] n8n workflow successfully delivers a scheduled report to at least one channel
- [ ] OpenClaw skill successfully answers an on-demand chat query
- [x] README documents the n8n-vs-OpenClaw division of labor explicitly
- [x] No hardcoded API keys anywhere in committed code (use `.env`, and `.env` is in `.gitignore`)
- [x] Every generated report includes the non-financial-advice disclaimer — *enforced in `src/crew.py`, 100% in eval*

## 11. Notes for Claude Code

- This PRD is the spec. If a decision here seems wrong once you're in the code, flag it and propose an alternative rather than silently deviating.
- Prefer working, tested code at each phase checkpoint over a large untested diff spanning multiple phases.
- All package/library choices must stay within the free/open-source stack in Section 3 — if a task seems to need a paid API, stop and find the open-source equivalent instead, don't ask for a workaround with a paid key.
- Where this document says "verify against current docs" (OpenClaw's skill schema, NSE's listing CSV URL, CrewAI's tool decorator syntax for whatever version installs), actually check — don't guess at a plausible-looking API and move on.

---

## 12. Implementation notes / deviations log

Recorded per Section 11 ("flag it and propose an alternative rather than silently deviating"). None of these change the architecture or the free/open-source constraint.

| # | PRD says | Built as | Why |
|---|---|---|---|
| 1 | LLM default: Ollama `llama3.1:8b`, Groq as fallback | **Groq free tier is the default**; Ollama fully supported via `LLM_PROVIDER=ollama` + `OLLAMA_BASE_URL` (can point at a remote box) | The dev/deploy host has ~960 MB RAM / 2 cores — cannot run an 8B local model. Groq's free tier is the PRD's own sanctioned fallback (§3). `src/config.py` keeps both paths first-class. |
| 2 | Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) | **Same model**, served through ChromaDB's bundled **ONNX** build (`onnxruntime`) instead of PyTorch | `sentence-transformers` + `torch` is ~1 GB installed and OOMs at query time in <1 GB RAM. ONNX MiniLM produces the same vectors at ~50 MB. `torch` removed from `requirements.txt`. |
| 3 | LLM client via `langchain-ollama` / `langchain-groq` | `crewai.LLM`, model string `groq/...` or `ollama/...`; `crewai[litellm]` extra required | No separate langchain packages needed. Ollama routes through CrewAI's native OpenAI-compatible client; **Groq must route via LiteLLM** — CrewAI 1.x has no native Groq provider and its native OpenAI provider ignores a custom `base_url` (issue #5139). Also needs a `cache_breakpoint` monkeypatch (issue #5886). See `src/config.py` and the `crewai-groq-integration` project note. |
| 4 | `get_technical_analysis(ticker, period)` ported "as-is" | Default `period` changed `"1mo"` → `"1y"`; trend label guarded when history < 200 days | Latent prototype bug: a 200-day SMA needs ~200 trading days; with 1 month of data every SMA/trend value is `NaN`/wrong. |
| 5 | Tools return `pd.DataFrame` (as in prototype) | Tools return formatted `"key: value"` text | Current CrewAI feeds tool output to the LLM as text; a stringified DataFrame is noisier and drops column context. |
| 6 | News: yfinance `.news` + Google News RSS backup | Implemented both — `fetch_stock_news_raw` tries yfinance, falls back to RSS via `feedparser` | yfinance `.news` schema is unstable and has no SLA (§3); the fallback keeps ingest working. |
| 7 | Ticker CSV seeded from prototype's ~50 names | Same 50, `aliases` column (`;`-separated). No `HDFC` row (only `HDFCBANK`) | HDFC Ltd merged into HDFC Bank in 2023 and is delisted, so `"HDFC"` correctly resolves to `HDFCBANK.NS`. Eval q3 expects this. |
| 8 | `resolve_ticker` — "top-1 nearest neighbor" | Cosine-similarity floor (`MATCH_THRESHOLD` 0.42); below `STRONG_MATCH_SIMILARITY` (0.75) it also requires a lexical anchor — a query word (≥3 chars, minus stopwords) must appear in the matched name/aliases | Makes DoD "fails gracefully" / rubric criterion 4 actually true. MiniLM matched short OOV tokens on shared subwords (`"Xyzcorp"` → `"Hero MotoCorp"` at 0.55); the lexical anchor blocks that without rejecting legit short queries like `"TCS"` (0.59). |
| 9 | Single 4-agent sequential crew (resolve → analyse → news → write) | **Two crews**: a 1-agent resolver, then — only on a confirmed ticker — a 3-agent analysis crew. Out-of-coverage returns a fixed refusal, no analysis agents run. | A small local model (Groq gpt-oss / Ollama qwen2.5:7b) ignores "pass the not-found note through" and analyses the company from parametric knowledge anyway (writes a full Tesla report), or a downstream agent loops to `max_iter` and crashes the run. Gating in code is the only reliable fix. |
| 10 | Judge scores every rubric criterion | `graceful_failure` and `ticker_accuracy` for out-of-universe queries are scored **deterministically** in `run_eval.py` (does the report contain the coverage-refusal language), not by the judge | The judge returned `null` / faithfulness 5 for reports that analysed the wrong company. The safety criteria must not depend on judge leniency. |
| 11 | `retrieve_knowledge` returns top-k passages | Drops hits below `KNOWLEDGE_MIN_RELEVANCE` (0.30 cosine); if none survive, returns "evidence unavailable" | The thin seed store returns top-k hits at ~0.1 similarity (unrelated articles). Handing those to a small model as "the news" stalls it trying to relate an unrelated article to the question. |
| 12 | `max_iter` left at CrewAI default (25) | `AGENT_MAX_ITER` (8) on every agent; `Crew(max_rpm=…)` Groq-only; optional `AGENT_MAX_EXECUTION_TIME` wall-clock ceiling | 25 tool-loop iterations per agent × rate-limit backoff = runs that appear hung for many minutes. |

**Deferred (skeleton only, unchanged from PRD intent):** Phase 5 n8n, Phase 6 OpenClaw (skill schema still unverified — see `automation/openclaw/finsight_skill/README.md`).

**Phase 4 (done):** `src/api.py` exposes `POST /research` (query → report + `ticker` / `in_coverage` / `provider` / `model` / `elapsed_seconds`) and `GET /health`; `app/streamlit_app.py` is a thin UI that calls the API, never the crew. `src/crew.py::research()` returns the structured `ResearchResult`; `run_research()` is the report-string wrapper kept for the CLI and eval.
