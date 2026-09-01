"""
Central configuration for FinSight AI.

Everything that varies by environment (which LLM, where Chroma persists data,
which model names) lives here so the rest of the codebase doesn't scatter
os.environ calls everywhere.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LLM selection --------------------------------------------------------
# "groq"   -> hosted free tier, needs GROQ_API_KEY (no card: console.groq.com)
# "ollama" -> fully local, no key, needs an Ollama server at OLLAMA_BASE_URL
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq deprecated llama-3.3-70b-versatile / llama-3.1-8b-instant for the free
# and developer tiers on 2026-06-17. openai/gpt-oss-120b is Groq's recommended
# replacement. See https://console.groq.com/docs/deprecations
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# CrewAI 1.x has no native "groq" provider. Its native OpenAI provider ignores
# a custom base_url (crewAI issue #5139, wontfix), so pointing it at Groq's
# OpenAI-compatible endpoint silently falls back to api.openai.com. Instead we
# route through LiteLLM with a "groq/" model prefix, which needs the
# crewai[litellm] extra installed.

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# The crew is tool-call heavy; a weak local model loops or emits malformed
# calls. qwen2.5:7b handles it on a 16GB Mac. Also raise the Ollama server
# context (default 4096 truncates CrewAI's long prompts and causes the loops):
# run `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. See .env.example.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Judge for eval/ — defaults to the same provider/model as generation.
JUDGE_LLM_PROVIDER = (os.getenv("JUDGE_LLM_PROVIDER") or LLM_PROVIDER).lower()

# Deterministic outputs everywhere (research + judging).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Groq's free tier is capped at 8000 tokens/minute; a multi-agent run blows
# through that in bursts and gets HTTP 429s ("try again in Ns"). LiteLLM honours
# the Retry-After and backs off when num_retries > 0, which lets the per-minute
# window refill instead of aborting the whole crew. Bump this (or move to Groq's
# Dev tier) if runs still fail. Also cap response length to spend fewer tokens.
LLM_NUM_RETRIES = int(os.getenv("LLM_NUM_RETRIES", "6"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# CrewAI agents default to max_iter=25 tool-loop iterations before forcing a
# final answer. A small model on the ReAct loop (Groq gpt-oss) sometimes fails
# to stop when a tool returns a sentinel like "TICKER_NOT_FOUND: Zomato" and
# keeps re-calling tools until it hits that wall — 25 slow, rate-limited
# iterations per agent. Cap it: the researcher needs ~2 tool calls, the
# analyst ~4.
AGENT_MAX_ITER = int(os.getenv("AGENT_MAX_ITER", "8"))
# Throttle the whole crew under Groq's 30 requests/minute free-tier ceiling.
# Ignored for Ollama (local, no rate limit) — see src/crew.py.
CREW_MAX_RPM = int(os.getenv("CREW_MAX_RPM", "25"))
# Hard per-agent wall-clock ceiling in seconds. 0 = disabled (the default —
# a legitimate local run on a small model can genuinely take 10+ min). Set it
# (e.g. 900) if an agent hangs so the run aborts and moves on instead of
# stalling forever. A hang usually means Ollama's context is too small and
# CrewAI is looping on conversation summarisation — raise OLLAMA_CONTEXT_LENGTH
# first.
AGENT_MAX_EXECUTION_TIME = int(os.getenv("AGENT_MAX_EXECUTION_TIME", "0"))

# --- Paths --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "data" / "chroma"))
TICKER_DATA_CSV = Path(
    os.getenv("TICKER_DATA_CSV", BASE_DIR / "data" / "tickers" / "nse_tickers.csv")
)

# --- Embedding model (local, free) -------------------------------------
# This is the model name; the actual runtime is Chroma's bundled ONNX build
# (see src/rag/vectorstore.py). Kept here for documentation / parity with PRD.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- API server -------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# Where the Streamlit app (and any other client) reaches the FastAPI service.
API_URL = os.getenv("FINSIGHT_API_URL", f"http://localhost:{API_PORT}")


def active_model_name() -> str:
    """Human-readable 'provider/model' for the currently selected LLM — surfaced
    in the API response and the Streamlit sidebar."""
    if LLM_PROVIDER == "groq":
        return f"groq/{GROQ_MODEL}"
    if LLM_PROVIDER == "ollama":
        return f"ollama/{OLLAMA_MODEL}"
    return LLM_PROVIDER

# --- Delivery channels for automation (optional) ---------------------
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DISCLAIMER = (
    "\n\n---\n*This report is generated for educational and portfolio "
    "demonstration purposes only. It is not financial advice. Do your own "
    "research and consult a licensed advisor before making investment "
    "decisions.*"
)


def _disable_cache_breakpoints() -> None:
    """Stop CrewAI from tagging messages with a ``cache_breakpoint`` flag.

    CrewAI's agent executors mark stable messages for Anthropic-style prompt
    caching, but its LiteLLM path forwards that key verbatim in the request
    body. Groq (and other OpenAI-compatible APIs) reject the unknown property
    with a 400 (crewAI issue #5886). The executors re-import this function on
    every call, so neutralising it here is enough; the only effect is that we
    forgo prompt-cache discounts, which Groq/Ollama don't offer anyway.
    """
    try:
        import crewai.llms.cache as _cache

        _cache.mark_cache_breakpoint = lambda message: message
    except Exception:  # pragma: no cover - future CrewAI may drop this module
        pass


def _build_llm(provider: str):
    """Return a crewai.LLM configured for `provider`.

    Providers:
      - Ollama  -> native "ollama" provider, model "ollama/<name>"
      - Groq    -> via LiteLLM, model "groq/<name>" (needs crewai[litellm])
    """
    from crewai import LLM

    _disable_cache_breakpoints()

    if provider == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is unset. Add it to .env "
                "(free key at https://console.groq.com/keys)."
            )
        return LLM(
            model=f"groq/{GROQ_MODEL}",
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            # Unknown kwargs are folded into additional_params and forwarded to
            # litellm.completion(), which is where num_retries takes effect.
            num_retries=LLM_NUM_RETRIES,
        )

    if provider == "ollama":
        return LLM(
            model=f"ollama/{OLLAMA_MODEL}",
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'groq' or 'ollama')")


def agent_runtime_kwargs() -> dict:
    """Shared Agent(...) runtime knobs, so the agent builders stay consistent."""
    kwargs = {"max_iter": AGENT_MAX_ITER}
    if AGENT_MAX_EXECUTION_TIME > 0:
        kwargs["max_execution_time"] = AGENT_MAX_EXECUTION_TIME
    return kwargs


def get_llm():
    """LLM used by all four research agents."""
    return _build_llm(LLM_PROVIDER)


def get_judge_llm():
    """LLM used by eval/run_eval.py's LLM-as-judge."""
    return _build_llm(JUDGE_LLM_PROVIDER)
