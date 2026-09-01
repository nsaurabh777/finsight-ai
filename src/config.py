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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Groq exposes an OpenAI-compatible API. CrewAI 1.x has no native "groq"
# provider (and we don't install the heavy litellm fallback), so we reach Groq
# through CrewAI's native OpenAI provider pointed at this base URL.
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Judge for eval/ — defaults to the same provider/model as generation.
JUDGE_LLM_PROVIDER = (os.getenv("JUDGE_LLM_PROVIDER") or LLM_PROVIDER).lower()

# Deterministic outputs everywhere (research + judging).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

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


def _build_llm(provider: str):
    """Return a crewai.LLM configured for `provider`.

    CrewAI 1.x ships native provider clients (no litellm needed for these):
      - Ollama  -> native "ollama" provider, model "ollama/<name>"
      - Groq    -> no native provider, but Groq is OpenAI-compatible, so we use
                   the native OpenAI provider with custom_openai=True pointed at
                   GROQ_BASE_URL.
    """
    from crewai import LLM

    if provider == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is unset. Add it to .env "
                "(free key at https://console.groq.com/keys)."
            )
        return LLM(
            model=GROQ_MODEL,
            custom_openai=True,
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    if provider == "ollama":
        return LLM(
            model=f"ollama/{OLLAMA_MODEL}",
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'groq' or 'ollama')")


def get_llm():
    """LLM used by all four research agents."""
    return _build_llm(LLM_PROVIDER)


def get_judge_llm():
    """LLM used by eval/run_eval.py's LLM-as-judge."""
    return _build_llm(JUDGE_LLM_PROVIDER)
