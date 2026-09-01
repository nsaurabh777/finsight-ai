"""
Manual / demo UI for FinSight AI.

Deliberately thin: it collects a query, POSTs it to the FastAPI /research
endpoint, and renders the returned markdown report. All pipeline logic lives
in src/crew.py behind the API (PRD Section 1 — one backend, many callers), so
this file never imports the crew directly.

Run (API must be up separately):
    uvicorn src.api:app --port 8000
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import requests
import streamlit as st

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
# root, so `import src...` needs a hand.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import API_URL  # noqa: E402

# A full report on a local Ollama model can take several minutes.
REQUEST_TIMEOUT_S = 600

st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="centered")
st.title("📈 FinSight AI")
st.caption(
    "Multi-agent, RAG-grounded research on NSE-listed stocks. "
    "Educational demo — not financial advice."
)

with st.sidebar:
    st.subheader("Backend")
    st.write(f"API: `{API_URL}`")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"up · {health.get('model', 'unknown model')}")
    except requests.RequestException:
        st.error("API unreachable — start it with\n`uvicorn src.api:app --port 8000`")
    st.divider()
    st.caption(
        "Example queries:\n\n"
        "- Is KPIT Technologies a good long-term stock?\n"
        "- What is Infosys' debt-to-equity ratio?\n"
        "- How risky is Adani Enterprises?\n"
        "- Is Zomato a buy?  _(out of coverage — should refuse)_"
    )

query = st.text_input(
    "Your question",
    placeholder="Is KPIT Technologies a good long-term stock?",
)

if st.button("Research", type="primary", disabled=not query.strip()):
    with st.spinner("Running the crew — this can take a few minutes on a local model…"):
        try:
            resp = requests.post(
                f"{API_URL}/research",
                json={"query": query.strip()},
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.Timeout:
            st.error(f"Timed out after {REQUEST_TIMEOUT_S}s. The model may be too "
                     f"slow on this host, or the API is stuck.")
            st.stop()
        except requests.ConnectionError:
            st.error(f"Could not reach the API at {API_URL}. Is it running?")
            st.stop()

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
        st.error(f"API returned {resp.status_code}: {detail}")
        st.stop()

    data = resp.json()

    cols = st.columns(3)
    cols[0].metric("Ticker", data.get("ticker") or "—")
    cols[1].metric("In coverage", "Yes" if data.get("in_coverage") else "No")
    cols[2].metric("Elapsed", f"{data.get('elapsed_seconds', 0):.0f}s")

    st.markdown("---")
    st.markdown(data["report"])
