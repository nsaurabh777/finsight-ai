"""Shared text hygiene for the knowledge store.

Google News RSS (the fallback news source) returns each article's `summary`
as an HTML blob — often an `<ol>` of *related* headlines with `<a>`/`<font>`
tags and `&nbsp;` entities, several hundred to a few thousand characters.
Embedding that pollutes the vectors, and handing 5 such passages back to a
local model balloons its context and stalls generation.

Both the ingest path (src/rag/ingest_news.py) and the retrieval path
(src/tools/knowledge_retriever.py) run text through clean_passage() so the
store holds — and agents receive — compact plain text.
"""
import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# One news passage rarely needs more than this to ground a claim; the source
# URL in the metadata is the pointer to the full article.
MAX_PASSAGE_CHARS = 400


def clean_passage(text: str, max_chars: int = MAX_PASSAGE_CHARS) -> str:
    """Strip HTML tags/entities, collapse whitespace, truncate on a word
    boundary."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
