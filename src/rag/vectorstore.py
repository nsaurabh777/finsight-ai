"""
Shared ChromaDB wrapper used by both the ticker-resolution store and the
knowledge/news store. Keeping one wrapper avoids duplicating persistence /
embedding-model setup in two places.

Design choices (see PRD.md Section 3 note on embeddings):
- Persistence: chromadb.PersistentClient writes to CHROMA_PERSIST_DIR on disk,
  so ingest is a one-time step and query-time has no rebuild cost.
- Embeddings: Chroma's DefaultEmbeddingFunction IS all-MiniLM-L6-v2 (the model
  the PRD specifies) but served via onnxruntime (~50MB) instead of
  sentence-transformers + PyTorch (~1GB). Same vectors, fraction of the RAM.
  The ONNX model file (~80MB) downloads once on first use into Chroma's cache.
- Distance: cosine (set via hnsw:space). MiniLM vectors are ~normalized, so
  cosine similarity = 1 - cosine_distance is a clean [0, 1]-ish confidence
  score, which ticker_resolver.py thresholds on for graceful failure.
"""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_PERSIST_DIR

# One shared embedding function instance — building it loads the ONNX model, so
# we don't want one per collection.
_EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()


class VectorStore:
    """
    Thin wrapper around a single Chroma collection.

    Usage:
        store = VectorStore(collection_name="tickers")
        store.add(ids=[...], documents=[...], metadatas=[...])
        results = store.query("KPIT technologies", top_k=1)
        # -> [{"id", "document", "metadata", "distance", "similarity"}, ...]
    """

    def __init__(self, collection_name: str, persist_dir: Path = CHROMA_PERSIST_DIR):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_EMBEDDING_FN,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """Upsert (not insert) so re-running an ingest script is idempotent —
        the same id overwrites rather than duplicating."""
        if not ids:
            return
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of `ids` already present in the collection.
        Used by ingest_news.py to skip re-embedding articles it already has."""
        if not ids:
            return set()
        got = self._collection.get(ids=ids, include=[])
        return set(got["ids"])

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        res = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where or None,
        )
        # Chroma returns each field as a list-of-lists (one inner list per query
        # text); we only ever send one query text, so unwrap [0].
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {
                "id": _id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": 1.0 - dist,  # cosine distance -> similarity
            }
            for _id, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
