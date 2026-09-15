"""
jarvis/memory/vector_store.py
─────────────────────────────
Long-term memory via a local ChromaDB vector store (v0.2).

Uses sentence-transformers embeddings entirely on-device — no paid APIs.
Access the shared instance through ``get_vector_store()``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from jarvis.config import settings
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_COLLECTION_NAME = "long_term_memory"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vector_store: VectorStore | None = None


class VectorStore:
    """Persistent local vector store for cross-session facts."""

    def __init__(self, path: str | None = None) -> None:
        db_path = Path(path or settings.vector_db_path).resolve()
        db_path.mkdir(parents=True, exist_ok=True)

        self._path = db_path
        self._current_session_id: str = "default"

        log.info("vector_store_init", path=str(db_path), model=_EMBEDDING_MODEL)

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL,
        )
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "vector_store_ready",
            collection=_COLLECTION_NAME,
            count=self._collection.count(),
        )

    # ── Session context ────────────────────────────────────────────────────────

    @property
    def current_session_id(self) -> str:
        return self._current_session_id

    def set_session(self, session_id: str) -> None:
        """Associate subsequent remember_fact calls with this chat session."""
        self._current_session_id = session_id
        log.debug("vector_store_session_set", session_id=session_id)

    # ── Write / read ───────────────────────────────────────────────────────────

    def add_fact(self, session_id: str, fact: str) -> str:
        """
        Embed and persist a long-term fact.

        Returns:
            A human-readable success or ERROR string.
        """
        fact = (fact or "").strip()
        if not fact:
            return "ERROR: Cannot remember an empty fact."

        fact_id = str(uuid.uuid4())
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        try:
            self._collection.add(
                ids=[fact_id],
                documents=[fact],
                metadatas=[{
                    "session_id": session_id,
                    "timestamp": timestamp,
                }],
            )
        except Exception as e:
            log.error("vector_store_add_failed", error=str(e), session_id=session_id)
            return f"ERROR: Failed to store fact: {e}"

        log.info(
            "vector_store_fact_added",
            fact_id=fact_id,
            session_id=session_id,
            chars=len(fact),
        )
        return f"Remembered: {fact}"

    def search_facts(self, query: str, limit: int = 3) -> str:
        """
        Similarity-search long-term memory and return a formatted string.

        Returns:
            Formatted matches, a no-results message, or an ERROR string.
        """
        query = (query or "").strip()
        if not query:
            return "ERROR: Memory search query must not be empty."

        try:
            limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            limit = 3

        try:
            count = self._collection.count()
            if count == 0:
                log.info("vector_store_search_empty", query=query)
                return "No long-term memories stored yet."

            n_results = min(limit, count)
            raw = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("vector_store_search_failed", query=query, error=str(e))
            return f"ERROR: Memory search failed: {e}"

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        if not documents:
            log.info("vector_store_search_no_hits", query=query)
            return f"No relevant long-term memories found for: '{query}'."

        lines = [
            f'Long-term memory matches for: "{query}"',
            f"Found {len(documents)} result(s):",
            "",
        ]
        for i, doc in enumerate(documents, start=1):
            meta = metadatas[i - 1] if i - 1 < len(metadatas) else {}
            dist = distances[i - 1] if i - 1 < len(distances) else None
            ts = (meta or {}).get("timestamp", "unknown")
            sid = (meta or {}).get("session_id", "unknown")
            score = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
            lines.append(f"[{i}] {doc}")
            lines.append(f"    (session={sid}, stored={ts}, distance={score})")
            lines.append("")

        log.info(
            "vector_store_search_ok",
            query=query,
            hits=len(documents),
            limit=limit,
        )
        return "\n".join(lines).rstrip()


def get_vector_store() -> VectorStore:
    """Return the process-wide VectorStore singleton (lazy init)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Clear the singleton (tests / re-init). Does not delete on-disk data."""
    global _vector_store
    _vector_store = None
