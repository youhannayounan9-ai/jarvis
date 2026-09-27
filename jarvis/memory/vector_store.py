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
# v0.20: documents live in a DEDICATED collection so document knowledge and
# personal facts never mix into one retrieval space. Same embedding model
# (all-MiniLM-L6-v2, on-device) and cosine space as personal memory.
KNOWLEDGE_COLLECTION_NAME = "knowledge_base"
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
        # v0.20: knowledge collection shares the client + embedding fn.
        self._knowledge_collection = self._client.get_or_create_collection(
            name=KNOWLEDGE_COLLECTION_NAME,
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


    # ── v0.20: knowledge base (separate collection; see KNOWLEDGE_COLLECTION_NAME) ──

    def add_knowledge_chunks(
        self,
        chunks: list[str],
        ids: list[str],
        metadatas: list[dict],
    ) -> int:
        """Embed + persist document chunks in the knowledge collection."""
        if not chunks:
            return 0
        self._knowledge_collection.add(
            ids=ids, documents=chunks, metadatas=metadatas
        )
        log.info("knowledge_chunks_added", count=len(chunks))
        return len(chunks)

    def delete_knowledge_chunks(self, where: dict) -> int:
        """Delete knowledge chunks matching a metadata filter; returns count."""
        got = self._knowledge_collection.get(where=where, include=[])
        ids = got.get("ids") or []
        if ids:
            self._knowledge_collection.delete(ids=ids)
            log.info("knowledge_chunks_deleted", count=len(ids), where=where)
        return len(ids)

    def knowledge_chunk_ids(self, where: dict) -> list[str]:
        """Chunk ids matching a metadata filter (bounded by Chroma get)."""
        got = self._knowledge_collection.get(where=where, include=[])
        return list(got.get("ids") or [])

    def knowledge_count(self, where: dict | None = None) -> int:
        """Number of knowledge chunks (optionally per document filter)."""
        if where is None:
            return int(self._knowledge_collection.count())
        return len(self.knowledge_chunk_ids(where))

    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
        max_distance: float | None = None,
    ) -> list[dict]:
        """Bounded semantic search over the knowledge collection.

        Returns a list of {text, metadata, distance, id} dicts, most
        relevant first (Chroma returns distance-ascending for cosine).
        Deterministic for identical queries + identical collections.
        """
        query = (query or "").strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 20))
        count = self._knowledge_collection.count()
        if count == 0:
            return []
        raw = self._knowledge_collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        results: list[dict] = []
        for i, doc in enumerate(documents):
            dist = distances[i] if i < len(distances) else None
            if (
                max_distance is not None
                and isinstance(dist, (int, float))
                and dist > max_distance
            ):
                continue
            results.append({
                "id": ids[i] if i < len(ids) else "",
                "text": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": dist,
            })
        log.info(
            "knowledge_search",
            query_chars=len(query),
            hits=len(results),
            top_k=top_k,
            filtered=where is not None,
        )
        return results


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
