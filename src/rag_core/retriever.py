"""Semantic retrieval with topic / note-type filtering and relevance scores."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.rag_core.vectorstore import build_filter, collection_space, distance_to_similarity, get_vector_store


@dataclass
class RetrievedChunk:
    """A retrieved passage plus how well it matched (cosine similarity, 0..1)."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.source

    @property
    def course(self) -> str:
        return self.metadata.get("course", "")

    @property
    def notes_type(self) -> str:
        return self.metadata.get("notes_type", "")


def snippet(text: str, limit: int = 420) -> str:
    """One-line plain-text preview of a passage (markdown heading marks removed)."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def retrieve(
    store,
    query: str,
    k: int = 5,
    course: Optional[str] = None,
    notes_types: Optional[list] = None,
    min_score: float = 0.0,
    max_per_source: int = 0,
    fetch_factor: int = 4,
) -> list[RetrievedChunk]:
    """Top-``k`` passages for ``query``, best first.

    Args:
        store: Chroma store from :func:`get_vector_store`.
        course / notes_types: Restrict the search (metadata filter applied
            *inside* the DB, so it is fast and exact).
        min_score: Drop passages with cosine similarity below this.
        max_per_source: Cap passages per document (0 = unlimited) so one long
            document can't crowd out everything else.
        fetch_factor: Over-fetch ``k * fetch_factor`` candidates before the
            score/diversity filters.
    """
    query = (query or "").strip()
    if not query:
        return []
    space = collection_space(store)
    hits = store.similarity_search_with_score(
        query, k=max(k, 1) * max(fetch_factor, 1), filter=build_filter(course, notes_types)
    )
    results: list[RetrievedChunk] = []
    per_source: dict = {}
    for doc, distance in sorted(hits, key=lambda h: h[1]):
        score = distance_to_similarity(distance, space)
        if score < min_score:
            continue
        source = doc.metadata.get("source", "unknown")
        if max_per_source and per_source.get(source, 0) >= max_per_source:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        results.append(RetrievedChunk(text=doc.page_content, score=score, metadata=doc.metadata))
        if len(results) >= k:
            break
    return results


def create_retriever(config: dict):
    """A plain LangChain retriever over the store (kept for backward compatibility)."""
    store = get_vector_store(config)
    return store.as_retriever(search_kwargs={"k": config["rag_core"]["retriever"]["k"]})
