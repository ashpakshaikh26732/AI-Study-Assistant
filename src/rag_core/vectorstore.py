"""Vector-store access plus the read helpers the UI needs (topics, library table)."""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.documents import Document

from src.rag_core.embedder import get_embeddings
from src.rag_core.manifest import load_manifest


def get_vector_store(config: dict, embeddings=None):
    """Open (creating if needed) the persistent Chroma collection.

    The collection uses cosine distance, so ``similarity = 1 - distance``.
    """
    from langchain_chroma import Chroma

    db = config["rag_core"]["database"]
    os.makedirs(db["persist_directory"], exist_ok=True)
    return Chroma(
        collection_name=db["collection_name"],
        embedding_function=embeddings or get_embeddings(config),
        persist_directory=db["persist_directory"],
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def collection_space(store) -> str:
    """Distance metric of a store's collection (``cosine`` for stores we build)."""
    try:
        return store._collection.configuration["hnsw"]["space"]
    except Exception:
        return "l2"  # stores made by the first version of the app used Chroma's default


def distance_to_similarity(distance: float, space: str = "cosine") -> float:
    """Convert a Chroma distance to a cosine similarity (vectors are normalised)."""
    if space == "l2":  # squared L2 on unit vectors: d = 2 - 2*cos
        return 1.0 - distance / 2.0
    return 1.0 - distance


def build_filter(course: Optional[str] = None, notes_types: Optional[list] = None) -> Optional[dict]:
    """Chroma ``where`` filter for a course and/or a set of note types."""
    clauses = []
    if course:
        clauses.append({"course": course})
    if notes_types:
        clauses.append({"notes_type": {"$in": list(notes_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def store_size(store) -> int:
    """Number of chunks in the store."""
    try:
        return store._collection.count()
    except Exception:
        return len(store.get(include=[])["ids"])


def get_topic_chunks(store, course: str, notes_types: Optional[list] = None) -> list[Document]:
    """All chunks of a course in reading order (document, then position)."""
    data = store.get(where=build_filter(course, notes_types), include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or [])
    ]
    docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)))
    return docs


def sample_evenly(items: list, n: int) -> list:
    """Pick ``n`` items spread evenly across ``items`` (all of them if fewer)."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def library_table(config: dict, store=None):
    """One row per indexed document: source, title, course, notes_type, chunks, chars.

    Reads the manifest (instant). If there is no manifest but a store exists
    (built by an older version), falls back to grouping the store's metadata.
    """
    import pandas as pd

    columns = ["source", "title", "specialization", "course", "notes_type", "chunks", "chars"]
    files = load_manifest(config).get("files", {})
    rows = [
        {
            "source": source,
            "title": info.get("title", source),
            "specialization": info.get("specialization", ""),
            "course": info.get("course", ""),
            "notes_type": info.get("notes_type", ""),
            "chunks": info.get("chunks", 0),
            "chars": info.get("chars", 0),
        }
        for source, info in files.items()
    ]
    if not rows and store is not None:
        metas = store.get(include=["metadatas"]).get("metadatas") or []
        grouped: dict = {}
        for meta in metas:
            meta = meta or {}
            row = grouped.setdefault(
                meta.get("source", "unknown"),
                {
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title") or os.path.basename(meta.get("source", "unknown")),
                    "specialization": meta.get("specialization", ""),
                    "course": meta.get("course", ""),
                    "notes_type": meta.get("notes_type", ""),
                    "chunks": 0,
                    "chars": 0,
                },
            )
            row["chunks"] += 1
        rows = list(grouped.values())
    return pd.DataFrame(rows, columns=columns)
