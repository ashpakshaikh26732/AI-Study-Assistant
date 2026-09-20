"""Embedding model access (shared, cached, GPU-aware)."""
from __future__ import annotations

import functools
import logging

log = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """``auto`` -> ``cuda`` when a GPU is visible, else ``cpu``."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken
        return "cpu"


@functools.lru_cache(maxsize=4)
def _build_embeddings(model_name: str, device: str, batch_size: int):
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("Loading embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        # Normalised vectors make cosine similarity a plain dot product and keep
        # scores comparable between retrieval and quiz grading.
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


def get_embeddings(config: dict):
    """The (process-wide cached) embedding model described by the config."""
    emb = config["rag_core"]["embedding"]
    return _build_embeddings(
        emb["model_name"], resolve_device(emb.get("device", "auto")), int(emb.get("batch_size", 32))
    )


def embed_and_store(documents, config: dict, embeddings=None) -> None:
    """Add chunk Documents to the vector store (kept for backward compatibility).

    Prefer :func:`src.rag_core.indexer.sync_index`, which is incremental.
    """
    from src.rag_core.chunker import chunk_id
    from src.rag_core.vectorstore import get_vector_store

    store = get_vector_store(config, embeddings)
    store.add_documents(documents, ids=[chunk_id(d) for d in documents])
