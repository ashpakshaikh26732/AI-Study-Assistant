"""Cached, lazily-created backend objects shared by all pages.

Streamlit re-runs the script on every interaction, so anything expensive
(models, DB handles) lives behind ``st.cache_resource`` and is created only
when a page first needs it - the app itself starts instantly.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_config
from src.llm.model_loader import LLMUnavailableError, detect_backends, load_llm, resolve_provider
from src.rag_core.embedder import get_embeddings
from src.rag_core.manifest import manifest_mtime
from src.rag_core.vectorstore import get_vector_store, library_table


@st.cache_resource(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner="Loading the embedding model (first time only)…")
def get_embedding_model():
    return get_embeddings(get_config())


@st.cache_resource(show_spinner=False)
def get_store():
    return get_vector_store(get_config(), get_embedding_model())


@st.cache_resource(show_spinner="Starting the language model…")
def _cached_llm(provider: str):
    return load_llm(get_config(), provider)


def get_llm(provider: str):
    """The chat model for ``provider`` or ``None`` (retrieval-only).

    Raises :class:`LLMUnavailableError` when an explicitly chosen backend isn't
    usable. Failures are deliberately *not* cached, so installing Ollama and
    clicking again just works.
    """
    if resolve_provider(get_config(), provider) == "none":
        return None
    return _cached_llm(provider)


def safe_llm(provider: str):
    """``(llm, error_message)`` - never raises."""
    try:
        return get_llm(provider), None
    except LLMUnavailableError as exc:
        return None, str(exc)
    except Exception as exc:  # backend crashed while loading
        return None, f"Could not start the language model: {exc}"


@st.cache_data(ttl=10, show_spinner=False)
def backend_statuses():
    return detect_backends(get_config())


@st.cache_data(show_spinner=False)
def _library(_mtime: float):
    config = get_config()
    table = library_table(config)  # reads the manifest only - no model needed
    legacy_store = Path(config["rag_core"]["database"]["persist_directory"]) / "chroma.sqlite3"
    if table.empty and legacy_store.exists():  # index built by an older version: inspect the store itself
        table = library_table(config, get_store())
    return table


def library():
    """Library DataFrame; refreshes automatically whenever the index changes."""
    return _library(manifest_mtime(get_config()))


def index_size() -> int:
    """Number of indexed passages (from the manifest, so it doesn't load any model)."""
    lib = library()
    return int(lib["chunks"].sum()) if not lib.empty else 0


def llm_label(provider: str) -> str:
    """Human label for the sidebar/status, e.g. ``ollama · llama3.2:3b``."""
    cfg = get_config()["llm"]
    resolved = resolve_provider(get_config(), provider)
    if resolved == "none":
        return "Retrieval-only"
    model = cfg.get(resolved, {}).get("model", "")
    return f"{resolved} · {model}"
