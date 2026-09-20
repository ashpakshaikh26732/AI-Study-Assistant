"""Bits shared by several pages."""
from __future__ import annotations

import streamlit as st

from src.app import services
from src.rag_core.indexer import plan_changes, sync_index


def run_index_sync(rebuild: bool = False):
    """Run the incremental indexer with a live progress bar; returns its report."""
    config = services.get_config()
    bar = st.progress(0.0, text="Preparing…")

    def progress(done: int, total: int, current: str) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"Embedding {done}/{total} passages · {current[-55:]}")

    report = sync_index(config, rebuild=rebuild, progress=progress, store=services.get_store(),
                        embeddings=services.get_embedding_model())
    bar.empty()
    services._library.clear()
    services.backend_statuses.clear()
    return report


@st.cache_resource(show_spinner="Loading the speech model (first time only)…")
def _asr():
    from src.voice.speech_to_text import load_whisper_model

    return load_whisper_model(services.get_config())


def transcribe_cached(audio_bytes: bytes) -> str:
    """Transcribe recorded audio; the Whisper model loads on first use only."""
    from src.voice.speech_to_text import transcribe_audio

    try:
        return transcribe_audio(audio_bytes, _asr())
    except Exception as exc:
        st.error(f"Speech recognition failed: {exc}")
        return ""


def flash(message: str) -> None:
    """Queue a success message to show on the next run (survives ``st.rerun()``)."""
    st.session_state["_flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("_flash", None)
    if message:
        st.success(message, icon=":material/check_circle:")


def describe_report(report) -> str:
    parts = []
    if report.rebuilt:
        parts.append("rebuilt from scratch")
    for label, n in (("added", report.added), ("updated", report.updated), ("removed", report.removed)):
        if n:
            parts.append(f"{n} {label}")
    parts.append(f"{report.unchanged} unchanged")
    return f"{', '.join(parts)} — {report.chunks_added} passages embedded in {report.seconds:.0f}s."


def empty_index_onboarding() -> None:
    """Friendly first-run card shown when nothing is indexed yet."""
    config = services.get_config()
    plan = plan_changes(config)
    st.info("**Your library is empty.** Add some notes and index them to start chatting.", icon=":material/inbox:")
    if plan.new:
        st.write(f"Found **{len(plan.new)}** processed note files in `{config['data']['processed_path']}` "
                 "waiting to be indexed.")
        st.caption("Indexing runs on your CPU and takes a few minutes the first time; "
                   "after that only new or changed notes are processed.")
        if st.button("Index my notes now", type="primary", icon=":material/rocket_launch:"):
            report = run_index_sync()
            st.success(describe_report(report))
            st.rerun()
    else:
        st.write("No processed notes found yet. Two ways to add some:")
        st.markdown(
            "1. Open the **Library** page and upload PDFs (or text files).\n"
            f"2. Or put PDFs in `{config['data']['raw_path']}`, run `python run_preprocessing.py`, "
            "then come back."
        )
        link_to("library", "Go to Library", ":material/library_books:")


def link_to(page_key: str, label: str, icon: str | None = None) -> None:
    """Link to another page registered in ``main.py`` (no-op if not registered)."""
    page = st.session_state.get("_pages", {}).get(page_key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)
