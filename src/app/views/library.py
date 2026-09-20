"""Library: add notes, keep the index in sync, inspect documents and backends."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from src.app import services, ui
from src.app.views.common import describe_report, flash, run_index_sync, show_flash
from src.llm.model_loader import resolve_provider
from src.preprocessing.document_parser import process_document, tesseract_available
from src.rag_core.embedder import resolve_device
from src.rag_core.indexer import plan_changes

NOTE_TYPES = ["handwritten notes", "lecture slides", "related research papers", "other…"]


def _safe(segment: str) -> str:
    """A folder/file name that can't escape the data directory."""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", segment).strip().strip(".")
    return cleaned or "untitled"


def _save_uploads(files, specialization: str, course: str, notes_type: str) -> tuple[int, list[str]]:
    """Store uploaded files under data/raw and extract text into data/processed."""
    config = services.get_config()
    parts = [p for p in (_safe(specialization) if specialization.strip() else "", _safe(course), _safe(notes_type)) if p]
    raw_dir = Path(config["data"]["raw_path"]).joinpath(*parts)
    processed_dir = Path(config["data"]["processed_path"]).joinpath(*parts)
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved, problems = 0, []
    for upload in files:
        name = _safe(Path(upload.name).name)
        src = raw_dir / name
        src.write_bytes(upload.getvalue())
        try:
            stats = process_document(src, processed_dir / (Path(name).stem + ".txt"), config)
        except Exception as exc:
            problems.append(f"{name}: {exc}")
            continue
        if stats["chars"] == 0:
            hint = "" if tesseract_available() else " (scanned PDF - install Tesseract to OCR it)"
            problems.append(f"{name}: no text could be extracted{hint}")
        else:
            saved += 1
    return saved, problems


def _add_notes_tab() -> None:
    st.write("Upload PDFs, `.txt` or `.md` files. They are stored in your data folder, converted to text, "
             "and added to the search index - only the new files are processed.")
    with st.form("upload_form"):
        c1, c2 = st.columns(2)
        course = c1.text_input("Course / topic name *", placeholder="e.g. Sequence Models")
        specialization = c2.text_input("Specialization (optional)", placeholder="e.g. Deep Learning Specialization")
        kind = st.selectbox("Type of notes", NOTE_TYPES)
        custom = st.text_input("Custom type", placeholder="e.g. exam prep") if kind == NOTE_TYPES[-1] else ""
        files = st.file_uploader("Files", type=["pdf", "txt", "md"], accept_multiple_files=True)
        submitted = st.form_submit_button("Add to library", type="primary", icon=":material/upload:")
    if not submitted:
        return
    notes_type = (custom or "other") if kind == NOTE_TYPES[-1] else kind
    if not course.strip() or not files:
        st.error("Enter a course name and choose at least one file.")
        return
    with st.spinner("Extracting text…"):
        saved, problems = _save_uploads(files, specialization, course, notes_type)
    for problem in problems:
        st.warning(problem)
    if saved:
        report = run_index_sync()
        flash(f"Added {saved} file(s). {describe_report(report)}")
        st.rerun()  # refresh the counters above; the message is shown after the rerun


def _documents_tab() -> None:
    lib = services.library()
    if lib.empty:
        st.info("No documents indexed yet.")
        return
    query = st.text_input("Filter documents", placeholder="Type part of a title or course…", label_visibility="collapsed")
    view = lib
    if query:
        needle = query.lower()
        mask = lib[["title", "course", "notes_type", "specialization"]].apply(
            lambda col: col.astype(str).str.lower().str.contains(needle, regex=False)
        ).any(axis=1)
        view = lib[mask]
    st.dataframe(
        view.sort_values(["course", "notes_type", "title"]),
        hide_index=True,
        width="stretch",
        column_config={
            "source": None,
            "title": "Document",
            "specialization": "Specialization",
            "course": "Course",
            "notes_type": "Type",
            "chunks": st.column_config.NumberColumn("Passages", format="%d"),
            "chars": st.column_config.NumberColumn("Characters", format="%d"),
        },
    )
    st.caption(f"{len(view):,} of {len(lib):,} documents")


def _system_tab() -> None:
    config = services.get_config()
    st.markdown("**Language model backends**")
    resolved = resolve_provider(config, st.session_state.provider)
    for status in services.backend_statuses():
        icon = ":material/check_circle:" if status.available else ":material/circle:"
        st.markdown(f"{icon} **{status.provider}** — {status.detail}")
    st.caption(f"Currently using: **{services.llm_label(st.session_state.provider)}** "
               f"(setting: `{st.session_state.provider}` → `{resolved}`)")

    st.markdown("**Index**")
    emb = config["rag_core"]["embedding"]
    chunking = config["rag_core"]["chunking"]
    st.markdown(
        f"- Embedding model: `{emb['model_name']}` on **{resolve_device(emb.get('device', 'auto'))}**\n"
        f"- Chunking: {chunking['chunk_size']} characters, {chunking['chunk_overlap']} overlap\n"
        f"- Vector store: `{config['rag_core']['database']['persist_directory']}`\n"
        f"- OCR: {'available' if tesseract_available() else 'Tesseract not installed - scanned pages will be skipped'}"
    )
    with st.expander("Rebuild the index from scratch"):
        st.caption("Wipes the vector store and re-embeds every note. You only need this after changing the "
                   "embedding model or chunk size (those changes also trigger it automatically).")
        if st.checkbox("I understand this can take several minutes on a CPU", key="confirm_rebuild"):
            if st.button("Rebuild now", icon=":material/build:"):
                st.success(describe_report(run_index_sync(rebuild=True)))


def render() -> None:
    st.title("Library")
    show_flash()
    config = services.get_config()
    lib = services.library()
    plan = plan_changes(config)

    cols = st.columns(3)
    with cols[0]:
        ui.kpi("Documents indexed", f"{len(lib):,}")
    with cols[1]:
        ui.kpi("Passages", f"{int(lib['chunks'].sum()) if not lib.empty else 0:,}")
    with cols[2]:
        ui.kpi("Waiting to index", f"{plan.pending}", "new, changed or removed files" if plan.pending else "index is up to date")
    st.write("")

    if plan.pending:
        st.warning(
            f"{len(plan.new)} new, {len(plan.changed)} changed and {len(plan.removed)} removed file(s) in "
            f"`{config['data']['processed_path']}` since the last index.", icon=":material/sync_problem:",
        )
        if st.button("Sync index now", type="primary", icon=":material/sync:"):
            flash(describe_report(run_index_sync()))
            st.rerun()

    add_tab, docs_tab, system_tab = st.tabs(
        [":material/upload_file: Add notes", ":material/description: Documents", ":material/settings: System"]
    )
    with add_tab:
        _add_notes_tab()
    with docs_tab:
        _documents_tab()
    with system_tab:
        _system_tab()
