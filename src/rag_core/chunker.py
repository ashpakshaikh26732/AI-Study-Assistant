"""Split processed text files into overlapping chunks tagged with metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 30  # drop empty/one-word fragments


def read_document_text(file_path) -> str:
    """Read a text file, tolerating the UTF-8 BOM that some editors add."""
    return Path(file_path).read_text(encoding="utf-8-sig", errors="replace")


def parse_source_metadata(file_path, base_dir) -> dict:
    """Derive metadata from a file's location relative to ``base_dir``.

    Layout convention: ``<Specialization>/<Course>/<NotesType>/<file>``

    * 3+ folder levels -> specialization, course, notes_type
    * 2 levels         -> course, notes_type
    * 1 level          -> course
    * 0 levels         -> course "General"

    ``source`` is the path relative to ``base_dir`` (forward slashes) so it stays
    valid when the project moves between machines (e.g. laptop <-> Colab/Drive).
    """
    path = Path(file_path)
    try:
        rel = path.resolve().relative_to(Path(base_dir).resolve())
    except ValueError:  # file outside base_dir: fall back to its bare name
        rel = Path(path.name)
    dirs = list(rel.parts[:-1])

    meta = {"source": rel.as_posix(), "title": rel.stem}
    if len(dirs) >= 3:
        meta.update(specialization=dirs[0], course=dirs[1], notes_type=dirs[2])
    elif len(dirs) == 2:
        meta.update(course=dirs[0], notes_type=dirs[1])
    elif len(dirs) == 1:
        meta.update(course=dirs[0], notes_type="notes")
    else:
        meta.update(course="General", notes_type="notes")
    return meta


def chunk_text(text: str, metadata: dict, config: dict) -> list[Document]:
    """Split ``text`` into chunk Documents, each carrying ``metadata`` + ``chunk_index``."""
    cfg = config["rag_core"]["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text)]
    pieces = [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]
    return [
        Document(page_content=piece, metadata={**metadata, "chunk_index": i})
        for i, piece in enumerate(pieces)
    ]


def chunk_single_document(file_path, config: dict, base_dir=None) -> list[Document]:
    """Read one processed file and return its chunks.

    Args:
        file_path: Path to a ``.txt`` / ``.md`` file.
        config: Project config (chunk size/overlap, processed path).
        base_dir: Root the metadata path is computed against; defaults to
            ``data.processed_path``.
    """
    base_dir = base_dir or config["data"]["processed_path"]
    text = read_document_text(file_path)
    return chunk_text(text, parse_source_metadata(file_path, base_dir), config)


def chunk_id(doc: Document) -> str:
    """Deterministic id for a chunk: same file + position + text => same id."""
    raw = f"{doc.metadata['source']}\x00{doc.metadata['chunk_index']}\x00{doc.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
