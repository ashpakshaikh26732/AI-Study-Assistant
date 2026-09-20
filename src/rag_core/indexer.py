"""Incremental, resumable indexing of processed notes into the vector store.

Only files that are new or changed (by content hash) are re-embedded, files
that disappeared are removed, and progress is saved after every file - so an
interrupted run (or a rebuild after adding one note) costs almost nothing.
Embedding is the slow step on CPU (~15 chunks/s), which is why this matters.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.rag_core.chunker import chunk_id, chunk_text, parse_source_metadata, read_document_text
from src.rag_core.manifest import empty_manifest, index_settings, load_manifest, save_manifest
from src.rag_core.vectorstore import get_vector_store, store_size

log = logging.getLogger(__name__)

INDEXABLE_SUFFIXES = {".txt", ".md"}
ADD_BATCH = 128  # chunks handed to the embedder/DB per call


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_added: int = 0
    seconds: float = 0.0
    rebuilt: bool = False
    failed: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.rebuilt)


def discover_files(processed_dir) -> dict[str, Path]:
    """Map ``relative/posix/path`` -> absolute Path for every indexable file."""
    root = Path(processed_dir)
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in INDEXABLE_SUFFIXES
    }


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SyncPlan:
    """What a sync would do, computed without embedding anything."""

    new: list = field(default_factory=list)
    changed: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    unchanged: int = 0
    hashes: dict = field(default_factory=dict)  # rel path -> sha1

    @property
    def pending(self) -> int:
        return len(self.new) + len(self.changed) + len(self.removed)


def plan_changes(config: dict, manifest: Optional[dict] = None, files: Optional[dict] = None) -> SyncPlan:
    """Compare ``data.processed_path`` with the manifest (cheap: just hashes files)."""
    files = discover_files(config["data"]["processed_path"]) if files is None else files
    known = (manifest if manifest is not None else load_manifest(config))["files"]
    plan = SyncPlan(removed=[rel for rel in known if rel not in files])
    for rel, path in files.items():
        sha = file_sha1(path)
        plan.hashes[rel] = sha
        if rel not in known:
            plan.new.append(rel)
        elif known[rel].get("sha1") != sha:
            plan.changed.append(rel)
        else:
            plan.unchanged += 1
    return plan


def sync_index(
    config: dict,
    rebuild: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
    embeddings=None,
    store=None,
) -> IndexReport:
    """Bring the vector store in line with ``data.processed_path``.

    Args:
        config: Project config.
        rebuild: Wipe the collection and re-embed everything.
        progress: Optional callback ``(chunks_done, chunks_total, current_file)``.
        embeddings / store: Injection points for tests.

    Returns:
        An :class:`IndexReport` describing what changed.
    """
    started = time.time()
    report = IndexReport()
    processed_dir = Path(config["data"]["processed_path"])
    files = discover_files(processed_dir)
    manifest = load_manifest(config)
    store = store or get_vector_store(config, embeddings)

    # Vectors are only comparable if built with the same chunking + model.
    settings_changed = bool(manifest["files"]) and manifest.get("settings") != index_settings(config)
    orphaned = not manifest["files"] and store_size(store) > 0  # store from an older app version
    if rebuild or settings_changed or orphaned:
        log.info("Rebuilding index from scratch")
        store.reset_collection()
        manifest = empty_manifest(config)
        report.rebuilt = True
    manifest["settings"] = index_settings(config)
    known = manifest["files"]

    plan = plan_changes(config, manifest, files)
    report.unchanged = plan.unchanged
    for rel in plan.removed:
        store.delete(where={"source": rel})
        del known[rel]
        report.removed += 1

    pending = []  # (rel, sha1, chars, metadata, chunks, already_indexed)
    for rel in plan.new + plan.changed:
        try:
            text = read_document_text(files[rel])
            meta = parse_source_metadata(files[rel], processed_dir)
            pending.append((rel, plan.hashes[rel], len(text), meta, chunk_text(text, meta, config), rel in known))
        except Exception as exc:
            report.failed.append((rel, str(exc)))

    total = sum(len(item[4]) for item in pending)
    done = 0
    for rel, sha, chars, meta, chunks, existed in pending:
        try:
            if existed:  # replace: drop stale chunks before adding the new ones
                store.delete(where={"source": rel})
            for start in range(0, len(chunks), ADD_BATCH):
                batch = chunks[start : start + ADD_BATCH]
                store.add_documents(batch, ids=[chunk_id(d) for d in batch])
                done += len(batch)
                if progress:
                    progress(done, total, rel)
        except Exception as exc:
            log.warning("Indexing failed for %s: %s", rel, exc)
            report.failed.append((rel, str(exc)))
            known.pop(rel, None)  # partially written; retry next run
            save_manifest(config, manifest)
            continue
        known[rel] = {
            "sha1": sha,
            "chunks": len(chunks),
            "chars": chars,
            **{k: v for k, v in meta.items() if k in ("title", "specialization", "course", "notes_type")},
        }
        save_manifest(config, manifest)  # checkpoint: safe to interrupt after any file
        report.updated += 1 if existed else 0
        report.added += 0 if existed else 1
        report.chunks_added += len(chunks)

    save_manifest(config, manifest)
    report.seconds = time.time() - started
    return report
