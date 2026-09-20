"""Turn raw study material (PDF / text / markdown) into cleaned ``.txt`` files.

Typed PDFs are read with PyMuPDF. Pages with no selectable text (scans,
handwriting) are OCR'd with Tesseract *per page*, so a PDF that mixes typed and
scanned pages is handled correctly. Pages are rendered in memory - no temporary
image files are written.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from src.preprocessing.text_cleaner import cleaning_fn

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = {".pdf"} | TEXT_SUFFIXES


def tesseract_available() -> bool:
    """True when the ``tesseract`` binary is on PATH."""
    return shutil.which("tesseract") is not None


def _open_pdf(file_path):
    import pymupdf  # imported lazily so the rest of the app works without it

    return pymupdf.open(file_path)


def _ocr_page(page, dpi: int, language: str) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(image, lang=language)


def extract_text_from_pdf(file_path) -> str:
    """Extract selectable text from a PDF with PyMuPDF (no OCR).

    Returns an empty string if the file cannot be read.
    """
    try:
        with _open_pdf(file_path) as doc:
            return "\n\n".join(page.get_text() for page in doc)
    except Exception as exc:  # corrupt / encrypted / missing file
        log.warning("Could not read %s: %s", file_path, exc)
        return ""


def ocr_pdf(file_path, config: dict) -> str:
    """OCR every page of a PDF (for fully scanned documents)."""
    ocr_cfg = config.get("ocr", {})
    if not tesseract_available():
        log.warning("Tesseract is not installed; cannot OCR %s", file_path)
        return ""
    try:
        with _open_pdf(file_path) as doc:
            return "\n\n".join(
                _ocr_page(page, ocr_cfg.get("dpi", 200), ocr_cfg.get("language", "eng"))
                for page in doc
            )
    except Exception as exc:
        log.warning("OCR failed for %s: %s", file_path, exc)
        return ""


def extract_pdf(file_path, config: dict) -> tuple[str, dict]:
    """Hybrid extraction: text layer first, OCR only for pages that lack one.

    Returns:
        ``(text, stats)`` where stats has ``pages``, ``ocr_pages`` and
        ``blank_pages`` (pages with no text that could not be OCR'd).
    """
    ocr_cfg = config.get("ocr", {})
    can_ocr = ocr_cfg.get("enabled", True) and tesseract_available()
    min_chars = ocr_cfg.get("min_chars_per_page", 25)
    stats = {"pages": 0, "ocr_pages": 0, "blank_pages": 0}
    parts = []
    with _open_pdf(file_path) as doc:
        for page in doc:
            stats["pages"] += 1
            text = page.get_text()
            if len(text.strip()) < min_chars:
                if can_ocr:
                    text = _ocr_page(page, ocr_cfg.get("dpi", 200), ocr_cfg.get("language", "eng"))
                    stats["ocr_pages"] += 1
                else:
                    stats["blank_pages"] += 1
            parts.append(text)
    return "\n\n".join(parts), stats


def process_document(src: Path, dst: Path, config: dict) -> dict:
    """Extract + clean one file and write it to ``dst``. Returns extraction stats."""
    stats = {"pages": 0, "ocr_pages": 0, "blank_pages": 0}
    if src.suffix.lower() == ".pdf":
        text, stats = extract_pdf(src, config)
    else:
        text = src.read_text(encoding="utf-8-sig", errors="replace")
    text = cleaning_fn(text)
    stats["chars"] = len(text)
    if text:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    return stats


def process_all_documents(
    config: dict,
    force: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Process every supported file under ``data.raw_path`` into ``data.processed_path``.

    The folder structure is mirrored. Files whose output is already newer than
    the source are skipped unless ``force`` is set - so re-running is cheap and
    your manual corrections to processed files are never overwritten.

    Args:
        config: Loaded project config.
        force: Reprocess everything (overwrites existing processed files!).
        progress: Optional callback ``(index, total, relative_path)``.

    Returns:
        Summary dict: ``processed``, ``skipped``, ``failed`` (list of
        ``(path, error)``), ``empty`` (paths that produced no text) and
        ``ocr_pages``.
    """
    raw_root = Path(config["data"]["raw_path"])
    out_root = Path(config["data"]["processed_path"])
    summary = {"processed": 0, "skipped": 0, "failed": [], "empty": [], "ocr_pages": 0}
    if not raw_root.exists():
        log.warning("Raw data folder does not exist: %s", raw_root)
        return summary

    files = sorted(
        p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(raw_root)
        dst = out_root / rel.with_suffix(".txt")
        if progress:
            progress(i, len(files), rel.as_posix())
        if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            summary["skipped"] += 1
            continue
        try:
            stats = process_document(src, dst, config)
        except Exception as exc:
            log.warning("Failed to process %s: %s", rel, exc)
            summary["failed"].append((rel.as_posix(), str(exc)))
            continue
        summary["ocr_pages"] += stats["ocr_pages"]
        if stats["chars"] == 0:
            summary["empty"].append(rel.as_posix())
        else:
            summary["processed"] += 1
    return summary
