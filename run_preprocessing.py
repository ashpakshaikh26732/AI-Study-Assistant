"""Step 1: convert raw PDFs (and .txt/.md) into cleaned text files.

Reads   data/raw/<Specialization>/<Course>/<NotesType>/file.pdf
Writes  data/processed/<same structure>/file.txt

Typed PDFs are read directly; pages without selectable text (scans, handwriting)
are OCR'd if Tesseract is installed. Files already processed are skipped, so your
manual corrections are never overwritten (use --force to redo everything).

Usage:
    python run_preprocessing.py
    python run_preprocessing.py --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.preprocessing.document_parser import process_all_documents, tesseract_available  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI Study Assistant - raw data preprocessing")
    parser.add_argument("--config", default=None, help="Path to a config YAML (default: config.yaml)")
    parser.add_argument("--force", action="store_true", help="Reprocess files even if output exists")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Raw notes:      {config['data']['raw_path']}")
    print(f"Processed into: {config['data']['processed_path']}")
    if config["ocr"]["enabled"] and not tesseract_available():
        print("Note: Tesseract isn't installed, so scanned/handwritten pages can't be OCR'd "
              "(install it with `sudo apt install tesseract-ocr`).")

    def progress(i, total, rel):
        print(f"[{i}/{total}] {rel}")

    summary = process_all_documents(config, force=args.force, progress=progress)
    print(
        f"\nProcessed {summary['processed']}, skipped {summary['skipped']} already done, "
        f"{summary['ocr_pages']} pages OCR'd."
    )
    for path in summary["empty"]:
        print(f"  no text extracted: {path}")
    for path, error in summary["failed"]:
        print(f"  FAILED {path}: {error}")
    print("\nNext: review the .txt files if you like, then run `python build_vector_store.py`.")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
