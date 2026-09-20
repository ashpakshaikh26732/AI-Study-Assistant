"""Build or update the vector store from the processed notes.

Incremental by default: only new or changed files are embedded, files that were
deleted are removed, and progress is checkpointed after every file, so it is
safe to interrupt and re-run.

Usage:
    python build_vector_store.py                 # sync with data/processed
    python build_vector_store.py --rebuild       # wipe and re-embed everything
    python build_vector_store.py --config my.yaml
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.rag_core.indexer import discover_files, sync_index  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI Study Assistant - build/update the vector store")
    parser.add_argument("--config", default=None, help="Path to a config YAML (default: config.yaml)")
    parser.add_argument("--rebuild", action="store_true", help="Wipe the index and re-embed everything")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    processed = config["data"]["processed_path"]
    if not discover_files(processed):
        print(f"No .txt/.md files found in {processed}.")
        print("Run `python run_preprocessing.py` first (or put text files there).")
        return 1

    print(f"Indexing {processed}")
    started = time.time()

    def progress(done, total, current):
        elapsed = max(time.time() - started, 1e-6)
        eta = (total - done) / (done / elapsed) if done else 0
        print(f"\r[{done}/{total} chunks] ~{eta / 60:4.1f} min left  {current[:60]:<60}", end="", flush=True)

    report = sync_index(config, rebuild=args.rebuild, progress=progress)
    print()
    print(
        f"Done in {report.seconds:.0f}s: {report.added} added, {report.updated} updated, "
        f"{report.removed} removed, {report.unchanged} unchanged "
        f"({report.chunks_added} chunks embedded)."
    )
    for path, error in report.failed:
        print(f"  FAILED {path}: {error}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
