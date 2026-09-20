"""Ask one question from the command line (no UI) and print the answer + sources.

Usage:
    python run_pipeline.py "What is a bidirectional RNN?"
    python run_pipeline.py "Explain dropout" --course "Improving Deep Neural Networks_ Hyperparameter Tuning, Regularization and Optimization"
    python run_pipeline.py "What is a GRU?" --provider none      # retrieval only, no LLM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.features.generator import NOT_FOUND_MESSAGE, extractive_answer, stream_answer  # noqa: E402
from src.llm.model_loader import LLMUnavailableError, load_llm  # noqa: E402
from src.rag_core.retriever import retrieve  # noqa: E402
from src.rag_core.vectorstore import get_vector_store, store_size  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask the study assistant a question")
    parser.add_argument("question")
    parser.add_argument("--config", default=None)
    parser.add_argument("--course", default=None, help="Only search this course")
    parser.add_argument("--provider", default=None, help="auto | ollama | gemini | transformers | none")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    store = get_vector_store(config)
    if store_size(store) == 0:
        print("The vector store is empty. Run `python build_vector_store.py` first.")
        return 1

    cfg = config["rag_core"]["retriever"]
    chunks = retrieve(
        store, args.question, k=cfg["k"], course=args.course,
        min_score=cfg["min_score"], max_per_source=cfg["max_per_source"],
    )
    if not chunks:
        print(NOT_FOUND_MESSAGE)
        return 0

    try:
        llm = load_llm(config, args.provider)
    except LLMUnavailableError as exc:
        print(f"[LLM unavailable] {exc}\n")
        llm = None

    print("Answer:\n")
    if llm is None:
        print(extractive_answer(chunks))
    else:
        for piece in stream_answer(llm, args.question, chunks):
            print(piece, end="", flush=True)
        print()

    print("\nSources:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] {chunk.title} ({chunk.course} - {chunk.notes_type})  relevance {chunk.score:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
