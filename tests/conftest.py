"""Shared fixtures: an isolated config and offline fakes (no model downloads needed)."""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words are similar."""

    DIM = 128

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.DIM)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def cfg(tmp_path):
    """The default config with every path redirected into a temp dir."""
    from src.config import load_config

    config = load_config()
    config["data"].update(
        raw_path=str(tmp_path / "raw"),
        processed_path=str(tmp_path / "processed"),
        cache_path=str(tmp_path / "cache"),
    )
    config["memory"]["sqlite_database_path"] = str(tmp_path / "memory.db")
    config["rag_core"]["database"]["persist_directory"] = str(tmp_path / "vector_store")
    config["rag_core"]["chunking"].update(chunk_size=200, chunk_overlap=20)
    return config


@pytest.fixture
def notes(cfg):
    """A small processed-notes tree; returns a helper to add/modify files."""
    root = Path(cfg["data"]["processed_path"])

    def write(rel: str, text: str) -> Path:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    write("DL Spec/Sequence Models/handwritten notes/gru.txt",
          "A GRU has an update gate and a reset gate.\n\nThe update gate decides how much of the past to keep. " * 4)
    write("DL Spec/Sequence Models/lecture slides/lstm.txt",
          "An LSTM has forget, input and output gates plus a cell state.\n\nIt handles long sequences well. " * 4)
    write("DL Spec/Optimization/handwritten notes/adam.txt",
          "Adam combines momentum and RMSprop with bias correction for faster gradient descent. " * 6)
    return write
