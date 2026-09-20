"""Semantic grading of quiz answers (meaning over exact wording)."""
from __future__ import annotations

import numpy as np


def similarity_score(user_answer: str, correct_answer: str, embedding_model) -> float:
    """Cosine similarity (-1..1) between the two answers' embeddings; 0 for an empty answer."""
    if not user_answer or not user_answer.strip():
        return 0.0
    user_vec, correct_vec = embedding_model.embed_documents([user_answer, correct_answer])
    denom = np.linalg.norm(user_vec) * np.linalg.norm(correct_vec)
    if denom == 0:
        return 0.0
    return float(np.dot(user_vec, correct_vec) / denom)


def grade_answer(user_answer: str, correct_answer: str, embedding_model, config: dict) -> tuple[str, float]:
    """Grade an answer.

    Returns:
        ``(verdict, score)`` where verdict is ``"correct"`` (score >= threshold),
        ``"close"`` (within ``close_margin`` below it) or ``"incorrect"``.
    """
    quiz_cfg = config["features"]["quiz"]
    score = similarity_score(user_answer, correct_answer, embedding_model)
    threshold = quiz_cfg["similarity_threshold"]
    if score >= threshold:
        return "correct", score
    if score >= threshold - quiz_cfg.get("close_margin", 0.15):
        return "close", score
    return "incorrect", score


def grade_user_answer(user_answer, correct_answer, embedding_model, config) -> bool:
    """True if the answer's meaning matches closely enough (kept for compatibility)."""
    return similarity_score(user_answer, correct_answer, embedding_model) >= config["features"]["quiz"][
        "similarity_threshold"
    ]
