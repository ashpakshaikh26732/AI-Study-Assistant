"""Flashcard generation with tolerant JSON parsing.

Small local models often wrap JSON in prose or code fences, so instead of a
strict parser we pull the JSON out of whatever came back, and fall back to
regex-extracting question/answer pairs. Cards are produced in small batches so
the UI can show them as they arrive.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.features.generator import chunk_text_of
from src.rag_core.vectorstore import sample_evenly


@dataclass
class Flashcard:
    question: str
    answer: str
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


FLASHCARD_SYSTEM = "You write accurate study flashcards from a student's notes. You reply with JSON only."

FLASHCARD_PROMPT = """From the excerpts below, write {n} flashcards that test the most important concepts, definitions and facts.

Rules:
- Each question must be answerable from the excerpts alone.
- Keep answers to one or two sentences.
- Reply with ONLY a JSON array, no other text, in exactly this shape:
[{{"question": "...", "answer": "..."}}]

Excerpts:
{context}
"""

_PAIR_RE = re.compile(
    r'"question"\s*:\s*"(?P<q>(?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"(?P<a>(?:[^"\\]|\\.)*)"', re.DOTALL
)


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_flashcards(text: str) -> list[Flashcard]:
    """Extract flashcards from raw model output; returns [] if nothing usable."""
    if not text:
        return []
    text = re.sub(r"```(?:json)?", "", text)

    candidates = []
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("flashcards", data.get("cards", [data]))
        cards = [
            Flashcard(_clean(item["question"]), _clean(item["answer"]))
            for item in data
            if isinstance(item, dict) and item.get("question") and item.get("answer")
        ]
        if cards:
            return cards

    # Truncated / malformed JSON: salvage whatever complete pairs are present.
    cards = []
    for m in _PAIR_RE.finditer(text):
        try:
            q, a = json.loads(f'"{m.group("q")}"'), json.loads(f'"{m.group("a")}"')
        except json.JSONDecodeError:
            q, a = m.group("q"), m.group("a")
        if q.strip() and a.strip():
            cards.append(Flashcard(_clean(q), _clean(a)))
    return cards


def _generate_batch(llm, context: str, n: int) -> list[Flashcard]:
    messages = [
        SystemMessage(content=FLASHCARD_SYSTEM),
        HumanMessage(content=FLASHCARD_PROMPT.format(n=n, context=context)),
    ]
    raw = "".join(chunk_text_of(p) for p in llm.stream(messages))
    return parse_flashcards(raw)


def iter_flashcards(llm, chunks, n_cards: int = 8, group_size: int = 2, max_chunks: int = 8) -> Iterator[list[Flashcard]]:
    """Yield lists of new flashcards, one list per LLM call, until ``n_cards`` exist.

    ``chunks`` are LangChain Documents. An even sample of at most ``max_chunks``
    is split into groups of ``group_size`` passages, one LLM call per group.
    """
    sample = sample_evenly(list(chunks), max_chunks)
    groups = [sample[i : i + group_size] for i in range(0, len(sample), group_size)] or []
    if not groups:
        return
    per_group = max(2, -(-n_cards // len(groups)))  # ceil division, at least 2
    seen: set[str] = set()
    produced = 0
    for group in groups:
        if produced >= n_cards:
            break
        context = "\n\n---\n\n".join(c.page_content for c in group)
        source = group[0].metadata.get("title", "")
        new = []
        for card in _generate_batch(llm, context, per_group):
            key = card.question.lower()
            if key in seen:
                continue
            seen.add(key)
            card.source = source
            new.append(card)
        new = new[: n_cards - produced]
        produced += len(new)
        if new:
            yield new


def generate_flashcards(llm, chunks, n_cards: int = 8, **kwargs) -> list[Flashcard]:
    """All flashcards at once (non-streaming)."""
    return [card for batch in iter_flashcards(llm, chunks, n_cards, **kwargs) for card in batch]


def cards_from_dicts(items: Optional[list]) -> list[Flashcard]:
    """Rebuild cards from cached JSON."""
    return [Flashcard(**item) for item in (items or [])]
