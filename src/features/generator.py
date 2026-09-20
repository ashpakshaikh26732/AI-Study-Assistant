"""Grounded question answering over retrieved passages.

Built directly on LangChain-core messages instead of ``RetrievalQA`` (which was
removed from ``langchain`` 1.x), so it works with any chat model and streams.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.rag_core.retriever import RetrievedChunk, snippet

SYSTEM_PROMPT = """You are a study assistant answering questions from the student's own notes.

Rules:
- Use ONLY the numbered passages in the context. Do not add outside facts.
- Cite the passages you used like [1] or [2][3] right after the claim.
- If the passages do not contain the answer, say "I couldn't find that in your notes." and stop.
- Be clear and concise. Use short bullet points or steps when they help. Write formulas in plain text."""

_MAX_HISTORY_CHARS = 1200


def chunk_text_of(message_chunk) -> str:
    """Plain text of a streamed chat-model chunk (handles string or block-list content)."""
    content = getattr(message_chunk, "content", message_chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return ""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the passages so the model can cite them."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.title} ({chunk.course} - {chunk.notes_type})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


_FOLLOWUP_PHRASES = ("what about", "how about", "what else", "explain more", "tell me more", "go on")
_REFERENCE_WORDS = {"it", "its", "they", "them", "their", "this", "these", "those"}


def _is_followup(question: str) -> bool:
    words = re.findall(r"[a-z']+", question.lower())
    if not words or len(words) > 12:
        return False
    text = " ".join(words)
    if any(text.startswith(p) for p in _FOLLOWUP_PHRASES) or words[0] in {"and", "also", "but", "so", "then"}:
        return True
    if len(words) <= 2 and words[0] in {"why", "how", "example", "examples"}:
        return True  # "why?", "how so?", "examples?" - but not "what is dropout"
    return bool(_REFERENCE_WORDS & set(words))  # "how does it work?"


def condense_query(question: str, history: Optional[list[dict]]) -> str:
    """Make follow-ups ("and what about GRUs?", "how does it work?") retrievable.

    A cheap heuristic instead of an extra LLM call (which is slow on CPU): when the
    question *looks like* a follow-up, prepend the previous user question for retrieval.
    Self-contained questions are left alone, even short ones.
    """
    if history and _is_followup(question):
        previous = [m["content"] for m in history if m.get("role") == "user"]
        if previous:
            return f"{previous[-1]} {question}"
    return question


def build_messages(
    question: str, chunks: list[RetrievedChunk], history: Optional[list[dict]] = None, turns: int = 3
) -> list[BaseMessage]:
    """System prompt + recent turns + the question with its numbered context."""
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in (history or [])[-2 * turns :]:
        text = m["content"][:_MAX_HISTORY_CHARS]
        messages.append(HumanMessage(content=text) if m["role"] == "user" else AIMessage(content=text))
    messages.append(
        HumanMessage(content=f"Context from my notes:\n\n{format_context(chunks)}\n\nQuestion: {question}")
    )
    return messages


def stream_answer(
    llm, question: str, chunks: list[RetrievedChunk], history: Optional[list[dict]] = None, turns: int = 3
) -> Iterator[str]:
    """Yield the answer piece by piece as the model generates it."""
    for piece in llm.stream(build_messages(question, chunks, history, turns)):
        text = chunk_text_of(piece)
        if text:
            yield text


def answer_question(llm, question: str, chunks: list[RetrievedChunk], history: Optional[list[dict]] = None) -> str:
    """Non-streaming convenience wrapper around :func:`stream_answer`."""
    return "".join(stream_answer(llm, question, chunks, history)).strip()


def extractive_answer(chunks: list[RetrievedChunk], max_chars: int = 420) -> str:
    """No-LLM fallback: show the best-matching passages as the answer."""
    if not chunks:
        return "I couldn't find anything relevant in your notes for that."
    lines = ["**No language model is connected, so here are the best matches from your notes:**", ""]
    for i, chunk in enumerate(chunks[:3], start=1):
        lines.append(f"{i}. **{chunk.title}** *({chunk.course})*  \n   {snippet(chunk.text, max_chars)}")
    return "\n".join(lines)


NOT_FOUND_MESSAGE = "I couldn't find that in your notes. Try rephrasing, or widen the scope in the sidebar."


class _QAChain:
    """Backward-compatible ``.invoke(question)`` wrapper (old RetrievalQA-style dict)."""

    def __init__(self, store, llm, config):
        self.store, self.llm, self.config = store, llm, config

    def invoke(self, question: str) -> dict:
        from src.rag_core.retriever import retrieve

        cfg = self.config["rag_core"]["retriever"]
        chunks = retrieve(
            self.store, question, k=cfg["k"], min_score=cfg["min_score"], max_per_source=cfg["max_per_source"]
        )
        if not chunks:
            return {"result": NOT_FOUND_MESSAGE, "source_documents": []}
        answer = answer_question(self.llm, question, chunks) if self.llm else extractive_answer(chunks)
        return {"result": answer, "source_documents": chunks}


def create_qa_chain(store, llm, config: dict) -> _QAChain:
    """Build a question-answering helper: ``create_qa_chain(store, llm, config).invoke(q)``."""
    return _QAChain(store, llm, config)
