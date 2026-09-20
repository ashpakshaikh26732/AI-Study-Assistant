"""Topic summaries.

The first version map-reduced over *every* chunk of a course - dozens of LLM
calls, minutes on CPU, and easy to overflow the context window. Instead we take
an evenly spread sample of the course's passages and summarise them in a single
streamed call.
"""
from __future__ import annotations

from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from src.features.generator import chunk_text_of
from src.rag_core.vectorstore import sample_evenly

SUMMARY_SYSTEM = (
    "You are an expert academic assistant. You summarise a student's own notes "
    "faithfully - never invent content that is not in the excerpts."
)

SUMMARY_PROMPT = """Below are excerpts from the student's notes on "{topic}".

Write a well-organised study summary:
1. One or two sentences of overview.
2. The key concepts, each with a short explanation.
3. Important formulas, definitions or rules (plain text).
4. A short "remember this" list of the 3-5 most important takeaways.

Excerpts:
{context}
"""


def build_study_context(chunks, max_chunks: int = 8, max_chars: int = 7000) -> tuple[str, list]:
    """Join an even sample of ``chunks`` (Documents) into one prompt block.

    Returns ``(context_text, chunks_used)``.
    """
    used, total = [], 0
    for chunk in sample_evenly(list(chunks), max_chunks):
        if total + len(chunk.page_content) > max_chars and used:
            break
        used.append(chunk)
        total += len(chunk.page_content)
    context = "\n\n---\n\n".join(c.page_content for c in used)
    return context, used


def stream_summary(llm, topic: str, chunks, max_chunks: int = 8) -> Iterator[str]:
    """Yield a summary of ``topic`` piece by piece."""
    context, _ = build_study_context(chunks, max_chunks)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=SUMMARY_PROMPT.format(topic=topic, context=context)),
    ]
    for piece in llm.stream(messages):
        text = chunk_text_of(piece)
        if text:
            yield text
