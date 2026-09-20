"""Chat with your notes: scoped retrieval, streamed grounded answers, sources."""
from __future__ import annotations

import streamlit as st

from src.app import services, ui
from src.app.views.common import empty_index_onboarding
from src.features.generator import NOT_FOUND_MESSAGE, condense_query, extractive_answer, stream_answer
from src.rag_core.retriever import retrieve, snippet
from src.voice.text_to_speech import convert_text_to_speech


def _source_dicts(chunks) -> list[dict]:
    return [
        {"title": c.title, "course": c.course, "notes_type": c.notes_type, "score": c.score, "text": c.text}
        for c in chunks
    ]


def _suggestions(course: str | None) -> list[str]:
    if course:
        return [
            f"Give me an overview of {course}",
            f"What are the most important concepts in {course}?",
            f"What formulas or rules should I remember from {course}?",
        ]
    return [
        "What is the difference between a GRU and an LSTM?",
        "Explain how dropout regularization works",
        "What is retrieval-augmented generation?",
    ]


def _answer(question: str, scope: dict, history: list[dict]):
    """Retrieve, then produce the answer. Returns ``(markdown_answer, source_dicts)``."""
    config = services.get_config()
    r_cfg = config["rag_core"]["retriever"]
    store = services.get_store()

    with st.spinner("Searching your notes…"):
        chunks = []
        # Follow-ups are tried with the previous question folded in first; if that finds
        # nothing (e.g. the previous question was about something else), retry on its own.
        for query in dict.fromkeys([condense_query(question, history), question]):
            chunks = retrieve(
                store,
                query,
                k=scope["k"],
                course=scope["course"],
                notes_types=scope["types"],
                min_score=r_cfg["min_score"],
                max_per_source=r_cfg["max_per_source"],
            )
            if chunks:
                break

    if not chunks:
        closest = retrieve(store, question, k=3, course=scope["course"], notes_types=scope["types"])
        st.markdown(NOT_FOUND_MESSAGE)
        if closest:
            with st.expander("Closest matches (low relevance)"):
                for c in closest:
                    st.caption(f"{c.title} · {c.course} · {c.score:.0%}")
                    st.write(snippet(c.text, 250))
        return NOT_FOUND_MESSAGE, []

    llm, error = services.safe_llm(st.session_state.provider)
    sources = _source_dicts(chunks)
    if llm is None:
        if error:
            st.warning(error)
        text = extractive_answer(chunks)
        st.markdown(text)
        return text, sources

    try:
        turns = config["chat"]["history_turns"]
        text = st.write_stream(stream_answer(llm, question, chunks, history, turns))
    except Exception as exc:  # backend went away mid-answer, bad key, ...
        st.error(f"The language model failed: {exc}")
        text = extractive_answer(chunks)
        st.markdown(text)
    return text, sources


def render() -> None:
    scope = st.session_state.scope
    st.title("Chat with your notes")
    where = scope["course"] or "all your notes"
    types = f" · {', '.join(scope['types'])}" if scope["types"] else ""
    st.markdown(f'<p class="sa-hero">Searching <b>{where}</b>{types}</p>', unsafe_allow_html=True)

    if services.index_size() == 0:
        empty_index_onboarding()
        return

    messages = st.session_state.setdefault("messages", [])

    if not messages:
        st.write("Try one of these, or type your own question below:")
        for suggestion in _suggestions(scope["course"]):
            if st.button(suggestion, key=f"sugg_{suggestion}", icon=":material/lightbulb:"):
                st.session_state.pending_question = suggestion
                st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            ui.render_sources(message.get("sources", []))

    question = st.chat_input("Ask anything about your notes…") or st.session_state.pop("pending_question", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        text, sources = _answer(question, scope, history)
        ui.render_sources(sources)
        if st.session_state.get("voice_replies"):
            audio = convert_text_to_speech(text)
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
            else:
                st.caption("Voice reply unavailable (it needs an internet connection).")
    messages.append({"role": "assistant", "content": text, "sources": sources})
