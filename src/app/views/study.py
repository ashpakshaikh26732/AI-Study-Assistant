"""Study tools: summaries, flashcards, quizzes and a mistakes review."""
from __future__ import annotations

import csv
import io
import random

import streamlit as st

from src.app import services, ui
from src.app.views.common import empty_index_onboarding, link_to
from src.features.cache import cache_from_config, make_key
from src.features.flashcard_generator import Flashcard, cards_from_dicts, iter_flashcards
from src.features.quiz_engine import grade_answer
from src.features.summarizer import build_study_context, stream_summary
from src.memory.tracker import get_review_questions, initialize_database, log_attempt
from src.rag_core.vectorstore import get_topic_chunks

VERDICT_UI = {
    "correct": ("Correct!", ":material/check_circle:", st.success),
    "close": ("Close - you're on the right track.", ":material/adjust:", st.warning),
    "incorrect": ("Not quite.", ":material/cancel:", st.error),
}


# ------------------------------------------------------------------- helpers
def _course(scope_course: str | None) -> str | None:
    """The course to study: the sidebar scope, or a picker when scope is 'all'."""
    if scope_course:
        return scope_course
    courses = sorted(c for c in services.library()["course"].unique() if c)
    return st.selectbox("Choose a course to study", courses, index=None, placeholder="Pick a course…")


def _model_id() -> str:
    return services.llm_label(st.session_state.provider)


def _need_llm(feature: str):
    """Return the LLM, or show why it's missing and return None."""
    llm, error = services.safe_llm(st.session_state.provider)
    if llm is None:
        st.warning(
            f"**{feature} needs a language model.** " + (error or "Connect one in the sidebar Settings."),
            icon=":material/smart_toy:",
        )
    return llm


def _topic_chunks(course: str, types):
    return get_topic_chunks(services.get_store(), course, types or None)


def _cards_to_csv(cards: list[Flashcard]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows((c.question, c.answer) for c in cards)  # Anki-importable (front, back)
    return buffer.getvalue()


# ------------------------------------------------------------------- summary
def _summary_tab(course: str, types) -> None:
    config = services.get_config()
    cache = cache_from_config(config)
    n = config["features"]["study"]["max_context_chunks"]
    chunks = _topic_chunks(course, types)
    _, used = build_study_context(chunks, n)
    key = make_key("summary", course, _model_id(), *(c.page_content[:80] for c in used))
    cached = cache.get(key)

    st.caption(f"Summarises {len(used)} passages sampled evenly across {len(chunks)} in this course.")
    if cached:
        st.markdown(cached["text"])
        st.caption("Saved result. Regenerating gives a fresh version.")
    label = "Regenerate summary" if cached else "Generate summary"
    if st.button(label, type="primary", icon=":material/summarize:"):
        llm = _need_llm("Summaries")
        if llm:
            try:
                with st.container():
                    text = st.write_stream(stream_summary(llm, course, chunks, n))
                cache.set(key, {"text": text})
            except Exception as exc:
                st.error(f"The language model failed: {exc}")


# ----------------------------------------------------------------- flashcards
def _load_or_generate_cards(course: str, types, n_cards: int, regenerate: bool, progress=None) -> list[Flashcard]:
    config = services.get_config()
    cache = cache_from_config(config)
    chunks = _topic_chunks(course, types)
    key = make_key("cards", course, _model_id(), n_cards, *(c.page_content[:80] for c in chunks[:50]))
    if not regenerate:
        cached = cards_from_dicts(cache.get(key))
        if cached:
            return cached
    llm = _need_llm("Flashcards")
    if llm is None:
        return []
    cards: list[Flashcard] = []
    try:
        for batch in iter_flashcards(llm, chunks, n_cards, max_chunks=config["features"]["study"]["max_context_chunks"]):
            cards.extend(batch)
            if progress:
                progress(cards)
    except Exception as exc:
        st.error(f"The language model failed: {exc}")
    if cards:
        cache.set(key, [c.to_dict() for c in cards])
    return cards


def _flashcards_tab(course: str, types) -> None:
    default_n = services.get_config()["features"]["study"]["flashcards"]
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    n_cards = left.slider("Number of cards", 4, 16, default_n, key="fc_n")
    regenerate = right.button("New set", icon=":material/refresh:", key="fc_regen")

    deck_key = f"deck::{course}"
    if regenerate or deck_key not in st.session_state:
        if regenerate or st.button("Generate flashcards", type="primary", icon=":material/style:", key="fc_go"):
            box = st.empty()

            def show(cards):
                box.info(f"Generated {len(cards)} of {n_cards} cards…")

            with st.spinner("Writing flashcards from your notes…"):
                cards = _load_or_generate_cards(course, types, n_cards, regenerate, show)
            box.empty()
            if cards:
                st.session_state[deck_key] = {"cards": cards, "i": 0, "show": False}
                st.rerun()
        return

    deck = st.session_state[deck_key]
    cards: list[Flashcard] = deck["cards"]
    i = deck["i"]
    st.progress((i + 1) / len(cards), text=f"Card {i + 1} of {len(cards)}")
    answer_html = f'<div class="a">{_esc(cards[i].answer)}</div>' if deck["show"] else ""
    st.markdown(f'<div class="sa-card"><div class="q">{_esc(cards[i].question)}</div>{answer_html}</div>',
                unsafe_allow_html=True)
    if cards[i].source:
        st.caption(f"From: {cards[i].source}")

    prev, flip, nxt = st.columns(3)
    if prev.button("Previous", disabled=i == 0, width="stretch", icon=":material/arrow_back:"):
        deck.update(i=i - 1, show=False)
        st.rerun()
    if flip.button("Hide answer" if deck["show"] else "Show answer", width="stretch", type="primary"):
        deck["show"] = not deck["show"]
        st.rerun()
    if nxt.button("Next", disabled=i == len(cards) - 1, width="stretch", icon=":material/arrow_forward:"):
        deck.update(i=i + 1, show=False)
        st.rerun()
    st.download_button("Download as CSV (Anki)", _cards_to_csv(cards), file_name=f"{course}_flashcards.csv",
                       mime="text/csv", icon=":material/download:")


def _esc(text: str) -> str:
    import html

    return html.escape(text)


# ---------------------------------------------------------------------- quiz
def _run_quiz(state_key: str, topic_for: callable, cards: list[Flashcard] | None = None, *, restart_label: str) -> None:
    """Shared quiz loop. ``cards`` starts a new quiz; otherwise continues the one in session state."""
    config = services.get_config()
    if cards:
        st.session_state[state_key] = {"cards": cards, "i": 0, "score": 0, "phase": "ask", "last": None, "log": []}
        st.rerun()
    quiz = st.session_state.get(state_key)
    if not quiz:
        return

    total = len(quiz["cards"])
    if quiz["i"] >= total:
        if quiz["score"] == total:
            st.balloons()
        st.success(f"Quiz complete! You scored **{quiz['score']} / {total}**.", icon=":material/emoji_events:")
        import pandas as pd

        st.dataframe(pd.DataFrame(quiz["log"]), hide_index=True, width="stretch")
        if st.button(restart_label, key=f"{state_key}_again", icon=":material/replay:"):
            del st.session_state[state_key]
            st.rerun()
        return

    card: Flashcard = quiz["cards"][quiz["i"]]
    topic = topic_for(card)
    st.progress(quiz["i"] / total, text=f"Question {quiz['i'] + 1} of {total} · score {quiz['score']}")
    st.markdown(f'<div class="sa-card"><div class="q">{_esc(card.question)}</div></div>', unsafe_allow_html=True)

    if quiz["phase"] == "ask":
        with st.form(f"{state_key}_form", clear_on_submit=True):
            answer = st.text_area("Your answer", height=110, placeholder="Answer in your own words…")
            submitted = st.form_submit_button("Check answer", type="primary", icon=":material/task_alt:")
        if submitted:
            verdict, score = grade_answer(answer, card.answer, services.get_embedding_model(), config)
            initialize_database(config)
            log_attempt(topic, card.question, verdict == "correct", config, score, answer, card.answer)
            quiz["score"] += verdict == "correct"
            quiz["last"] = {"verdict": verdict, "score": score, "answer": answer}
            quiz["log"].append({"Question": card.question, "Your answer": answer or "(blank)",
                                "Result": verdict, "Similarity": f"{score:.0%}"})
            quiz["phase"] = "feedback"
            st.rerun()
    else:
        last = quiz["last"]
        message, icon, show = VERDICT_UI[last["verdict"]]
        show(f"**{message}**  Similarity to the reference answer: {last['score']:.0%}", icon=icon)
        st.markdown(f"**Reference answer:** {card.answer}")
        if st.button("Next question" if quiz["i"] + 1 < total else "Finish", type="primary",
                     icon=":material/arrow_forward:", key=f"{state_key}_next"):
            quiz.update(i=quiz["i"] + 1, phase="ask", last=None)
            st.rerun()


def _quiz_tab(course: str, types) -> None:
    n = st.slider("Questions", 3, 12, 5, key="quiz_n")
    deck = st.session_state.get(f"deck::{course}")
    if deck:
        st.caption("Uses the flashcards you already generated for this course (shuffled).")
    if st.button("Start a new quiz", type="primary", icon=":material/quiz:", key="quiz_go"):
        if deck:  # reuse the current deck: no language model needed
            cards = deck["cards"]
        else:
            with st.spinner("Preparing questions from your notes…"):
                cards = _load_or_generate_cards(course, types, n, regenerate=False)
        if cards:
            picked = random.sample(cards, min(n, len(cards)))
            _run_quiz("quiz", lambda _c: course, picked, restart_label="Take another quiz")
    else:
        _run_quiz("quiz", lambda _c: course, None, restart_label="Take another quiz")


def _review_tab() -> None:
    config = services.get_config()
    initialize_database(config)
    review = get_review_questions(config, limit=10)
    if not review and "review" not in st.session_state:
        st.info("No mistakes to review yet - questions you get wrong in quizzes will show up here. "
                "(This mode needs no language model.)", icon=":material/celebration:")
        return
    topics = {r["question"]: r["topic"] for r in review}
    if st.button(f"Review my {len(review)} weakest questions", type="primary", icon=":material/history_edu:",
                 key="review_go") and review:
        cards = [Flashcard(r["question"], r["answer"]) for r in review]
        _run_quiz("review", lambda c: topics.get(c.question, "Review"), cards, restart_label="Review again")
    else:
        _run_quiz("review", lambda c: topics.get(c.question, "Review"), None, restart_label="Review again")


# ---------------------------------------------------------------------- page
def render() -> None:
    st.title("Study tools")
    if services.index_size() == 0:
        empty_index_onboarding()
        return
    scope = st.session_state.scope
    course = _course(scope["course"])
    types = scope["types"]

    summary_tab, cards_tab, quiz_tab, review_tab = st.tabs(
        [":material/summarize: Summary", ":material/style: Flashcards", ":material/quiz: Quiz",
         ":material/history_edu: Review mistakes"]
    )
    with review_tab:
        _review_tab()
    if not course:
        for tab in (summary_tab, cards_tab, quiz_tab):
            with tab:
                st.info("Pick a course above (or in the sidebar) to use this tool.", icon=":material/arrow_upward:")
        return
    with summary_tab:
        _summary_tab(course, types)
    with cards_tab:
        _flashcards_tab(course, types)
    with quiz_tab:
        _quiz_tab(course, types)
    link_to("dashboard", "See your progress", ":material/monitoring:")
