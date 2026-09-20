import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.features.cache import JsonCache, make_key
from src.features.flashcard_generator import Flashcard, generate_flashcards, iter_flashcards, parse_flashcards
from src.features.generator import (
    NOT_FOUND_MESSAGE, answer_question, build_messages, chunk_text_of, condense_query, create_qa_chain,
    extractive_answer, format_context, stream_answer,
)
from src.features.summarizer import build_study_context, stream_summary
from src.llm import model_loader
from src.llm.model_loader import BackendStatus, LLMUnavailableError, load_llm, resolve_provider
from src.rag_core.indexer import sync_index
from src.rag_core.retriever import RetrievedChunk
from src.rag_core.vectorstore import get_vector_store


def chunk(text="Passage text.", title="doc", course="Course", notes_type="lecture slides", score=0.8):
    return RetrievedChunk(text, score, {"source": f"{title}.txt", "title": title, "course": course,
                                        "notes_type": notes_type})


class TestProviderSelection:
    def _no_backends(self, monkeypatch):
        down = lambda name: (lambda config: BackendStatus(name, False, "down", "m"))
        monkeypatch.setattr(model_loader, "_cuda_available", lambda: False)
        monkeypatch.setattr(model_loader, "_ollama_status", down("ollama"))
        monkeypatch.setattr(model_loader, "_gemini_status", down("gemini"))
        monkeypatch.setattr(model_loader, "_transformers_status", down("transformers"))
        monkeypatch.setitem(model_loader._STATUS, "ollama", model_loader._ollama_status)
        monkeypatch.setitem(model_loader._STATUS, "gemini", model_loader._gemini_status)

    def test_auto_with_nothing_available_is_none(self, cfg, monkeypatch):
        self._no_backends(monkeypatch)
        assert resolve_provider(cfg, "auto") == "none"
        assert load_llm(cfg, "auto") is None

    def test_none_returns_no_llm(self, cfg):
        assert load_llm(cfg, "none") is None

    def test_auto_prefers_gpu_then_ollama_never_gemini(self, cfg, monkeypatch):
        self._no_backends(monkeypatch)
        monkeypatch.setattr(model_loader, "_gemini_status", lambda c: BackendStatus("gemini", True, "ok", "m"))
        assert resolve_provider(cfg, "auto") == "none"  # cloud is opt-in only
        monkeypatch.setattr(model_loader, "_ollama_status", lambda c: BackendStatus("ollama", True, "ok", "m"))
        assert resolve_provider(cfg, "auto") == "ollama"
        monkeypatch.setattr(model_loader, "_transformers_status", lambda c: BackendStatus("transformers", True, "ok", "m"))
        assert resolve_provider(cfg, "auto") == "transformers"

    def test_explicit_unavailable_backend_raises_actionable_error(self, cfg, monkeypatch):
        self._no_backends(monkeypatch)
        with pytest.raises(LLMUnavailableError, match="down"):
            load_llm(cfg, "ollama")

    def test_unknown_provider(self, cfg):
        with pytest.raises(LLMUnavailableError, match="Unknown"):
            load_llm(cfg, "skynet")

    def test_ollama_status_when_server_is_down(self, cfg):
        cfg["llm"]["ollama"]["base_url"] = "http://127.0.0.1:9"  # nothing listens here
        status = model_loader._ollama_status(cfg)
        assert not status.available and "ollama pull" in status.detail

    def test_ollama_status_model_missing_and_present(self, cfg, monkeypatch):
        import io
        import urllib.request

        def fake_urlopen(tags):
            return lambda url, timeout=0: io.BytesIO(json.dumps({"models": [{"name": n} for n in tags]}).encode())

        model = cfg["llm"]["ollama"]["model"]
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(["other:1b"]))
        assert not model_loader._ollama_status(cfg).available
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen([model]))
        assert model_loader._ollama_status(cfg).available

    def test_gemini_needs_key(self, cfg, monkeypatch):
        monkeypatch.delenv(cfg["llm"]["gemini"]["api_key_env"], raising=False)
        assert not model_loader._gemini_status(cfg).available


class TestGenerator:
    def test_chunk_text_of_handles_strings_and_block_lists(self):
        assert chunk_text_of(AIMessage(content="hi")) == "hi"
        assert chunk_text_of(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"
        assert chunk_text_of("raw") == "raw"

    def test_context_is_numbered_for_citations(self):
        ctx = format_context([chunk("Alpha", "one"), chunk("Beta", "two")])
        assert ctx.startswith("[1] one (Course - lecture slides)\nAlpha") and "[2] two" in ctx

    def test_followups_borrow_the_previous_question(self):
        history = [{"role": "user", "content": "What is a GRU?"}, {"role": "assistant", "content": "A gate unit."}]
        assert condense_query("and an LSTM?", history) == "What is a GRU? and an LSTM?"
        assert condense_query("How does it work?", history) == "What is a GRU? How does it work?"
        assert condense_query("Why?", history) == "What is a GRU? Why?"
        assert condense_query("what about LSTMs", history).startswith("What is a GRU?")
        assert condense_query("and an LSTM?", []) == "and an LSTM?"  # nothing to borrow

    def test_self_contained_questions_are_never_rewritten(self):
        """Regression: a short unrelated question must not inherit the previous topic."""
        history = [{"role": "user", "content": "What is a bidirectional RNN?"}, {"role": "assistant", "content": "..."}]
        for question in ("what is the capital of Peru", "Define dropout", "What is backpropagation?",
                         "Please explain in detail how backpropagation through time works"):
            assert condense_query(question, history) == question

    def test_snippet_strips_markdown_headings_and_truncates(self):
        from src.rag_core.retriever import snippet

        assert snippet("## Bi-directional RNNs\n\nThis diagram   shows more.") == "Bi-directional RNNs This diagram shows more."
        long = snippet("word " * 200, 50)
        assert len(long) <= 51 and long.endswith("…")

    def test_messages_have_system_history_and_context(self):
        history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        msgs = build_messages("q2", [chunk("Alpha")], history)
        assert isinstance(msgs[0], SystemMessage) and "ONLY" in msgs[0].content
        assert isinstance(msgs[1], HumanMessage) and isinstance(msgs[2], AIMessage)
        assert "Alpha" in msgs[-1].content and msgs[-1].content.endswith("Question: q2")

    def test_history_is_limited(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(20)]
        assert len(build_messages("q", [chunk()], history, turns=2)) == 1 + 4 + 1

    def test_streaming_and_full_answer(self):
        llm = FakeListChatModel(responses=["The answer is 42 [1]."])
        pieces = list(stream_answer(llm, "q", [chunk()]))
        assert len(pieces) > 1 and "".join(pieces) == "The answer is 42 [1]."
        assert answer_question(FakeListChatModel(responses=["  done  "]), "q", [chunk()]) == "done"

    def test_extractive_fallback(self):
        text = extractive_answer([chunk("x " * 500, "big")])
        assert "No language model" in text and "big" in text and "…" in text
        assert "couldn't find" in extractive_answer([])

    def test_qa_chain_end_to_end(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        store = get_vector_store(cfg, fake_embeddings)
        cfg["rag_core"]["retriever"]["min_score"] = 0.2
        result = create_qa_chain(store, FakeListChatModel(responses=["Two gates [1]."]), cfg).invoke("update gate reset gate")
        assert result["result"] == "Two gates [1]." and result["source_documents"][0].title == "gru"
        offline = create_qa_chain(store, None, cfg).invoke("update gate reset gate")
        assert "No language model" in offline["result"]
        nothing = create_qa_chain(store, None, cfg).invoke("zebra quantum pineapple")
        assert nothing == {"result": NOT_FOUND_MESSAGE, "source_documents": []}


class TestFlashcards:
    def test_plain_json_array(self):
        cards = parse_flashcards('[{"question": "What is X?", "answer": "Y."}]')
        assert cards == [Flashcard("What is X?", "Y.")]

    def test_code_fences_and_chatter(self):
        raw = 'Sure! Here you go:\n```json\n[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]\n```\nEnjoy!'
        assert [c.question for c in parse_flashcards(raw)] == ["Q1", "Q2"]

    def test_wrapped_in_object(self):
        assert len(parse_flashcards('{"flashcards": [{"question": "Q", "answer": "A"}]}')) == 1

    def test_truncated_json_is_salvaged(self):
        raw = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2 cut of'
        assert [c.question for c in parse_flashcards(raw)] == ["Q1"]

    def test_garbage_gives_nothing(self):
        assert parse_flashcards("I cannot do that.") == [] and parse_flashcards("") == []

    def test_items_missing_fields_are_skipped(self):
        assert parse_flashcards('[{"question": "Q"}, {"question": "Q2", "answer": "A2"}]') == [Flashcard("Q2", "A2")]

    def test_generation_batches_dedupes_and_caps(self, cfg, notes, fake_embeddings):
        from src.rag_core.vectorstore import get_topic_chunks

        sync_index(cfg, embeddings=fake_embeddings)
        docs = get_topic_chunks(get_vector_store(cfg, fake_embeddings), "Sequence Models")
        reply = json.dumps([{"question": "Same?", "answer": "a"}, {"question": "Other?", "answer": "b"}])
        llm = FakeListChatModel(responses=[reply])
        batches = list(iter_flashcards(llm, docs, n_cards=3, group_size=1, max_chunks=4))
        cards = [c for b in batches for c in b]
        assert len(cards) == 2 and {c.question for c in cards} == {"Same?", "Other?"}  # duplicates removed
        assert len(generate_flashcards(FakeListChatModel(responses=[reply]), docs, n_cards=1)) == 1  # capped


class TestSummaryAndCache:
    def test_study_context_samples_within_budget(self, cfg, notes, fake_embeddings):
        from langchain_core.documents import Document

        docs = [Document(page_content=f"passage {i} " * 20) for i in range(30)]
        context, used = build_study_context(docs, max_chunks=5, max_chars=100000)
        assert len(used) == 5 and context.count("---") == 4
        _, small = build_study_context(docs, max_chunks=30, max_chars=500)
        assert 1 <= len(small) < 30

    def test_stream_summary(self):
        from langchain_core.documents import Document

        out = "".join(stream_summary(FakeListChatModel(responses=["Summary."]), "Topic", [Document(page_content="text")]))
        assert out == "Summary."

    def test_json_cache_roundtrip(self, tmp_path):
        cache = JsonCache(tmp_path / "c")
        assert cache.get("k") is None
        cache.set("k", {"text": "héllo"})
        assert cache.get("k") == {"text": "héllo"}
        assert make_key("a", 1) == make_key("a", 1) != make_key("a", 2)
