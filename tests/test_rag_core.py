from pathlib import Path

from src.rag_core.chunker import chunk_id, chunk_single_document, chunk_text, parse_source_metadata
from src.rag_core.indexer import discover_files, plan_changes, sync_index
from src.rag_core.retriever import retrieve
from src.rag_core.vectorstore import (
    build_filter, collection_space, distance_to_similarity, get_topic_chunks, get_vector_store,
    library_table, sample_evenly, store_size,
)


class TestMetadata:
    def test_three_levels(self, tmp_path):
        f = tmp_path / "Spec" / "Course" / "lecture slides" / "w1.txt"
        meta = parse_source_metadata(f, tmp_path)
        assert meta == {"source": "Spec/Course/lecture slides/w1.txt", "title": "w1", "specialization": "Spec",
                        "course": "Course", "notes_type": "lecture slides"}

    def test_two_levels_are_course_and_type(self, tmp_path):
        meta = parse_source_metadata(tmp_path / "LangChain" / "handwritten notes" / "n.txt", tmp_path)
        assert meta["course"] == "LangChain" and meta["notes_type"] == "handwritten notes"
        assert "specialization" not in meta  # Chroma can't store None

    def test_one_and_zero_levels(self, tmp_path):
        assert parse_source_metadata(tmp_path / "Course" / "n.txt", tmp_path)["course"] == "Course"
        assert parse_source_metadata(tmp_path / "n.txt", tmp_path)["course"] == "General"

    def test_file_outside_base_dir_falls_back(self, tmp_path):
        meta = parse_source_metadata("/somewhere/else/x.txt", tmp_path)
        assert meta["source"] == "x.txt"


class TestChunking:
    def test_chunks_carry_metadata_and_index(self, cfg):
        chunks = chunk_text("word " * 200, {"source": "a.txt", "course": "C"}, cfg)
        assert len(chunks) > 1
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert all(c.metadata["course"] == "C" for c in chunks)
        assert all(len(c.page_content) <= cfg["rag_core"]["chunking"]["chunk_size"] for c in chunks)

    def test_tiny_fragments_dropped(self, cfg):
        assert chunk_text("hi", {"source": "a"}, cfg) == []

    def test_ids_are_deterministic_and_unique(self, cfg):
        chunks = chunk_text("alpha beta gamma " * 60, {"source": "a.txt"}, cfg)
        ids = [chunk_id(c) for c in chunks]
        assert ids == [chunk_id(c) for c in chunks] and len(set(ids)) == len(ids)

    def test_bom_is_stripped(self, cfg, tmp_path):
        f = tmp_path / "Course" / "n.txt"
        f.parent.mkdir()
        f.write_bytes("﻿Hello world this is a sufficiently long note.".encode("utf-8"))
        assert not chunk_single_document(f, cfg, base_dir=tmp_path)[0].page_content.startswith("﻿")

    def test_paragraphs_are_preferred_split_points(self, cfg):
        text = ("First paragraph sentence. " * 5) + "\n\n" + ("Second paragraph sentence. " * 5)
        chunks = chunk_text(text, {"source": "a"}, cfg)
        assert chunks[0].page_content.endswith("sentence.") and chunks[1].page_content.startswith("Second")


class TestIndexer:
    def test_first_sync_indexes_everything(self, cfg, notes, fake_embeddings):
        report = sync_index(cfg, embeddings=fake_embeddings)
        assert report.added == 3 and report.unchanged == 0 and not report.failed
        assert store_size(get_vector_store(cfg, fake_embeddings)) == report.chunks_added > 0

    def test_second_sync_is_a_noop(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        again = sync_index(cfg, embeddings=fake_embeddings)
        assert (again.added, again.updated, again.removed, again.chunks_added) == (0, 0, 0, 0)
        assert again.unchanged == 3 and not again.changed

    def test_changed_file_is_replaced_not_duplicated(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        store = get_vector_store(cfg, fake_embeddings)
        before = store_size(store)
        notes("DL Spec/Optimization/handwritten notes/adam.txt", "Adam is an optimizer. " * 6)
        report = sync_index(cfg, embeddings=fake_embeddings)
        assert report.updated == 1 and report.unchanged == 2
        after = store.get(where={"source": "DL Spec/Optimization/handwritten notes/adam.txt"})
        assert all("optimizer" in d for d in after["documents"])  # old text is gone
        assert store_size(store) < before

    def test_new_and_removed_files(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        plan = plan_changes(cfg)
        assert plan.pending == 0
        notes("DL Spec/Optimization/lecture slides/sgd.txt", "Stochastic gradient descent uses minibatches. " * 5)
        Path(cfg["data"]["processed_path"], "DL Spec/Optimization/handwritten notes/adam.txt").unlink()
        plan = plan_changes(cfg)
        assert plan.new == ["DL Spec/Optimization/lecture slides/sgd.txt"]
        assert plan.removed == ["DL Spec/Optimization/handwritten notes/adam.txt"]
        report = sync_index(cfg, embeddings=fake_embeddings)
        assert (report.added, report.removed) == (1, 1)
        sources = {m["source"] for m in get_vector_store(cfg, fake_embeddings).get(include=["metadatas"])["metadatas"]}
        assert "DL Spec/Optimization/handwritten notes/adam.txt" not in sources

    def test_changing_chunk_size_forces_rebuild(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        cfg["rag_core"]["chunking"]["chunk_size"] = 120
        report = sync_index(cfg, embeddings=fake_embeddings)
        assert report.rebuilt and report.added == 3

    def test_rebuild_keeps_cosine_space(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        sync_index(cfg, rebuild=True, embeddings=fake_embeddings)
        assert collection_space(get_vector_store(cfg, fake_embeddings)) == "cosine"

    def test_progress_callback_reaches_total(self, cfg, notes, fake_embeddings):
        calls = []
        sync_index(cfg, embeddings=fake_embeddings, progress=lambda d, t, f: calls.append((d, t)))
        assert calls and calls[-1][0] == calls[-1][1]

    def test_library_table_reads_manifest(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        table = library_table(cfg)
        assert len(table) == 3 and set(table["course"]) == {"Sequence Models", "Optimization"}
        assert table["chunks"].sum() > 0 and (table["chars"] > 0).all()

    def test_discover_ignores_other_files(self, cfg, notes):
        Path(cfg["data"]["processed_path"], "x.pdf").write_bytes(b"%PDF")
        assert all(k.endswith((".txt", ".md")) for k in discover_files(cfg["data"]["processed_path"]))


class TestRetrieval:
    def _store(self, cfg, notes, fake_embeddings):
        sync_index(cfg, embeddings=fake_embeddings)
        return get_vector_store(cfg, fake_embeddings)

    def test_finds_relevant_passage_first(self, cfg, notes, fake_embeddings):
        store = self._store(cfg, notes, fake_embeddings)
        hits = retrieve(store, "update gate reset gate GRU", k=3)
        assert hits[0].title == "gru" and hits[0].score > hits[-1].score >= 0

    def test_course_and_type_filters(self, cfg, notes, fake_embeddings):
        store = self._store(cfg, notes, fake_embeddings)
        hits = retrieve(store, "gates", k=10, course="Sequence Models", notes_types=["lecture slides"])
        assert hits and {h.title for h in hits} == {"lstm"}
        assert retrieve(store, "gates", k=10, course="Nonexistent") == []

    def test_min_score_filters_weak_matches(self, cfg, notes, fake_embeddings):
        store = self._store(cfg, notes, fake_embeddings)
        assert retrieve(store, "zebra quantum pineapple", k=5, min_score=0.5) == []

    def test_max_per_source_limits_dominance(self, cfg, notes, fake_embeddings):
        store = self._store(cfg, notes, fake_embeddings)
        hits = retrieve(store, "gate", k=10, max_per_source=1)
        assert len({h.source for h in hits}) == len(hits)

    def test_empty_query(self, cfg, notes, fake_embeddings):
        assert retrieve(self._store(cfg, notes, fake_embeddings), "   ") == []

    def test_topic_chunks_are_in_reading_order(self, cfg, notes, fake_embeddings):
        store = self._store(cfg, notes, fake_embeddings)
        docs = get_topic_chunks(store, "Sequence Models", ["handwritten notes"])
        assert [d.metadata["chunk_index"] for d in docs] == sorted(d.metadata["chunk_index"] for d in docs)
        assert {d.metadata["title"] for d in docs} == {"gru"}


class TestHelpers:
    def test_build_filter_shapes(self):
        assert build_filter() is None
        assert build_filter("C") == {"course": "C"}
        assert build_filter(notes_types=["a", "b"]) == {"notes_type": {"$in": ["a", "b"]}}
        assert build_filter("C", ["a"]) == {"$and": [{"course": "C"}, {"notes_type": {"$in": ["a"]}}]}

    def test_distance_to_similarity(self):
        assert distance_to_similarity(0.25, "cosine") == 0.75
        assert distance_to_similarity(0.5, "l2") == 0.75  # squared L2 on unit vectors

    def test_sample_evenly(self):
        assert sample_evenly(list(range(10)), 5) == [0, 2, 4, 6, 8]
        assert sample_evenly([1, 2], 5) == [1, 2] and sample_evenly([1, 2], 0) == []
