from pathlib import Path

from src.config import deep_merge, load_config, resolve_path
from src.preprocessing.document_parser import extract_pdf, extract_text_from_pdf, process_all_documents
from src.preprocessing.text_cleaner import cleaning_fn


class TestCleaner:
    def test_collapses_spaces_and_tabs(self):
        assert cleaning_fn("This   string \t has   too   many   spaces.") == "This string has too many spaces."

    def test_keeps_paragraph_structure(self):
        assert cleaning_fn("Line one\nLine two\n\n\n\nNew paragraph") == "Line one\nLine two\n\nNew paragraph"

    def test_strips_edges_and_bom(self):
        assert cleaning_fn("﻿  \n  hello world \n ") == "hello world"

    def test_dehyphenates_across_line_breaks(self):
        assert cleaning_fn("classi-\nfication of data") == "classification of data"

    def test_does_not_join_before_capitals_or_bullets(self):
        assert cleaning_fn("Jean-\nPaul") == "Jean-\nPaul"

    def test_windows_line_endings(self):
        assert cleaning_fn("a\r\nb\r\n\r\nc") == "a\nb\n\nc"

    def test_clean_text_unchanged(self):
        assert cleaning_fn("This is a clean string.") == "This is a clean string."


def _make_pdf(path: Path, pages: list[str]) -> None:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestParser:
    def test_extracts_typed_pdf_text(self, tmp_path):
        pdf = tmp_path / "typed.pdf"
        _make_pdf(pdf, ["Gradient descent minimises the cost function.", "Second page about GRUs."])
        text = extract_text_from_pdf(pdf)
        assert "Gradient descent" in text and "GRUs" in text

    def test_unreadable_pdf_returns_empty_string(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        assert extract_text_from_pdf(bad) == ""

    def test_blank_page_is_reported_when_ocr_off(self, tmp_path, cfg):
        pdf = tmp_path / "mixed.pdf"
        _make_pdf(pdf, ["Plenty of selectable text on this first page.", ""])
        cfg["ocr"]["enabled"] = False
        text, stats = extract_pdf(pdf, cfg)
        assert stats == {"pages": 2, "ocr_pages": 0, "blank_pages": 1}
        assert "selectable text" in text

    def test_process_all_documents_mirrors_structure_and_skips_done(self, cfg):
        raw = Path(cfg["data"]["raw_path"])
        (raw / "Spec" / "Course" / "lecture slides").mkdir(parents=True)
        _make_pdf(raw / "Spec" / "Course" / "lecture slides" / "w1.pdf", ["Backpropagation computes gradients."])
        (raw / "Spec" / "Course" / "handwritten notes").mkdir(parents=True)
        (raw / "Spec" / "Course" / "handwritten notes" / "n.txt").write_text("plain   notes\n\n\n\nhere", encoding="utf-8")

        first = process_all_documents(cfg)
        out = Path(cfg["data"]["processed_path"]) / "Spec" / "Course"
        assert first["processed"] == 2 and first["skipped"] == 0 and not first["failed"]
        assert "Backpropagation" in (out / "lecture slides" / "w1.txt").read_text(encoding="utf-8")
        assert (out / "handwritten notes" / "n.txt").read_text(encoding="utf-8") == "plain notes\n\nhere"

        (out / "handwritten notes" / "n.txt").write_text("MY MANUAL EDIT", encoding="utf-8")
        second = process_all_documents(cfg)
        assert second["skipped"] == 2 and second["processed"] == 0
        assert (out / "handwritten notes" / "n.txt").read_text(encoding="utf-8") == "MY MANUAL EDIT"  # never clobbered

    def test_missing_raw_folder_is_not_an_error(self, cfg):
        assert process_all_documents(cfg)["processed"] == 0


class TestConfig:
    def test_deep_merge_overrides_nested_keys_only(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}, "b": 1}

    def test_relative_paths_resolve_against_project_root(self):
        cfg = load_config()
        assert Path(cfg["data"]["raw_path"]).is_absolute()
        assert Path(cfg["rag_core"]["database"]["persist_directory"]).is_absolute()

    def test_data_dir_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STUDY_ASSISTANT_DATA_DIR", str(tmp_path / "drive"))
        monkeypatch.setenv("STUDY_ASSISTANT_INDEX_DIR", str(tmp_path / "index"))
        cfg = load_config()
        assert cfg["data"]["raw_path"] == str(tmp_path / "drive" / "raw")
        assert cfg["memory"]["sqlite_database_path"] == str(tmp_path / "drive" / "memory.db")
        assert cfg["rag_core"]["database"]["persist_directory"] == str(tmp_path / "index")

    def test_provider_env_override(self, monkeypatch):
        monkeypatch.setenv("STUDY_ASSISTANT_LLM_PROVIDER", "none")
        assert load_config()["llm"]["provider"] == "none"

    def test_absolute_paths_untouched(self):
        assert resolve_path("/abs/path", data_dir="/elsewhere") == "/abs/path"
