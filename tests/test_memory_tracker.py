import datetime
import sqlite3

from src.memory.tracker import (
    get_attempts, get_review_questions, get_stats, get_weak_topics, initialize_database, log_attempt, log_mistake,
)


def test_tracker_functions(cfg):
    """The original behaviour: log mistakes, get topics ranked by count."""
    cfg["memory"]["limit"] = 3
    initialize_database(cfg)
    log_mistake("Topic A", "Question about RNNs", cfg)
    log_mistake("Topic B", "Question about Vectors", cfg)
    log_mistake("Topic A", "Another question about RNNs", cfg)
    assert get_weak_topics(cfg) == [("Topic A", 2), ("Topic B", 1)]


def test_weak_topics_respects_limit(cfg):
    cfg["memory"]["limit"] = 1
    initialize_database(cfg)
    for topic in ("A", "A", "B"):
        log_mistake(topic, "q", cfg)
    assert get_weak_topics(cfg) == [("A", 2)]


def test_creates_missing_parent_directory(cfg, tmp_path):
    cfg["memory"]["sqlite_database_path"] = str(tmp_path / "does" / "not" / "exist" / "memory.db")
    initialize_database(cfg)
    log_mistake("T", "q", cfg)
    assert get_weak_topics(cfg) == [("T", 1)]


def test_existing_old_database_keeps_working(cfg):
    """A memory.db created by the first version (mistakes table only) is upgraded in place."""
    with sqlite3.connect(cfg["memory"]["sqlite_database_path"]) as conn:
        conn.execute("CREATE TABLE mistakes (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, "
                     "question TEXT NOT NULL, timestamp TEXT NOT NULL)")
        conn.execute("INSERT INTO mistakes (topic, question, timestamp) VALUES ('Old', 'q', '2025-09-01 10:00:00')")
    initialize_database(cfg)
    log_attempt("New", "q2", False, cfg, 0.2, "wrong", "right")
    assert dict(get_weak_topics(cfg)) == {"Old": 1, "New": 1}
    assert len(get_attempts(cfg)) == 1


def test_wrong_attempts_also_land_in_mistakes(cfg):
    initialize_database(cfg)
    log_attempt("T", "q1", True, cfg, 0.95)
    log_attempt("T", "q2", False, cfg, 0.3, "x", "y")
    assert get_weak_topics(cfg) == [("T", 1)]


def test_stats_accuracy_topics_and_daily(cfg):
    initialize_database(cfg)
    assert get_stats(cfg)["total"] == 0 and get_stats(cfg)["accuracy"] is None
    for correct in (True, True, False):
        log_attempt("A", f"q{correct}", correct, cfg, 0.9 if correct else 0.1, "u", "c")
    log_attempt("B", "qb", False, cfg, 0.0, "", "c")
    stats = get_stats(cfg)
    assert (stats["total"], stats["correct"]) == (4, 2) and stats["accuracy"] == 0.5
    by_topic = stats["by_topic"].set_index("topic")
    assert by_topic.loc["A", "attempts"] == 3 and by_topic.loc["B", "accuracy"] == 0
    assert len(stats["daily"]) == 1 and stats["streak_days"] == 1


def test_streak_counts_consecutive_days(cfg):
    initialize_database(cfg)
    today = datetime.date.today()
    with sqlite3.connect(cfg["memory"]["sqlite_database_path"]) as conn:
        for days_ago in (0, 1, 2, 4):  # gap on day 3
            ts = f"{today - datetime.timedelta(days=days_ago)} 12:00:00"
            conn.execute("INSERT INTO attempts (topic, question, correct, timestamp) VALUES ('T', 'q', 1, ?)", (ts,))
    assert get_stats(cfg)["streak_days"] == 3


def test_review_questions_are_latest_wrong_with_answers(cfg):
    initialize_database(cfg)
    log_attempt("T", "solved later", False, cfg, 0.1, "a", "ref1")
    log_attempt("T", "solved later", True, cfg, 0.9, "a", "ref1")      # now correct -> not reviewed
    log_attempt("T", "still wrong", False, cfg, 0.1, "a", "ref2")
    log_attempt("T", "no reference", False, cfg, 0.1, "a", "")          # can't be re-quizzed
    assert get_review_questions(cfg) == [{"topic": "T", "question": "still wrong", "answer": "ref2"}]
