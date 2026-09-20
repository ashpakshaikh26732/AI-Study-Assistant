"""SQLite memory: what you got wrong, and how you're doing over time.

Two tables:
  * ``mistakes`` - the original one-row-per-wrong-answer log (unchanged, so an
    existing memory.db keeps working).
  * ``attempts`` - every quiz answer, right or wrong, with the similarity score
    and the reference answer. Powers the dashboard and "review my mistakes".
"""
from __future__ import annotations

import datetime
import os
import sqlite3
from contextlib import closing, contextmanager

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


@contextmanager
def _connect(config: dict):
    path = config["memory"]["sqlite_database_path"]
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        with conn:  # commit on success, roll back on error
            yield conn


def _now() -> str:
    return datetime.datetime.now().strftime(_TS_FORMAT)


def initialize_database(config: dict) -> None:
    """Create the tables if they don't exist (safe to call on every start)."""
    with _connect(config) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS mistakes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   topic TEXT NOT NULL,
                   question TEXT NOT NULL,
                   timestamp TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   topic TEXT NOT NULL,
                   question TEXT NOT NULL,
                   correct INTEGER NOT NULL,
                   similarity REAL,
                   user_answer TEXT,
                   correct_answer TEXT,
                   timestamp TEXT NOT NULL)"""
        )


def log_mistake(topic: str, question: str, config: dict) -> None:
    """Record a wrong quiz answer (original API)."""
    with _connect(config) as conn:
        conn.execute(
            "INSERT INTO mistakes (topic, question, timestamp) VALUES (?, ?, ?)", (topic, question, _now())
        )


def log_attempt(
    topic: str,
    question: str,
    correct: bool,
    config: dict,
    similarity: float = 0.0,
    user_answer: str = "",
    correct_answer: str = "",
) -> None:
    """Record one quiz answer. Wrong answers are also added to ``mistakes``."""
    with _connect(config) as conn:
        conn.execute(
            """INSERT INTO attempts
                   (topic, question, correct, similarity, user_answer, correct_answer, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (topic, question, int(correct), float(similarity), user_answer, correct_answer, _now()),
        )
        if not correct:
            conn.execute(
                "INSERT INTO mistakes (topic, question, timestamp) VALUES (?, ?, ?)", (topic, question, _now())
            )


def get_weak_topics(config: dict) -> list[tuple]:
    """Topics with the most logged mistakes: ``[(topic, count), ...]`` (top ``memory.limit``)."""
    with _connect(config) as conn:
        rows = conn.execute(
            """SELECT topic, COUNT(*) AS mistake_count FROM mistakes
               GROUP BY topic ORDER BY mistake_count DESC, topic ASC LIMIT ?""",
            (config["memory"]["limit"],),
        ).fetchall()
    return rows


def get_attempts(config: dict):
    """All quiz attempts as a DataFrame (empty if none)."""
    import pandas as pd

    cols = ["id", "topic", "question", "correct", "similarity", "user_answer", "correct_answer", "timestamp"]
    with _connect(config) as conn:
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM attempts ORDER BY id").fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["correct"] = df["correct"].astype(bool)
    return df


def get_review_questions(config: dict, limit: int = 10) -> list[dict]:
    """Distinct questions you last got wrong, newest first, with their reference answers.

    Only attempts that stored a reference answer can be re-quizzed.
    """
    with _connect(config) as conn:
        rows = conn.execute(
            """SELECT a.topic, a.question, a.correct_answer
               FROM attempts a
               JOIN (SELECT question, MAX(id) AS last_id FROM attempts GROUP BY question) latest
                    ON a.id = latest.last_id
               WHERE a.correct = 0 AND COALESCE(a.correct_answer, '') != ''
               ORDER BY a.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [{"topic": t, "question": q, "answer": a} for t, q, a in rows]


def get_stats(config: dict) -> dict:
    """Numbers for the dashboard.

    Returns a dict with ``total``, ``correct``, ``accuracy`` (0..1 or None),
    ``streak_days`` (consecutive days up to today with at least one attempt),
    ``by_topic`` (DataFrame: topic, attempts, correct, accuracy) and ``daily``
    (DataFrame: date, attempts, correct, accuracy).
    """
    import pandas as pd

    df = get_attempts(config)
    empty_topic = pd.DataFrame(columns=["topic", "attempts", "correct", "accuracy"])
    empty_daily = pd.DataFrame(columns=["date", "attempts", "correct", "accuracy"])
    if df.empty:
        return {
            "total": 0, "correct": 0, "accuracy": None, "streak_days": 0,
            "by_topic": empty_topic, "daily": empty_daily,
        }

    by_topic = (
        df.groupby("topic")["correct"].agg(attempts="count", correct="sum").reset_index()
    )
    by_topic["accuracy"] = by_topic["correct"] / by_topic["attempts"]
    by_topic = by_topic.sort_values(["accuracy", "attempts"], ascending=[True, False]).reset_index(drop=True)

    df["date"] = df["timestamp"].dt.normalize()
    daily = df.groupby("date")["correct"].agg(attempts="count", correct="sum").reset_index()
    daily["accuracy"] = daily["correct"] / daily["attempts"]

    days = set(daily["date"].dt.date)
    streak, day = 0, datetime.date.today()
    if day not in days:  # a streak isn't broken until a whole day is missed
        day -= datetime.timedelta(days=1)
    while day in days:
        streak += 1
        day -= datetime.timedelta(days=1)

    total, correct = len(df), int(df["correct"].sum())
    return {
        "total": total, "correct": correct, "accuracy": correct / total, "streak_days": streak,
        "by_topic": by_topic, "daily": daily,
    }
