"""Dashboard: library overview and practice progress.

Chart rules (from the dataviz method): one y-scale per chart, thin rounded bars,
one colour per measure, fixed colour per note type (never re-assigned by rank),
a legend whenever there are two or more series, and a data-table view under each
chart for accessibility.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.app import services, ui
from src.app.views.common import link_to
from src.memory.tracker import get_attempts, get_stats, initialize_database

SURFACE = {"light": "#ffffff", "dark": "#0e1117"}  # Streamlit's default page backgrounds


def _bar_chart(df: pd.DataFrame, label: str, value: str, value_title: str, color: str, fmt: str = "d"):
    """Horizontal bars, sorted high->low, one colour."""
    height = max(110, 26 * len(df))
    return (
        alt.Chart(df)
        .mark_bar(color=color, size=14, cornerRadiusEnd=4)
        .encode(
            y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280, ticks=False, domain=False)),
            x=alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(format=fmt, grid=True, tickMinStep=1 if fmt == "d" else None)),
            tooltip=[alt.Tooltip(f"{label}:N", title="Course"), alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)],
        )
        .properties(height=height)
    )


def _mix_chart(mix: pd.DataFrame):
    """One stacked bar: share of passages by note type; legend labels carry the numbers."""
    theme = ui.theme_type()
    total = mix["chunks"].sum()
    mix = mix.copy()
    mix["label"] = mix.apply(lambda r: f"{r['notes_type']} · {r['chunks']:,} ({r['chunks'] / total:.0%})", axis=1)
    domain = mix["label"].tolist()
    colors = [ui.notes_type_color(t) for t in mix["notes_type"]]
    return (
        alt.Chart(mix.assign(all="Passages"))
        .mark_bar(size=38, stroke=SURFACE[theme], strokeWidth=2)
        .encode(
            x=alt.X("chunks:Q", stack="normalize", title=None, axis=alt.Axis(format="%", grid=False)),
            y=alt.Y("all:N", title=None, axis=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(orient="bottom", title=None, columns=1, symbolType="square", labelLimit=420)),
            order=alt.Order("notes_type:N"),
            tooltip=[alt.Tooltip("notes_type:N", title="Type"), alt.Tooltip("chunks:Q", title="Passages", format=",d")],
        )
        .properties(height=110)
    )


def _library_section() -> None:
    lib = services.library()
    st.subheader("Your library")
    if lib.empty:
        st.info("Nothing is indexed yet. Head to the Library page to add notes.", icon=":material/inbox:")
        link_to("library", "Open Library", ":material/library_books:")
        return

    words = int(lib["chars"].sum() / 6)
    cols = st.columns(4)
    with cols[0]:
        ui.kpi("Documents", f"{len(lib):,}")
    with cols[1]:
        ui.kpi("Passages indexed", f"{int(lib['chunks'].sum()):,}")
    with cols[2]:
        ui.kpi("Courses", f"{lib['course'].nunique():,}")
    with cols[3]:
        ui.kpi("Words of notes", f"≈ {words / 1000:,.0f}k" if words else "—", "estimated from characters")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    by_course = lib.groupby("course", as_index=False)["chunks"].sum().sort_values("chunks", ascending=False)
    with left:
        st.markdown("**Passages per course**")
        st.altair_chart(_bar_chart(by_course, "course", "chunks", "Passages", ui.colors()["blue"]), width="stretch")
        with st.expander("View data"):
            st.dataframe(by_course.rename(columns={"course": "Course", "chunks": "Passages"}),
                         hide_index=True, width="stretch")
    with right:
        st.markdown("**Mix of note types**")
        # Anything that isn't one of the three standard kinds (e.g. a project folder) is folded into "other".
        folded = lib.assign(notes_type=lib["notes_type"].map(ui.known_notes_type))
        mix = folded.groupby("notes_type", as_index=False)["chunks"].sum().sort_values("notes_type")
        if not mix.empty:
            st.altair_chart(_mix_chart(mix), width="stretch")
            with st.expander("View data"):
                docs = folded.groupby("notes_type").size().rename("Documents")
                table = mix.set_index("notes_type").join(docs).rename(columns={"chunks": "Passages"})
                st.dataframe(table, width="stretch")


def _practice_section() -> None:
    config = services.get_config()
    initialize_database(config)
    stats = get_stats(config)
    st.subheader("Your practice")
    if stats["total"] == 0:
        st.info("No quiz answers yet. Take a quiz on the Study page and your progress will appear here.",
                icon=":material/quiz:")
        link_to("study", "Go to Study tools", ":material/school:")
        return

    cols = st.columns(4)
    with cols[0]:
        ui.kpi("Accuracy", f"{stats['accuracy']:.0%}", f"{stats['correct']} of {stats['total']} correct")
    with cols[1]:
        ui.kpi("Answers given", f"{stats['total']:,}")
    with cols[2]:
        ui.kpi("Day streak", f"{stats['streak_days']}", "days in a row with practice")
    with cols[3]:
        weak = stats["by_topic"][stats["by_topic"]["attempts"] > stats["by_topic"]["correct"]]
        ui.kpi("Topics to revisit", f"{len(weak)}", "have at least one mistake")

    st.write("")
    left, right = st.columns(2, gap="large")
    c = ui.colors()
    with left:
        st.markdown("**Accuracy by day**")
        daily = stats["daily"]
        line = (
            alt.Chart(daily)
            .mark_line(color=c["blue"], strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True, color=c["blue"]))
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", tickCount="day", grid=False)),
                y=alt.Y("accuracy:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("date:T", title="Day", format="%b %d"),
                         alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%"),
                         alt.Tooltip("attempts:Q", title="Answers")],
            )
            .properties(height=260)
        )
        st.altair_chart(line, width="stretch")
        with st.expander("View data"):
            st.dataframe(daily.assign(date=daily["date"].dt.date, accuracy=(daily["accuracy"] * 100).round(0)),
                         hide_index=True, width="stretch")
    with right:
        st.markdown("**Mistakes by topic**")
        topics = stats["by_topic"].assign(mistakes=lambda d: d["attempts"] - d["correct"])
        topics = topics[topics["mistakes"] > 0].sort_values("mistakes", ascending=False).head(8)
        if topics.empty:
            st.success("No mistakes so far - great work!", icon=":material/celebration:")
        else:
            st.altair_chart(_bar_chart(topics, "topic", "mistakes", "Mistakes", c["orange"]), width="stretch")
            with st.expander("View data"):
                st.dataframe(topics[["topic", "attempts", "correct", "mistakes"]], hide_index=True, width="stretch")

    attempts = get_attempts(config)
    recent = attempts[~attempts["correct"]].sort_values("id", ascending=False).head(8)
    if not recent.empty:
        st.markdown("**Recent mistakes**")
        table = recent[["timestamp", "topic", "question", "correct_answer"]].rename(
            columns={"timestamp": "When", "topic": "Topic", "question": "Question", "correct_answer": "Answer"}
        )
        st.dataframe(table, hide_index=True, width="stretch")
        link_to("study", "Practice these on the Study page", ":material/history_edu:")


def render() -> None:
    st.title("Dashboard")
    _library_section()
    st.divider()
    _practice_section()
