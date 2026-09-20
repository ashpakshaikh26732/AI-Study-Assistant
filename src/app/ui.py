"""Shared UI pieces: styling, KPI tiles, source cards and chart helpers."""
from __future__ import annotations

import html

import streamlit as st

from src.rag_core.retriever import snippet

# Categorical palette validated with the dataviz skill's validate_palette.js
# (light + dark, all-pairs). Slots are assigned by entity, never by rank.
PALETTE = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "muted": "#898781"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "muted": "#898781"},
}
NOTES_TYPE_COLOR_KEY = {  # stable identity -> colour slot
    "handwritten notes": "blue",
    "lecture slides": "orange",
    "related research papers": "aqua",
}

CSS = """
<style>
/* Theme-neutral: every tint is derived from the current text colour, so it works in light and dark. */
.sa-kpi{border:1px solid color-mix(in srgb,currentColor 14%,transparent);border-radius:12px;
        padding:14px 16px;background:color-mix(in srgb,currentColor 4%,transparent);min-height:112px}
.sa-kpi .v{font-size:1.9rem;font-weight:650;line-height:1.15;letter-spacing:-.01em}
.sa-kpi .l{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.65;margin-bottom:2px}
.sa-kpi .s{font-size:.8rem;opacity:.6;margin-top:2px}
.sa-src{border-left:3px solid #2a78d6;padding:6px 12px;margin:8px 0;border-radius:0 8px 8px 0;
        background:color-mix(in srgb,currentColor 4%,transparent)}
.sa-src .t{font-weight:600;font-size:.92rem}
.sa-src .m{font-size:.78rem;opacity:.65}
.sa-src .b{font-size:.88rem;margin-top:4px;white-space:pre-wrap}
.sa-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;
         border:1px solid color-mix(in srgb,currentColor 20%,transparent)}
.sa-card{border:1px solid color-mix(in srgb,currentColor 16%,transparent);border-radius:14px;
         padding:22px 24px;background:color-mix(in srgb,currentColor 4%,transparent);margin:6px 0 14px}
.sa-card .q{font-size:1.25rem;font-weight:600;line-height:1.4}
.sa-card .a{font-size:1.05rem;line-height:1.5;margin-top:12px;padding-top:12px;
            border-top:1px dashed color-mix(in srgb,currentColor 25%,transparent)}
.sa-hero{font-size:1.05rem;opacity:.8;margin-top:-6px}
[data-testid="stSidebar"] .sa-pill{margin-top:2px}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def theme_type() -> str:
    """``light`` or ``dark`` (the viewer's active Streamlit theme)."""
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def colors() -> dict:
    return PALETTE.get(theme_type(), PALETTE["light"])


def kpi(label: str, value, sub: str = "") -> None:
    """A stat tile: big number, small label, optional sub-line."""
    sub_html = f'<div class="s">{html.escape(str(sub))}</div>' if sub else ""
    st.markdown(
        f'<div class="sa-kpi"><div class="l">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(str(value))}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, ok: bool | None = None) -> None:
    """A small status pill; ``ok`` adds a green/grey dot (always paired with text)."""
    dot = "" if ok is None else ("🟢 " if ok else "⚪ ")
    st.markdown(f'<span class="sa-pill">{dot}{html.escape(text)}</span>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    """Expandable, numbered source cards under an answer (matches [1], [2] citations)."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f'<div class="sa-src"><div class="t">[{i}] {html.escape(src["title"])}</div>'
                f'<div class="m">{html.escape(src["course"])} · {html.escape(src["notes_type"])} · '
                f'relevance {src["score"]:.0%}</div>'
                f'<div class="b">{html.escape(snippet(src["text"]))}</div></div>',
                unsafe_allow_html=True,
            )
            with st.popover("Full passage", icon=":material/article:"):
                st.write(src["text"])


def known_notes_type(notes_type: str) -> str:
    """The note type if it is one of the three standard kinds, else ``other``."""
    return notes_type if notes_type.lower() in NOTES_TYPE_COLOR_KEY else "other"


def notes_type_color(notes_type: str) -> str:
    key = NOTES_TYPE_COLOR_KEY.get(notes_type.lower())
    return colors()[key] if key else colors()["muted"]
