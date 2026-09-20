"""AI Study Assistant - Streamlit entry point.

Run with:  python run_app.py     (or: streamlit run src/app/main.py)

Pages: Chat, Study tools, Dashboard, Library. Backends load lazily, so the app
opens instantly; the embedding model loads on the first search and the language
model on the first question that needs it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root, wherever it lives (laptop, Colab, Drive)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="AI Study Assistant", page_icon=":material/school:", layout="wide", initial_sidebar_state="expanded"
)

from src.app import services, ui  # noqa: E402
from src.app.views import chat, dashboard, library, study  # noqa: E402
from src.llm.model_loader import PROVIDERS, resolve_provider  # noqa: E402


def _init_state(config: dict) -> None:
    st.session_state.setdefault("provider", config["llm"]["provider"])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("voice_replies", False)


def _sidebar(config: dict, pages: dict) -> None:
    """Scope, settings and voice input; they live above every page."""
    with st.sidebar:
        st.divider()
        lib = services.library()
        courses = sorted(c for c in lib["course"].unique() if c) if not lib.empty else []
        types = sorted(t for t in lib["notes_type"].unique() if t) if not lib.empty else []

        st.markdown("**Scope**")
        course = st.selectbox("Course", ["All courses"] + courses, key="scope_course", label_visibility="collapsed")
        chosen_types = st.multiselect("Note types", types, key="scope_types", placeholder="All note types")

        with st.expander("Settings", icon=":material/tune:"):
            st.selectbox("Language model", PROVIDERS, key="provider",
                         help="auto: GPU if present, else local Ollama, else retrieval-only. "
                              "gemini sends retrieved passages to Google.")
            k = st.slider("Passages per answer", 2, 10, config["rag_core"]["retriever"]["k"])
            st.toggle("Read answers aloud", key="voice_replies", help="Uses gTTS - needs an internet connection.")
            if st.button("Clear chat", icon=":material/delete:", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        resolved = resolve_provider(config, st.session_state.provider)
        ui.pill(services.llm_label(st.session_state.provider), ok=resolved != "none")
        if resolved == "none":
            st.caption("No language model connected: chat shows the best-matching passages. "
                       "See Library → System to set one up.")

        if config["voice"]["enabled"]:
            _voice_input(config, pages)

    st.session_state.scope = {
        "course": None if course == "All courses" else course,
        "types": chosen_types,
        "k": k,
    }


def _voice_input(config: dict, pages: dict) -> None:
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        return  # optional dependency
    with st.expander("Ask by voice", icon=":material/mic:"):
        audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", just_once=True, format="wav", key="mic")
        if audio:
            from src.app.views.common import transcribe_cached

            with st.spinner("Transcribing…"):
                text = transcribe_cached(audio["bytes"])
            if text:
                st.session_state.pending_question = text
                st.switch_page(pages["chat"])
            else:
                st.caption("Couldn't make out any speech - try again.")


def main() -> None:
    config = services.get_config()
    _init_state(config)
    ui.inject_css()

    pages = {
        "chat": st.Page(chat.render, title="Chat", icon=":material/chat:", url_path="chat", default=True),
        "study": st.Page(study.render, title="Study tools", icon=":material/school:", url_path="study"),
        "dashboard": st.Page(dashboard.render, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
        "library": st.Page(library.render, title="Library", icon=":material/library_books:", url_path="library"),
    }
    st.session_state["_pages"] = pages
    current = st.navigation(list(pages.values()))
    _sidebar(config, pages)
    current.run()


main()
