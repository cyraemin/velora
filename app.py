"""Velora — AI voice emotion recognition, acoustic analysis and expressive transcription.

Run with:  streamlit run app.py
"""
import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")        # hides "unauthenticated requests" notice
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("fontTools").setLevel(logging.ERROR)    # PDF font-subsetting chatter


class _DropClientResets(logging.Filter):
    """Windows asyncio logs a traceback when a browser tab closes abruptly; it is harmless."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, (ConnectionResetError, ConnectionAbortedError))


logging.getLogger("asyncio").addFilter(_DropClientResets())

import streamlit as st  # noqa: E402

from velora import ui  # noqa: E402

st.set_page_config(page_title="Velora", page_icon="assets/icon.svg", layout="wide",
                   initial_sidebar_state="expanded")
ui.apply_theme()
st.logo("assets/logo.svg", size="large", icon_image="assets/icon.svg")

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/live_studio.py", title="Live Studio", icon=":material/mic:"),
    st.Page("views/analysis.py", title="Analysis", icon=":material/insights:"),
    st.Page("views/history.py", title="History", icon=":material/history:"),
    st.Page("views/contrast_lab.py", title="Contrast Lab", icon=":material/compare_arrows:"),
    st.Page("views/about.py", title="About", icon=":material/info:"),
]
page = st.navigation(pages)
ui.sidebar_settings()
page.run()
