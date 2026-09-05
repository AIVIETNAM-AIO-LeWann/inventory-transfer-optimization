"""Load custom visual styles for the Streamlit dashboard."""

from pathlib import Path

import streamlit as st


STYLESHEET_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "styles.css"
)


def load_dashboard_styles(
    stylesheet_path: str | Path = STYLESHEET_PATH,
) -> None:
    """Read and inject the dashboard stylesheet."""

    path = Path(stylesheet_path)

    if not path.exists():
        return

    css = path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )
