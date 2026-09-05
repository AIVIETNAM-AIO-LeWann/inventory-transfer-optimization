"""Render the primary dashboard navigation."""

import streamlit as st

from src.dashboard.constants import (
    DASHBOARD_PAGES,
    DASHBOARD_PAGE_DISPLAY_NAMES,
    DASHBOARD_PAGE_SELECTION_KEY,
    OVERVIEW_PAGE,
)


def format_dashboard_page(
    page: str,
) -> str:
    """Return the display label for a dashboard page."""

    return DASHBOARD_PAGE_DISPLAY_NAMES.get(
        page,
        page.replace("_", " ").title(),
    )


def render_dashboard_navigation() -> str:
    """Render navigation and return the selected page."""

    selected_page = st.segmented_control(
        "Dashboard navigation",
        options=DASHBOARD_PAGES,
        selection_mode="single",
        default=OVERVIEW_PAGE,
        required=True,
        format_func=format_dashboard_page,
        key=DASHBOARD_PAGE_SELECTION_KEY,
        label_visibility="collapsed",
        width="stretch",
    )

    if selected_page is None:
        return OVERVIEW_PAGE

    return str(selected_page)
