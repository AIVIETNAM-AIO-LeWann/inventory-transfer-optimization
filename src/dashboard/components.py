"""Reusable visual components for the dashboard."""

from html import escape
from pathlib import Path
from textwrap import dedent

import streamlit as st


DASHBOARD_HERO_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "images"
    / "dashboard_hero.png"
)


def render_app_header(
    title: str,
    description: str,
) -> None:
    """Render the centered application header."""

    safe_title = escape(title)
    safe_description = escape(description)

    st.markdown(
        dedent(
            f"""
            <header class="dashboard-header">
                <div class="dashboard-header__eyebrow">
                    SMART INVENTORY NETWORK
                </div>
                <h1 class="dashboard-header__title">
                    {safe_title}
                </h1>
                <p class="dashboard-header__description">
                    {safe_description}
                </p>
            </header>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_overview_empty_state() -> None:
    """Render the welcome state before optimization runs."""

    with st.container(border=True):
        text_column, image_column = st.columns(
            (6, 5),
            gap="large",
            vertical_alignment="center",
        )

        with text_column:
            st.markdown(
                dedent(
                    """
                    <section class="overview-empty-state__content">
                        <div class="overview-empty-state__eyebrow">
                            READY TO OPTIMIZE
                        </div>
                        <h2 class="overview-empty-state__title">
                            Balance inventory across your network.
                        </h2>
                        <p class="overview-empty-state__description">
                            Forecast demand, identify stock gaps,
                            and create a cost-efficient transfer plan
                            for every store.
                        </p>
                        <div class="overview-empty-state__steps">
                            <div class="overview-empty-state__step">
                                <span>1</span>
                                Choose data
                            </div>
                            <div class="overview-empty-state__step">
                                <span>2</span>
                                Select methods
                            </div>
                            <div class="overview-empty-state__step">
                                <span>3</span>
                                Run optimization
                            </div>
                        </div>
                    </section>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        with image_column:
            if DASHBOARD_HERO_PATH.exists():
                st.image(
                    DASHBOARD_HERO_PATH,
                    width="stretch",
                )
            else:
                st.info(
                    "The dashboard illustration is unavailable."
                )


def render_result_badges(
    dataset_name: str,
    forecast_method: str,
    optimizer_name: str,
    requested_horizon_days: int,
    replenishment_horizon_days: int,
) -> None:
    """Render a compact summary of the active result."""

    safe_dataset_name = escape(dataset_name)
    safe_forecast_method = escape(forecast_method)
    safe_optimizer_name = escape(optimizer_name)

    st.markdown(
        dedent(
            f"""
            <div class="result-summary">
                <div class="result-summary__badge">
                    <span>Dataset</span>
                    <strong>{safe_dataset_name}</strong>
                </div>
                <div class="result-summary__badge">
                    <span>Forecast</span>
                    <strong>{safe_forecast_method}</strong>
                </div>
                <div class="result-summary__badge">
                    <span>Optimizer</span>
                    <strong>{safe_optimizer_name}</strong>
                </div>
                <div class="result-summary__badge">
                    <span>Planning Window</span>
                    <strong>
                        {requested_horizon_days} requested
                        · {replenishment_horizon_days} replenished
                    </strong>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_section_heading(
    title: str,
    description: str,
) -> None:
    """Render a consistent dashboard section heading."""

    st.markdown(
        dedent(
            f"""
            <div class="dashboard-section-heading">
                <h3>{escape(title)}</h3>
                <p>{escape(description)}</p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_configuration_card(
    title: str,
    items: tuple[tuple[str, str], ...],
) -> None:
    """Render configuration values as a summary card."""

    item_markup = "".join(
        (
            '<div class="configuration-card__item">'
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(value)}</strong>"
            "</div>"
        )
        for label, value in items
    )

    st.markdown(
        (
            '<section class="configuration-card">'
            f"<h3>{escape(title)}</h3>"
            f"{item_markup}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_page_empty_state(
    icon: str,
    title: str,
    description: str,
) -> None:
    """Render a compact empty state for a dashboard page."""

    st.markdown(
        dedent(
            f"""
            <section class="page-empty-state">
                <div class="page-empty-state__icon">
                    {escape(icon)}
                </div>
                <h3>{escape(title)}</h3>
                <p>{escape(description)}</p>
            </section>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
