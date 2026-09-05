"""Render dashboard input controls in the Streamlit sidebar."""

from dataclasses import dataclass

import streamlit as st

from src.config import (
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_FORECAST_METHOD,
    DEFAULT_OPTIMIZER,
    MAX_FORECAST_HORIZON_DAYS,
    MIN_FORECAST_HORIZON_DAYS,
    MOVING_AVERAGE_METHOD,
    MOVING_AVERAGE_WINDOW_DAYS,
    SUPPORTED_OPTIMIZERS,
)
from src.dashboard.constants import (
    DASHBOARD_FORECAST_METHODS,
    DATA_SOURCE_DISPLAY_NAMES,
    DATA_SOURCE_SELECTION_KEY,
    FORECAST_METHOD_DISPLAY_NAMES,
    MAX_UPLOAD_SIZE_MB,
    OPTIMIZER_DISPLAY_NAMES,
    REQUIRED_UPLOAD_FILENAMES,
    SAMPLE_DATA_SOURCE,
    UPLOADED_DATA_SOURCE,
)


@dataclass(frozen=True)
class SidebarControls:
    """Store all values selected in the sidebar."""

    data_source: str
    uploaded_file_name: str | None
    uploaded_zip_bytes: bytes | None
    requested_horizon_days: int
    forecast_method: str
    optimizer_name: str
    moving_average_window_days: int
    run_requested: bool
    clear_requested: bool


def render_sidebar() -> SidebarControls:
    """Render data and optimization controls."""

    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Choose a dataset",
        options=(
            SAMPLE_DATA_SOURCE,
            UPLOADED_DATA_SOURCE,
        ),
        format_func=lambda source: (
            DATA_SOURCE_DISPLAY_NAMES[source]
        ),
        key=DATA_SOURCE_SELECTION_KEY,
    )

    uploaded_file_name = None
    uploaded_zip_bytes = None

    if data_source == UPLOADED_DATA_SOURCE:
        uploaded_file = st.sidebar.file_uploader(
            "Upload project dataset",
            type=("zip",),
            help=(
                "Upload one ZIP with exactly the seven "
                "required CSV files at its root."
            ),
        )

        st.sidebar.caption(
            "Required files: "
            + ", ".join(REQUIRED_UPLOAD_FILENAMES)
        )
        st.sidebar.caption(
            f"Maximum ZIP size: {MAX_UPLOAD_SIZE_MB} MB"
        )

        if uploaded_file is not None:
            uploaded_file_name = uploaded_file.name
            uploaded_zip_bytes = uploaded_file.getvalue()

    st.sidebar.divider()
    st.sidebar.header("Optimization Settings")

    requested_horizon_days = st.sidebar.slider(
        "Days to prepare inventory",
        min_value=MIN_FORECAST_HORIZON_DAYS,
        max_value=MAX_FORECAST_HORIZON_DAYS,
        value=DEFAULT_FORECAST_HORIZON_DAYS,
        step=1,
    )

    forecast_method = st.sidebar.selectbox(
        "Forecast method",
        options=DASHBOARD_FORECAST_METHODS,
        index=DASHBOARD_FORECAST_METHODS.index(
            DEFAULT_FORECAST_METHOD
        ),
        format_func=lambda method: (
            FORECAST_METHOD_DISPLAY_NAMES.get(
                method,
                method.replace("_", " ").title(),
            )
        ),
        help=(
            "Historical Average and Moving Average "
            "do not require training. Random Forest "
            "and AdaBoost require a compatible "
            "trained model artifact."
        ),
    )

    moving_average_window_days = (
        MOVING_AVERAGE_WINDOW_DAYS
    )

    if forecast_method == MOVING_AVERAGE_METHOD:
        moving_average_window_days = int(
            st.sidebar.number_input(
                "Moving-average window",
                min_value=1,
                max_value=90,
                value=MOVING_AVERAGE_WINDOW_DAYS,
                step=1,
            )
        )

    optimizer_name = st.sidebar.selectbox(
        "Optimization algorithm",
        options=SUPPORTED_OPTIMIZERS,
        index=SUPPORTED_OPTIMIZERS.index(
            DEFAULT_OPTIMIZER
        ),
        format_func=lambda optimizer: (
            OPTIMIZER_DISPLAY_NAMES.get(
                optimizer,
                optimizer.replace("_", " ").title(),
            )
        ),
    )

    st.sidebar.info(
        "A request of 1-7 days replenishes 7 days. "
        "A request of 8-14 days replenishes 14 days."
    )

    run_column, clear_column = (
        st.sidebar.columns(
            (3, 2),
            gap="small",
        )
    )

    with run_column:
        run_requested = st.button(
            "Run optimization",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    with clear_column:
        clear_requested = st.button(
            "Clear",
            type="tertiary",
            icon=":material/delete_outline:",
            help="Clear data and optimization results.",
            width="stretch",
        )

    return SidebarControls(
        data_source=data_source,
        uploaded_file_name=uploaded_file_name,
        uploaded_zip_bytes=uploaded_zip_bytes,
        requested_horizon_days=(
            requested_horizon_days
        ),
        forecast_method=forecast_method,
        optimizer_name=optimizer_name,
        moving_average_window_days=(
            moving_average_window_days
        ),
        run_requested=run_requested,
        clear_requested=clear_requested,
    )
