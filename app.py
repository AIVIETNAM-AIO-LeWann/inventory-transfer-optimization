"""Streamlit application for inventory transfer optimization."""

import streamlit as st

from src.dashboard.constants import (
    APP_DESCRIPTION,
    APP_LAYOUT,
    APP_PAGE_ICON,
    APP_TITLE,
    DATA_MODELS_PAGE,
    FORECAST_PAGE,
    INVENTORY_PAGE,
    NETWORK_MAP_PAGE,
    OVERVIEW_PAGE,
    SAMPLE_DATA_SOURCE,
    SAMPLE_MODEL_DATASET_ID,
    TRANSFER_PAGE,
    UPLOADED_DATA_SOURCE,
)
from src.dashboard.navigation import (
    render_dashboard_navigation,
)
from src.dashboard.components import (
    render_app_header,
)
from src.dashboard.services import (
    create_project_data_fingerprint,
    load_sample_project_data,
    load_uploaded_project_data,
    run_pipeline,
)
from src.dashboard.sidebar import (
    SidebarControls,
    render_sidebar,
)
from src.dashboard.state import (
    activate_project_data,
    clear_dashboard_state,
    clear_pipeline_result,
    get_pipeline_result,
    pipeline_settings_match,
    store_pipeline_result,
)
from src.dashboard.styles import (
    load_dashboard_styles,
)
from src.dashboard.tabs.data_models import (
    render_data_models_tab,
)
from src.dashboard.tabs.forecast import (
    render_forecast_tab,
)
from src.dashboard.tabs.inventory import (
    render_inventory_tab,
)
from src.dashboard.tabs.network_map import (
    render_network_map_tab,
)
from src.dashboard.tabs.overview import (
    render_overview_tab,
)
from src.dashboard.tabs.transfers import (
    render_transfer_tab,
)
from src.data_loader import ProjectData


def load_selected_project_data(
    controls: SidebarControls,
) -> tuple[ProjectData, str] | None:
    """Load the selected data source and return its identity."""

    if controls.data_source == SAMPLE_DATA_SOURCE:
        project_data = load_sample_project_data()

        return (
            project_data,
            create_project_data_fingerprint(
                project_data
            ),
        )

    if controls.data_source == UPLOADED_DATA_SOURCE:
        if controls.uploaded_zip_bytes is None:
            return None

        project_data = load_uploaded_project_data(
            controls.uploaded_zip_bytes
        )

        return (
            project_data,
            create_project_data_fingerprint(
                project_data
            ),
        )

    raise ValueError(
        f"Unsupported data source: {controls.data_source}"
    )


def render_dashboard_page(
    project_data: ProjectData,
    controls: SidebarControls,
    selected_page: str,
) -> None:
    """Render only the selected dashboard page."""

    result = get_pipeline_result()

    if selected_page == OVERVIEW_PAGE:
        render_overview_tab(
            result=result
        )
        return

    if selected_page == FORECAST_PAGE:
        render_forecast_tab(
            result=result,
            project_data=project_data,
        )
        return

    if selected_page == INVENTORY_PAGE:
        render_inventory_tab(
            result=result,
            project_data=project_data,
        )
        return

    if selected_page == TRANSFER_PAGE:
        render_transfer_tab(
            result=result,
            project_data=project_data,
        )
        return

    if selected_page == NETWORK_MAP_PAGE:
        render_network_map_tab(
            project_data=project_data,
            result=result,
        )
        return

    if selected_page == DATA_MODELS_PAGE:
        render_data_models_tab(
            project_data=project_data,
            data_source=controls.data_source,
            uploaded_file_name=(
                controls.uploaded_file_name
            ),
        )
        return

    raise ValueError(
        f"Unsupported dashboard page: {selected_page}"
    )


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_PAGE_ICON,
        layout=APP_LAYOUT,
    )
    load_dashboard_styles()

    render_app_header(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
    )

    controls = render_sidebar()
    selected_page = (
        render_dashboard_navigation()
    )

    if controls.clear_requested:
        st.cache_data.clear()
        clear_dashboard_state()
        st.rerun()

    try:
        selected_data = load_selected_project_data(
            controls
        )
    except Exception as error:
        st.error(
            "The selected dataset could not be loaded."
        )
        st.exception(error)
        return

    if selected_data is None:
        st.info(
            "Upload one ZIP containing all seven required "
            "CSV files to continue."
        )
        return

    project_data, data_fingerprint = selected_data

    model_dataset_fingerprint = (
        SAMPLE_MODEL_DATASET_ID
        if controls.data_source == SAMPLE_DATA_SOURCE
        else data_fingerprint
    )

    activate_project_data(
        project_data=project_data,
        data_source=controls.data_source,
        data_fingerprint=data_fingerprint,
        uploaded_file_name=(
            controls.uploaded_file_name
        ),
    )

    current_pipeline_settings = {
        "requested_horizon_days": (
            controls.requested_horizon_days
        ),
        "forecast_method": (
            controls.forecast_method
        ),
        "optimizer_name": (
            controls.optimizer_name
        ),
        "moving_average_window_days": (
            controls.moving_average_window_days
        ),
        "dataset_fingerprint": (
            data_fingerprint
        ),
        "model_dataset_fingerprint": (
            model_dataset_fingerprint
        ),
    }

    existing_result = (
        get_pipeline_result()
    )

    settings_changed = (
        existing_result is not None
        and not pipeline_settings_match(
            current_pipeline_settings
        )
    )

    if settings_changed:
        clear_pipeline_result()

        if not controls.run_requested:
            st.info(
                "Optimization settings changed. "
                "Run optimization again to refresh "
                "the results."
            )

    if controls.run_requested:
        clear_pipeline_result()
        try:
            with st.spinner(
                "Forecasting demand and optimizing "
                "inventory transfers..."
            ):
                result = run_pipeline(
                    project_data=project_data,
                    requested_horizon_days=(
                        controls.requested_horizon_days
                    ),
                    forecast_method=(
                        controls.forecast_method
                    ),
                    optimizer_name=(
                        controls.optimizer_name
                    ),
                    moving_average_window_days=(
                        controls
                        .moving_average_window_days
                    ),
                    dataset_fingerprint=(
                        model_dataset_fingerprint
                    ),
                )

            store_pipeline_result(
                result=result,
                settings=(
                    current_pipeline_settings
                ),
            )

            st.success(
                "Forecasting and optimization "
                "completed."
            )

        except FileNotFoundError as error:
            st.error(
                "No compatible trained model was "
                "found for the active dataset."
            )

            st.info(
                "Open the Data & Models tab, train "
                "the selected model, and run "
                "optimization again."
            )

            st.caption(str(error))

        except Exception as error:
            st.error(
                "Optimization could not be completed."
            )

            st.exception(error)

    render_dashboard_page(
        project_data=project_data,
        controls=controls,
        selected_page=selected_page,
    )


if __name__ == "__main__":
    main()
