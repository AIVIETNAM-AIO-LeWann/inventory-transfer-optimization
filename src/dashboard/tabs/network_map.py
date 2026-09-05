"""Render the interactive store and route map tab."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.dashboard.components import (
    render_section_heading,
)
from src.dashboard.constants import (
    DEFAULT_MAX_DISPLAYED_ROUTES,
    ROUTE_DISPLAY_NAMES,
    STATUS_DISPLAY_NAMES,
)
from src.dashboard.data_views import (
    enrich_transfer_data,
)
from src.dashboard.formatters import (
    format_integer,
)
from src.dashboard.map_builder import (
    build_network_map,
    create_store_map_summary,
)
from src.data_loader import ProjectData
from src.inventory_analyzer import SHORTAGE_STATUS
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def filter_map_routes(
    transfer_data: pd.DataFrame,
    selected_route_types: list[str],
    selected_source_cities: list[str],
    selected_destination_cities: list[str],
    visible_store_ids: set[str],
) -> pd.DataFrame:
    """Filter routes displayed on the network map."""

    if transfer_data.empty:
        return transfer_data.copy()

    filtered_data = transfer_data.loc[
        transfer_data["route_type"].isin(
            selected_route_types
        )
        & transfer_data["from_store_id"].isin(
            visible_store_ids
        )
        & transfer_data["to_store_id"].isin(
            visible_store_ids
        )
    ].copy()

    city_filters = (
        ("from_city", selected_source_cities),
        ("to_city", selected_destination_cities),
    )

    for column, selected_values in city_filters:
        if selected_values:
            filtered_data = filtered_data.loc[
                filtered_data[column].isin(
                    selected_values
                )
            ]

    return filtered_data


def render_network_map_tab(
    project_data: ProjectData,
    result: OptimizationPipelineResult | None,
) -> None:
    """Render map filters, KPIs, markers, and routes."""

    st.subheader("Store Network Map")
    st.caption(
        "Explore store inventory health and planned "
        "transfers across the distribution network."
    )

    inventory_analysis = None
    transfer_plan = None
    transfer_data = pd.DataFrame()

    if result is not None:
        inventory_analysis = result.inventory_analysis
        transfer_plan = result.transfer_plan
        transfer_data = enrich_transfer_data(
            transfer_plan=transfer_plan,
            stores=project_data.stores,
            products=project_data.products,
        )

    store_summary = create_store_map_summary(
        stores=project_data.stores,
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
    )

    available_cities = sorted(
        store_summary["city"].unique()
    )
    available_statuses = [
        status
        for status in STATUS_DISPLAY_NAMES
        if status in set(store_summary["map_status"])
    ]
    available_route_types = [
        route_type
        for route_type in ROUTE_DISPLAY_NAMES
        if (
            not transfer_data.empty
            and route_type
            in set(transfer_data["route_type"])
        )
    ]
    filter_state = (
        "result"
        if result is not None
        else "preview"
    )

    render_section_heading(
        title="Map Filters",
        description=(
            "Focus the map on selected store statuses "
            "and transfer geographies."
        ),
    )

    with st.container(border=True):
        filter_columns = st.columns(5)

        selected_store_cities = (
            filter_columns[0].multiselect(
                "Store City",
                options=available_cities,
                default=available_cities,
                key="map_store_city_filter",
            )
        )
        selected_statuses = (
            filter_columns[1].multiselect(
                "Store Status",
                options=available_statuses,
                default=available_statuses,
                format_func=lambda status: (
                    STATUS_DISPLAY_NAMES.get(
                        status,
                        status.title(),
                    )
                ),
                key=(
                    "map_store_status_"
                    f"{filter_state}_filter"
                ),
            )
        )
        selected_route_types = (
            filter_columns[2].multiselect(
                "Route Type",
                options=available_route_types,
                default=available_route_types,
                format_func=lambda route_type: (
                    ROUTE_DISPLAY_NAMES.get(
                        route_type,
                        route_type.replace(
                            "_",
                            " ",
                        ).title(),
                    )
                ),
                key=(
                    "map_route_type_"
                    f"{filter_state}_filter"
                ),
                disabled=transfer_data.empty,
            )
        )
        selected_source_cities = (
            filter_columns[3].multiselect(
                "Source City",
                options=(
                    sorted(
                        transfer_data[
                            "from_city"
                        ].unique()
                    )
                    if not transfer_data.empty
                    else []
                ),
                key=(
                    "map_source_city_"
                    f"{filter_state}_filter"
                ),
                placeholder="All cities",
                disabled=transfer_data.empty,
            )
        )
        selected_destination_cities = (
            filter_columns[4].multiselect(
                "Destination City",
                options=(
                    sorted(
                        transfer_data[
                            "to_city"
                        ].unique()
                    )
                    if not transfer_data.empty
                    else []
                ),
                key=(
                    "map_destination_city_"
                    f"{filter_state}_filter"
                ),
                placeholder="All cities",
                disabled=transfer_data.empty,
            )
        )

    visible_summary = store_summary.loc[
        store_summary["city"].isin(
            selected_store_cities
        )
        & store_summary["map_status"].isin(
            selected_statuses
        )
    ].copy()

    if visible_summary.empty:
        st.warning(
            "No stores match the selected map filters."
        )
        return

    visible_store_ids = set(
        visible_summary["store_id"]
    )
    visible_stores = project_data.stores.loc[
        project_data.stores["store_id"].isin(
            visible_store_ids
        )
    ].copy()

    filtered_routes = filter_map_routes(
        transfer_data=transfer_data,
        selected_route_types=selected_route_types,
        selected_source_cities=(
            selected_source_cities
        ),
        selected_destination_cities=(
            selected_destination_cities
        ),
        visible_store_ids=visible_store_ids,
    )

    maximum_available_routes = len(filtered_routes)
    max_routes = st.slider(
        "Maximum Routes to Display",
        min_value=0,
        max_value=max(
            maximum_available_routes,
            1,
        ),
        value=min(
            maximum_available_routes,
            DEFAULT_MAX_DISPLAYED_ROUTES,
        ),
        step=1,
        disabled=maximum_available_routes == 0,
        help=(
            "Routes with the largest transfer quantities "
            "are displayed first."
        ),
    )

    if filtered_routes.empty:
        displayed_routes = filtered_routes.copy()
    else:
        displayed_routes = (
            filtered_routes.sort_values(
                "quantity",
                ascending=False,
            ).head(max_routes)
        )

    render_section_heading(
        title="Network Summary",
        description=(
            "The KPIs below reflect the stores and routes "
            "currently visible on the map."
        ),
    )

    summary_metrics = st.columns(4)
    summary_metrics[0].metric(
        "Visible Stores",
        format_integer(len(visible_summary)),
    )
    summary_metrics[1].metric(
        "Shortage Stores",
        format_integer(
            (
                visible_summary["map_status"]
                == SHORTAGE_STATUS
            ).sum()
        ),
    )
    summary_metrics[2].metric(
        "Displayed Routes",
        format_integer(len(displayed_routes)),
    )
    summary_metrics[3].metric(
        "Displayed Units",
        format_integer(
            displayed_routes["quantity"].sum()
            if not displayed_routes.empty
            else 0
        ),
    )

    network_map = build_network_map(
        stores=visible_stores,
        inventory_analysis=inventory_analysis,
        transfer_plan=filtered_routes,
        max_routes=max_routes,
    )

    st_folium(
        network_map,
        height=680,
        use_container_width=True,
        returned_objects=[],
    )

    if result is None:
        st.info(
            "Run optimization to color stores by inventory "
            "status and display transfer routes."
        )
    elif filtered_routes.empty:
        st.info(
            "No transfer routes match the current map "
            "filters. Store markers remain visible."
        )
