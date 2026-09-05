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
    filter_state = (
        "result"
        if result is not None
        else "preview"
    )
    status_filter_key = (
        "map_store_status_"
        f"{filter_state}_filter"
    )
    route_filter_key = (
        "map_route_type_"
        f"{filter_state}_filter"
    )
    source_filter_key = (
        "map_source_city_"
        f"{filter_state}_filter"
    )
    destination_filter_key = (
        "map_destination_city_"
        f"{filter_state}_filter"
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

        current_city_selection = (
            st.session_state.get(
                "map_store_city_filter"
            )
        )

        if current_city_selection is not None:
            valid_city_selection = [
                city
                for city in current_city_selection
                if city in available_cities
            ]

            if (
                current_city_selection
                and not valid_city_selection
            ):
                valid_city_selection = available_cities

            if (
                current_city_selection
                != valid_city_selection
            ):
                st.session_state[
                    "map_store_city_filter"
                ] = valid_city_selection

        selected_store_cities = (
            filter_columns[0].multiselect(
                "Store City",
                options=available_cities,
                default=available_cities,
                key="map_store_city_filter",
            )
        )

        city_store_summary = store_summary.loc[
            store_summary["city"].isin(
                selected_store_cities
            )
        ]
        available_statuses = [
            status
            for status in STATUS_DISPLAY_NAMES
            if status
            in set(city_store_summary["map_status"])
        ]

        current_status_selection = (
            st.session_state.get(
                status_filter_key
            )
        )

        if current_status_selection is not None:
            valid_status_selection = [
                status
                for status
                in current_status_selection
                if status in available_statuses
            ]

            if (
                current_status_selection
                and not valid_status_selection
            ):
                valid_status_selection = (
                    available_statuses
                )

            if (
                current_status_selection
                != valid_status_selection
            ):
                st.session_state[
                    status_filter_key
                ] = valid_status_selection

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
                key=status_filter_key,
            )
        )

        visible_summary = city_store_summary.loc[
            city_store_summary["map_status"].isin(
                selected_statuses
            )
        ].copy()
        visible_store_ids = set(
            visible_summary["store_id"]
        )

        if transfer_data.empty:
            candidate_routes = transfer_data.copy()
        else:
            candidate_routes = transfer_data.loc[
                transfer_data["from_store_id"].isin(
                    visible_store_ids
                )
                & transfer_data["to_store_id"].isin(
                    visible_store_ids
                )
            ].copy()

        available_route_types = [
            route_type
            for route_type in ROUTE_DISPLAY_NAMES
            if (
                not candidate_routes.empty
                and route_type
                in set(candidate_routes["route_type"])
            )
        ]

        current_route_selection = (
            st.session_state.get(
                route_filter_key
            )
        )

        if current_route_selection is not None:
            valid_route_selection = [
                route_type
                for route_type
                in current_route_selection
                if route_type in available_route_types
            ]

            if (
                current_route_selection
                and not valid_route_selection
            ):
                valid_route_selection = (
                    available_route_types
                )

            if (
                current_route_selection
                != valid_route_selection
            ):
                st.session_state[
                    route_filter_key
                ] = valid_route_selection

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
                key=route_filter_key,
                disabled=transfer_data.empty,
            )
        )

        route_filtered_data = candidate_routes

        if not route_filtered_data.empty:
            route_filtered_data = (
                route_filtered_data.loc[
                    route_filtered_data[
                        "route_type"
                    ].isin(selected_route_types)
                ]
            )

        available_source_cities = (
            sorted(
                route_filtered_data[
                    "from_city"
                ].unique()
            )
            if not route_filtered_data.empty
            else []
        )

        current_source_selection = (
            st.session_state.get(
                source_filter_key,
                [],
            )
        )
        valid_source_selection = [
            city
            for city in current_source_selection
            if city in available_source_cities
        ]

        if (
            current_source_selection
            != valid_source_selection
        ):
            st.session_state[
                source_filter_key
            ] = valid_source_selection

        selected_source_cities = (
            filter_columns[3].multiselect(
                "Source City",
                options=available_source_cities,
                key=source_filter_key,
                placeholder="All matching cities",
                disabled=transfer_data.empty,
            )
        )

        source_filtered_data = route_filtered_data

        if selected_source_cities:
            source_filtered_data = (
                source_filtered_data.loc[
                    source_filtered_data[
                        "from_city"
                    ].isin(selected_source_cities)
                ]
            )

        available_destination_cities = (
            sorted(
                source_filtered_data[
                    "to_city"
                ].unique()
            )
            if not source_filtered_data.empty
            else []
        )

        current_destination_selection = (
            st.session_state.get(
                destination_filter_key,
                [],
            )
        )
        valid_destination_selection = [
            city
            for city
            in current_destination_selection
            if city in available_destination_cities
        ]

        if (
            current_destination_selection
            != valid_destination_selection
        ):
            st.session_state[
                destination_filter_key
            ] = valid_destination_selection

        selected_destination_cities = (
            filter_columns[4].multiselect(
                "Destination City",
                options=available_destination_cities,
                key=destination_filter_key,
                placeholder="All matching cities",
                disabled=transfer_data.empty,
            )
        )

    if visible_summary.empty:
        st.warning(
            "No stores match the selected map filters."
        )
        return

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
