"""Render the transfer planner tab."""

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    create_transfer_quantity_chart,
    create_transport_cost_chart,
)
from src.dashboard.components import (
    render_page_empty_state,
    render_section_heading,
)
from src.dashboard.constants import (
    ROUTE_DISPLAY_NAMES,
)
from src.dashboard.data_views import (
    enrich_transfer_data,
)
from src.dashboard.formatters import (
    dataframe_to_csv_bytes,
    format_currency,
    format_decimal,
    format_integer,
)
from src.data_loader import ProjectData
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def filter_transfer_data(
    transfer_data: pd.DataFrame,
    selected_route_types: list[str],
    selected_source_cities: list[str],
    selected_destination_cities: list[str],
    selected_categories: list[str],
    selected_products: list[str],
) -> pd.DataFrame:
    """Apply dashboard filters to a transfer plan."""

    filtered_data = transfer_data.loc[
        transfer_data["route_type"].isin(
            selected_route_types
        )
    ].copy()

    filter_values = (
        ("from_city", selected_source_cities),
        ("to_city", selected_destination_cities),
        ("category", selected_categories),
        ("product_id", selected_products),
    )

    for column, selected_values in filter_values:
        if selected_values:
            filtered_data = filtered_data.loc[
                filtered_data[column].isin(
                    selected_values
                )
            ]

    return filtered_data


def render_transfer_tab(
    result: OptimizationPipelineResult | None,
    project_data: ProjectData,
) -> None:
    """Render transfer-plan charts, filters, table, and export."""

    st.subheader("Transfer Planner")

    if result is None:
        render_page_empty_state(
            icon="🚚",
            title="No transfer plan yet",
            description=(
                "Run optimization to match excess stock "
                "with stores that need replenishment."
            ),
        )
        return

    transfer_data = enrich_transfer_data(
        transfer_plan=result.transfer_plan,
        stores=project_data.stores,
        products=project_data.products,
    )

    if transfer_data.empty:
        render_page_empty_state(
            icon="✅",
            title="The inventory network is balanced",
            description=(
                "No inventory transfer is required for "
                "the selected planning horizon."
            ),
        )
        return

    product_names = (
        transfer_data[
            ["product_id", "product_name"]
        ]
        .drop_duplicates("product_id")
        .set_index("product_id")["product_name"]
        .to_dict()
    )

    available_route_types = [
        route_type
        for route_type in ROUTE_DISPLAY_NAMES
        if route_type in set(
            transfer_data["route_type"]
        )
    ]

    render_section_heading(
        title="Transfer Filters",
        description=(
            "Inspect transfer routes by geography "
            "and product."
        ),
    )

    with st.container(border=True):
        filter_columns = st.columns(5)

        selected_route_types = (
            filter_columns[0].multiselect(
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
                key="transfer_route_filter",
            )
        )

        route_filtered_data = transfer_data.loc[
            transfer_data["route_type"].isin(
                selected_route_types
            )
        ]
        available_source_cities = sorted(
            route_filtered_data[
                "from_city"
            ].unique()
        )

        current_source_selection = (
            st.session_state.get(
                "transfer_source_city_filter",
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
                "transfer_source_city_filter"
            ] = valid_source_selection

        selected_source_cities = (
            filter_columns[1].multiselect(
                "Source City",
                options=available_source_cities,
                key="transfer_source_city_filter",
                placeholder="All matching cities",
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

        available_destination_cities = sorted(
            source_filtered_data[
                "to_city"
            ].unique()
        )

        current_destination_selection = (
            st.session_state.get(
                "transfer_destination_city_filter",
                [],
            )
        )
        valid_destination_selection = [
            city
            for city in current_destination_selection
            if city in available_destination_cities
        ]

        if (
            current_destination_selection
            != valid_destination_selection
        ):
            st.session_state[
                "transfer_destination_city_filter"
            ] = valid_destination_selection

        selected_destination_cities = (
            filter_columns[2].multiselect(
                "Destination City",
                options=available_destination_cities,
                key=(
                    "transfer_destination_city_filter"
                ),
                placeholder="All matching cities",
            )
        )

        destination_filtered_data = (
            source_filtered_data
        )

        if selected_destination_cities:
            destination_filtered_data = (
                destination_filtered_data.loc[
                    destination_filtered_data[
                        "to_city"
                    ].isin(
                        selected_destination_cities
                    )
                ]
            )

        available_categories = sorted(
            destination_filtered_data[
                "category"
            ].unique()
        )

        current_category_selection = (
            st.session_state.get(
                "transfer_category_filter",
                [],
            )
        )
        valid_category_selection = [
            category
            for category in current_category_selection
            if category in available_categories
        ]

        if (
            current_category_selection
            != valid_category_selection
        ):
            st.session_state[
                "transfer_category_filter"
            ] = valid_category_selection

        selected_categories = (
            filter_columns[3].multiselect(
                "Category",
                options=available_categories,
                key="transfer_category_filter",
                placeholder="All matching categories",
            )
        )

        category_filtered_data = (
            destination_filtered_data
        )

        if selected_categories:
            category_filtered_data = (
                category_filtered_data.loc[
                    category_filtered_data[
                        "category"
                    ].isin(selected_categories)
                ]
            )

        available_product_ids = sorted(
            category_filtered_data[
                "product_id"
            ].unique()
        )

        current_product_selection = (
            st.session_state.get(
                "transfer_product_filter",
                [],
            )
        )
        valid_product_selection = [
            product_id
            for product_id
            in current_product_selection
            if product_id in available_product_ids
        ]

        if (
            current_product_selection
            != valid_product_selection
        ):
            st.session_state[
                "transfer_product_filter"
            ] = valid_product_selection

        selected_products = (
            filter_columns[4].multiselect(
                "Product",
                options=available_product_ids,
                format_func=lambda product_id: (
                    f"{product_id} · "
                    f"{product_names.get(product_id, product_id)}"
                ),
                key="transfer_product_filter",
                placeholder="All matching products",
            )
        )

    filtered_transfers = filter_transfer_data(
        transfer_data=transfer_data,
        selected_route_types=selected_route_types,
        selected_source_cities=(
            selected_source_cities
        ),
        selected_destination_cities=(
            selected_destination_cities
        ),
        selected_categories=selected_categories,
        selected_products=selected_products,
    )

    if filtered_transfers.empty:
        st.warning(
            "No transfers match the selected filters."
        )
        return

    total_units = filtered_transfers[
        "quantity"
    ].sum()
    average_distance = (
        (
            filtered_transfers["distance_km"]
            * filtered_transfers["quantity"]
        ).sum()
        / total_units
    )
    average_lead_time = (
        (
            filtered_transfers["lead_time_minutes"]
            * filtered_transfers["quantity"]
        ).sum()
        / total_units
    )

    render_section_heading(
        title="Transfer Summary",
        description=(
            "Volume, cost, and delivery performance "
            "for the selected routes."
        ),
    )

    summary_metrics = st.columns(4)

    summary_metrics[0].metric(
        "🚚 Planned Transfers",
        format_integer(len(filtered_transfers)),
    )
    summary_metrics[1].metric(
        "📦 Transferred Units",
        format_integer(total_units),
    )
    summary_metrics[2].metric(
        "💰 Transport Cost",
        format_currency(
            filtered_transfers[
                "total_transport_cost"
            ].sum()
        ),
    )
    summary_metrics[3].metric(
        "⏱️ Average Lead Time",
        (
            format_decimal(average_lead_time)
            + " minutes"
        ),
        delta=(
            format_decimal(average_distance)
            + " km average"
        ),
        delta_color="off",
    )

    chart_columns = st.columns(
        2,
        gap="large",
    )

    with chart_columns[0]:
        render_section_heading(
            title="Transferred Units by Route",
            description=(
                "Share of units assigned to each "
                "route type."
            ),
        )

        st.plotly_chart(
            create_transfer_quantity_chart(
                filtered_transfers
            ),
            width="stretch",
        )

    with chart_columns[1]:
        render_section_heading(
            title="Transport Cost by Route",
            description=(
                "Total transportation cost for each "
                "route type."
            ),
        )

        st.plotly_chart(
            create_transport_cost_chart(
                filtered_transfers
            ),
            width="stretch",
        )

    render_section_heading(
        title="Priority Transfers",
        description=(
            "Largest planned movements in the "
            "current transfer scope."
        ),
    )

    route_labels = {
        route_type: (
            "🏙️ "
            if display_name == "Intra-city"
            else "🌐 "
        )
        + display_name
        for route_type, display_name
        in ROUTE_DISPLAY_NAMES.items()
    }

    priority_transfers = (
        filtered_transfers.nlargest(
            10,
            "quantity",
        )
        .copy()
    )
    priority_transfers["route_type"] = (
        priority_transfers["route_type"].map(
            route_labels
        )
    )

    st.dataframe(
        priority_transfers[
            [
                "from_store_name",
                "from_city",
                "to_store_name",
                "to_city",
                "product_name",
                "quantity",
                "route_type",
                "distance_km",
                "lead_time_minutes",
                "total_transport_cost",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "from_store_name": "Source Store",
            "from_city": "Source City",
            "to_store_name": "Destination Store",
            "to_city": "Destination City",
            "product_name": "Product",
            "quantity": st.column_config.NumberColumn(
                "Quantity",
                format="%d",
            ),
            "route_type": "Route Type",
            "distance_km": st.column_config.NumberColumn(
                "Distance",
                format="%.2f km",
            ),
            "lead_time_minutes": (
                st.column_config.NumberColumn(
                    "Lead Time",
                    format="%.2f min",
                )
            ),
            "total_transport_cost": (
                st.column_config.NumberColumn(
                    "Transport Cost",
                    format="%,.0f VND",
                )
            ),
        },
    )

    display_transfers = filtered_transfers.copy()
    display_transfers["route_type"] = (
        display_transfers["route_type"].map(
            route_labels
        )
    )

    with st.expander(
        f"All Transfer Records "
        f"({len(display_transfers):,})"
    ):
        display_columns = (
            "transfer_id",
            "product_id",
            "product_name",
            "category",
            "from_store_id",
            "from_store_name",
            "from_city",
            "to_store_id",
            "to_store_name",
            "to_city",
            "quantity",
            "route_type",
            "distance_km",
            "lead_time_minutes",
            "transport_cost_per_unit",
            "total_transport_cost",
            "source_excess_before",
            "source_excess_after",
            "destination_shortage_before",
            "destination_shortage_after",
        )

        st.dataframe(
            display_transfers[
                list(display_columns)
            ],
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download Transfer Plan CSV",
            data=dataframe_to_csv_bytes(
                filtered_transfers
            ),
            file_name="transfer_plan.csv",
            mime="text/csv",
            icon=":material/download:",
        )
