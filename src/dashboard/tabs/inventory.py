"""Render the inventory health tab."""

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    create_inventory_status_chart,
)
from src.dashboard.components import (
    render_page_empty_state,
    render_section_heading,
)
from src.dashboard.constants import (
    STATUS_DISPLAY_NAMES,
)
from src.dashboard.data_views import (
    enrich_inventory_data,
)
from src.dashboard.formatters import (
    dataframe_to_csv_bytes,
    format_integer,
    format_percentage,
)
from src.data_loader import ProjectData
from src.inventory_analyzer import (
    BALANCED_STATUS,
    EXCESS_STATUS,
    SHORTAGE_STATUS,
)
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def filter_inventory_data(
    inventory_data: pd.DataFrame,
    selected_statuses: list[str],
    selected_cities: list[str],
    selected_stores: list[str],
    selected_categories: list[str],
    selected_products: list[str],
) -> pd.DataFrame:
    """Apply dashboard filters to inventory analysis."""

    filtered_data = inventory_data.loc[
        inventory_data["status"].isin(
            selected_statuses
        )
    ].copy()

    filter_values = (
        ("city", selected_cities),
        ("store_id", selected_stores),
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


def render_inventory_tab(
    result: OptimizationPipelineResult | None,
    project_data: ProjectData,
) -> None:
    """Render inventory status chart, filters, and export."""

    st.subheader("Inventory Health")

    if result is None:
        render_page_empty_state(
            icon="🏥",
            title="No inventory analysis yet",
            description=(
                "Run optimization to classify inventory "
                "as shortage, balanced, or excess."
            ),
        )
        return

    inventory_data = enrich_inventory_data(
        inventory_analysis=(
            result.inventory_analysis
        ),
        stores=project_data.stores,
        products=project_data.products,
    )

    store_names = (
        inventory_data[
            ["store_id", "store_name"]
        ]
        .drop_duplicates("store_id")
        .set_index("store_id")["store_name"]
        .to_dict()
    )
    product_names = (
        inventory_data[
            ["product_id", "product_name"]
        ]
        .drop_duplicates("product_id")
        .set_index("product_id")["product_name"]
        .to_dict()
    )

    available_statuses = [
        status
        for status in STATUS_DISPLAY_NAMES
        if status in set(
            inventory_data["status"]
        )
    ]

    render_section_heading(
        title="Inventory Filters",
        description=(
            "Inspect inventory health by status, "
            "location, or product."
        ),
    )

    with st.container(border=True):
        filter_columns = st.columns(5)

        selected_statuses = (
            filter_columns[0].multiselect(
                "Status",
                options=available_statuses,
                default=available_statuses,
                format_func=lambda status: (
                    STATUS_DISPLAY_NAMES.get(
                        status,
                        status.title(),
                    )
                ),
                key="inventory_status_filter",
            )
        )
        selected_cities = (
            filter_columns[1].multiselect(
                "City",
                options=sorted(
                    inventory_data["city"].unique()
                ),
                key="inventory_city_filter",
                placeholder="All cities",
            )
        )
        selected_stores = (
            filter_columns[2].multiselect(
                "Store",
                options=sorted(
                    inventory_data[
                        "store_id"
                    ].unique()
                ),
                format_func=lambda store_id: (
                    f"{store_id} · "
                    f"{store_names.get(store_id, store_id)}"
                ),
                key="inventory_store_filter",
                placeholder="All stores",
            )
        )
        selected_categories = (
            filter_columns[3].multiselect(
                "Category",
                options=sorted(
                    inventory_data[
                        "category"
                    ].unique()
                ),
                key="inventory_category_filter",
                placeholder="All categories",
            )
        )
        selected_products = (
            filter_columns[4].multiselect(
                "Product",
                options=sorted(
                    inventory_data[
                        "product_id"
                    ].unique()
                ),
                format_func=lambda product_id: (
                    f"{product_id} · "
                    f"{product_names.get(product_id, product_id)}"
                ),
                key="inventory_product_filter",
                placeholder="All products",
            )
        )

    filtered_inventory = filter_inventory_data(
        inventory_data=inventory_data,
        selected_statuses=selected_statuses,
        selected_cities=selected_cities,
        selected_stores=selected_stores,
        selected_categories=selected_categories,
        selected_products=selected_products,
    )

    if filtered_inventory.empty:
        st.warning(
            "No inventory rows match the selected filters."
        )
        return

    total_pairs = len(filtered_inventory)
    balanced_pairs = int(
        (
            filtered_inventory["status"]
            == BALANCED_STATUS
        ).sum()
    )

    render_section_heading(
        title="Inventory Summary",
        description=(
            "Current stock position for the selected "
            "store-product scope."
        ),
    )

    summary_metrics = st.columns(4)

    summary_metrics[0].metric(
        "📦 Total Current Stock",
        format_integer(
            filtered_inventory[
                "current_stock"
            ].sum()
        ),
    )
    summary_metrics[1].metric(
        "🔴 Shortage Units",
        format_integer(
            filtered_inventory[
                "shortage_quantity"
            ].sum()
        ),
    )
    summary_metrics[2].metric(
        "🔵 Excess Units",
        format_integer(
            filtered_inventory[
                "excess_quantity"
            ].sum()
        ),
    )
    summary_metrics[3].metric(
        "🟢 Healthy Pair Rate",
        format_percentage(
            balanced_pairs / total_pairs
        ),
    )

    visual_columns = st.columns(
        (2, 3),
        gap="large",
    )

    with visual_columns[0]:
        render_section_heading(
            title="Status Distribution",
            description=(
                "Share of shortage, balanced, "
                "and excess pairs."
            ),
        )

        st.plotly_chart(
            create_inventory_status_chart(
                filtered_inventory
            ),
            width="stretch",
        )

    with visual_columns[1]:
        render_section_heading(
            title="Highest Shortage Risk",
            description=(
                "Store-product pairs requiring "
                "the most replenishment."
            ),
        )

        shortage_rows = (
            filtered_inventory.loc[
                filtered_inventory["status"]
                == SHORTAGE_STATUS
            ]
            .nlargest(
                8,
                "shortage_quantity",
            )
        )

        if shortage_rows.empty:
            st.success(
                "No shortage rows in the selected scope."
            )
        else:
            st.dataframe(
                shortage_rows[
                    [
                        "store_name",
                        "product_name",
                        "city",
                        "current_stock",
                        "target_stock",
                        "shortage_quantity",
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "store_name": "Store",
                    "product_name": "Product",
                    "city": "City",
                    "current_stock": st.column_config.NumberColumn(
                        "Current Stock",
                        format="%d",
                    ),
                    "target_stock": st.column_config.NumberColumn(
                        "Target Stock",
                        format="%d",
                    ),
                    "shortage_quantity": (
                        st.column_config.NumberColumn(
                            "Shortage",
                            format="%d",
                        )
                    ),
                },
            )

    status_labels = {
        SHORTAGE_STATUS: "🔴 Shortage",
        BALANCED_STATUS: "🟢 Balanced",
        EXCESS_STATUS: "🔵 Excess",
    }
    display_inventory = filtered_inventory.copy()
    display_inventory["status"] = (
        display_inventory["status"].map(
            status_labels
        )
    )

    with st.expander(
        f"Inventory Records ({len(display_inventory):,})"
    ):
        display_columns = (
            "store_id",
            "store_name",
            "city",
            "product_id",
            "product_name",
            "category",
            "current_stock",
            "predicted_horizon_demand",
            "inventory_days",
            "target_stock",
            "status",
            "shortage_quantity",
            "excess_quantity",
        )

        st.dataframe(
            display_inventory[
                list(display_columns)
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "status": "Inventory Status",
                "inventory_days": (
                    st.column_config.NumberColumn(
                        "Inventory Days",
                        format="%.2f",
                    )
                ),
            },
        )

        st.download_button(
            "Download Inventory Analysis CSV",
            data=dataframe_to_csv_bytes(
                filtered_inventory
            ),
            file_name="inventory_analysis.csv",
            mime="text/csv",
            icon=":material/download:",
        )
