"""Render the demand forecast tab."""

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    create_forecast_chart,
)
from src.dashboard.components import (
    render_page_empty_state,
    render_section_heading,
)
from src.dashboard.data_views import (
    enrich_forecast_data,
)
from src.dashboard.formatters import (
    dataframe_to_csv_bytes,
    format_integer,
)
from src.data_loader import ProjectData
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def filter_forecast_data(
    forecast_data: pd.DataFrame,
    selected_cities: list[str],
    selected_stores: list[str],
    selected_categories: list[str],
    selected_products: list[str],
) -> pd.DataFrame:
    """Apply dashboard filters to enriched forecasts."""

    filtered_data = forecast_data.copy()

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


def render_forecast_tab(
    result: OptimizationPipelineResult | None,
    project_data: ProjectData,
) -> None:
    """Render forecast chart, filters, table, and export."""

    st.subheader("Demand Forecast")

    if result is None:
        render_page_empty_state(
            icon="📈",
            title="No demand forecast yet",
            description=(
                "Choose a forecasting method in the sidebar "
                "and run optimization to generate predictions."
            ),
        )
        return

    forecast_data = enrich_forecast_data(
        forecast=result.daily_forecast,
        stores=project_data.stores,
        products=project_data.products,
    )

    store_names = (
        forecast_data[
            ["store_id", "store_name"]
        ]
        .drop_duplicates("store_id")
        .set_index("store_id")["store_name"]
        .to_dict()
    )
    product_names = (
        forecast_data[
            ["product_id", "product_name"]
        ]
        .drop_duplicates("product_id")
        .set_index("product_id")["product_name"]
        .to_dict()
    )

    render_section_heading(
        title="Forecast Filters",
        description=(
            "Narrow the forecast by location or "
            "product attributes."
        ),
    )

    with st.container(border=True):
        filter_columns = st.columns(4)

        selected_cities = filter_columns[0].multiselect(
            "City",
            options=sorted(
                forecast_data["city"].unique()
            ),
            key="forecast_city_filter",
            placeholder="All cities",
        )

        available_store_data = forecast_data

        if selected_cities:
            available_store_data = (
                available_store_data.loc[
                    available_store_data["city"].isin(
                        selected_cities
                    )
                ]
            )

        available_store_ids = sorted(
            available_store_data[
                "store_id"
            ].unique()
        )

        current_store_selection = (
            st.session_state.get(
                "forecast_store_filter",
                [],
            )
        )

        valid_store_selection = [
            store_id
            for store_id in current_store_selection
            if store_id in available_store_ids
        ]

        if (
            current_store_selection
            != valid_store_selection
        ):
            st.session_state[
                "forecast_store_filter"
            ] = valid_store_selection

        selected_stores = (
            filter_columns[1].multiselect(
                "Store",
                options=available_store_ids,
                format_func=lambda store_id: (
                    f"{store_id} · "
                    f"{store_names.get(store_id, store_id)}"
                ),
                key="forecast_store_filter",
                placeholder="All matching stores",
            )
        )
        selected_categories = (
            filter_columns[2].multiselect(
                "Category",
                options=sorted(
                    forecast_data["category"].unique()
                ),
                key="forecast_category_filter",
                placeholder="All categories",
            )
        )

        available_product_data = forecast_data

        if selected_categories:
            available_product_data = (
                available_product_data.loc[
                    available_product_data[
                        "category"
                    ].isin(selected_categories)
                ]
            )

        available_product_ids = sorted(
            available_product_data[
                "product_id"
            ].unique()
        )

        current_product_selection = (
            st.session_state.get(
                "forecast_product_filter",
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
                "forecast_product_filter"
            ] = valid_product_selection

        selected_products = (
            filter_columns[3].multiselect(
                "Product",
                options=available_product_ids,
                format_func=lambda product_id: (
                    f"{product_id} · "
                    f"{product_names.get(product_id, product_id)}"
                ),
                key="forecast_product_filter",
                placeholder="All matching products",
            )
        )

    filtered_data = filter_forecast_data(
        forecast_data=forecast_data,
        selected_cities=selected_cities,
        selected_stores=selected_stores,
        selected_categories=selected_categories,
        selected_products=selected_products,
    )

    if filtered_data.empty:
        st.warning(
            "No forecast rows match the selected filters."
        )
        return

    daily_summary = (
        filtered_data.groupby(
            "forecast_date",
            as_index=False,
        )
        .agg(
            predicted_quantity=(
                "predicted_quantity",
                "sum",
            )
        )
        .sort_values("forecast_date")
    )
    peak_row = daily_summary.loc[
        daily_summary["predicted_quantity"].idxmax()
    ]
    total_demand = filtered_data[
        "predicted_quantity"
    ].sum()
    average_daily_demand = daily_summary[
        "predicted_quantity"
    ].mean()
    active_pairs = (
        filtered_data[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    render_section_heading(
        title="Forecast Summary",
        description=(
            "Key figures for the currently filtered "
            "forecast scope."
        ),
    )

    summary_metrics = st.columns(4)

    summary_metrics[0].metric(
        "📦 Total Predicted Demand",
        format_integer(total_demand),
    )
    summary_metrics[1].metric(
        "📊 Average Daily Demand",
        format_integer(average_daily_demand),
    )
    summary_metrics[2].metric(
        "📅 Peak Demand Date",
        pd.Timestamp(
            peak_row["forecast_date"]
        ).strftime("%d %b %Y"),
        delta=(
            format_integer(
                peak_row["predicted_quantity"]
            )
            + " units"
        ),
        delta_color="off",
    )
    summary_metrics[3].metric(
        "🔗 Active Store-Product Pairs",
        format_integer(active_pairs),
    )

    render_section_heading(
        title="Predicted Demand Trend",
        description=(
            "Daily predicted units across the "
            "selected forecast scope."
        ),
    )

    st.plotly_chart(
        create_forecast_chart(filtered_data),
        width="stretch",
    )

    with st.expander(
        f"Forecast Records ({len(filtered_data):,})"
    ):
        display_columns = (
            "forecast_date",
            "forecast_day",
            "store_id",
            "store_name",
            "city",
            "product_id",
            "product_name",
            "category",
            "predicted_quantity",
            "method",
        )

        st.dataframe(
            filtered_data[
                list(display_columns)
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "forecast_date": st.column_config.DateColumn(
                    "Forecast Date",
                    format="DD MMM YYYY",
                ),
                "predicted_quantity": (
                    st.column_config.NumberColumn(
                        "Predicted Quantity",
                        format="%.2f",
                    )
                ),
            },
        )

        st.download_button(
            "Download Forecast CSV",
            data=dataframe_to_csv_bytes(
                filtered_data
            ),
            file_name="demand_forecast.csv",
            mime="text/csv",
            icon=":material/download:",
        )
