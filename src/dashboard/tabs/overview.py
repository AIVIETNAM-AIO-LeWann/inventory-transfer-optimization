"""Render the optimization overview tab."""

import streamlit as st

from src.dashboard.charts import create_shortage_chart
from src.dashboard.components import (
    render_configuration_card,
    render_overview_empty_state,
    render_result_badges,
    render_section_heading,
)
from src.dashboard.constants import (
    DATA_SOURCE_DISPLAY_NAMES,
    FORECAST_METHOD_DISPLAY_NAMES,
    OPTIMIZER_DISPLAY_NAMES,
)
from src.dashboard.formatters import (
    format_currency,
    format_decimal,
    format_integer,
    format_percentage,
)
from src.dashboard.state import get_data_source
from src.optimization_pipeline import (
    OptimizationPipelineResult,
)


def render_overview_tab(
    result: OptimizationPipelineResult | None,
) -> None:
    """Render headline metrics and selected configuration."""

    st.subheader("Optimization Overview")

    if result is None:
        render_overview_empty_state()
        return

    metrics = result.metrics
    data_source = get_data_source()
    dataset_name = DATA_SOURCE_DISPLAY_NAMES.get(
        data_source,
        "Active Dataset",
    )
    forecast_method = (
        FORECAST_METHOD_DISPLAY_NAMES.get(
            result.forecast_method,
            result.forecast_method,
        )
    )
    optimizer_name = OPTIMIZER_DISPLAY_NAMES.get(
        result.optimizer_name,
        result.optimizer_name,
    )

    render_result_badges(
        dataset_name=dataset_name,
        forecast_method=forecast_method,
        optimizer_name=optimizer_name,
        requested_horizon_days=(
            result.requested_horizon_days
        ),
        replenishment_horizon_days=(
            result.replenishment_horizon_days
        ),
    )

    render_section_heading(
        title="Inventory Performance",
        description=(
            "Planning coverage and shortage recovery "
            "after the transfer plan."
        ),
    )

    inventory_metrics = st.columns(4)

    inventory_metrics[0].metric(
        "📅 Requested Horizon",
        f"{result.requested_horizon_days} days",
    )
    inventory_metrics[1].metric(
        "🎯 Replenishment Horizon",
        f"{result.replenishment_horizon_days} days",
    )
    inventory_metrics[2].metric(
        "⚠️ Remaining Shortage",
        format_integer(
            metrics["remaining_shortage"]
        ),
    )
    inventory_metrics[3].metric(
        "✅ Shortage Resolution",
        format_percentage(
            metrics["shortage_resolution_rate"]
        ),
    )

    render_section_heading(
        title="Transfer Efficiency",
        description=(
            "Transfer volume and transportation "
            "cost efficiency."
        ),
    )

    transfer_metrics = st.columns(4)

    transfer_metrics[0].metric(
        "🚚 Transfer Count",
        format_integer(metrics["transfer_count"]),
    )
    transfer_metrics[1].metric(
        "📦 Transferred Units",
        format_integer(
            metrics["transferred_quantity"]
        ),
    )
    transfer_metrics[2].metric(
        "💰 Transport Cost",
        format_currency(
            metrics["total_transport_cost"]
        ),
    )
    transfer_metrics[3].metric(
        "🧾 Average Cost per Unit",
        format_currency(
            metrics["average_cost_per_unit"]
        ),
    )

    content_columns = st.columns(
        (3, 2),
        gap="large",
    )

    with content_columns[0]:
        render_section_heading(
            title="Shortage Recovery",
            description=(
                "Compare the initial shortage with "
                "the remaining shortage."
            ),
        )

        shortage_chart = create_shortage_chart(
            total_shortage=metrics["total_shortage"],
            remaining_shortage=(
                metrics["remaining_shortage"]
            ),
        )

        st.plotly_chart(
            shortage_chart,
            width="stretch",
        )

    with content_columns[1]:
        render_configuration_card(
            title="Run Details",
            items=(
                (
                    "Execution Time",
                    (
                        format_decimal(
                            metrics[
                                "execution_time_seconds"
                            ],
                            decimal_places=4,
                        )
                        + " seconds"
                    ),
                ),
                (
                    "Average Distance",
                    (
                        format_decimal(
                            metrics[
                                "weighted_average_distance_km"
                            ]
                        )
                        + " km"
                    ),
                ),
                (
                    "Average Lead Time",
                    (
                        format_decimal(
                            metrics[
                                "weighted_average_lead_time_minutes"
                            ]
                        )
                        + " minutes"
                    ),
                ),
            ),
        )
