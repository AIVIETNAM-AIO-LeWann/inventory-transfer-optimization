"""Build Plotly charts used by dashboard tabs."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.constants import (
    ROUTE_COLORS,
    ROUTE_DISPLAY_NAMES,
    STATUS_COLORS,
    STATUS_DISPLAY_NAMES,
)


def require_columns(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    data_name: str,
) -> None:
    """Validate columns required by a dashboard chart."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{data_name} must be a pandas DataFrame."
        )

    missing_columns = (
        set(required_columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def create_forecast_chart(
    forecast_data: pd.DataFrame,
) -> go.Figure:
    """Create a daily total-demand line chart."""

    require_columns(
        data=forecast_data,
        required_columns=(
            "forecast_date",
            "predicted_quantity",
        ),
        data_name="forecast_data",
    )

    daily_summary = (
        forecast_data.groupby(
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

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=daily_summary["forecast_date"],
            y=daily_summary["predicted_quantity"],
            mode="lines+markers",
            line={
                "color": "#006D5B",
                "width": 3,
                "shape": "spline",
            },
            marker={
                "color": "#FFFFFF",
                "line": {
                    "color": "#006D5B",
                    "width": 2,
                },
                "size": 8,
            },
            fill="tozeroy",
            fillcolor="rgba(0, 109, 91, 0.10)",
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Predicted demand: %{y:,.0f} units"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        showlegend=False,
        font={
            "color": "#344054",
            "family": "Segoe UI, sans-serif",
        },
        xaxis_title=None,
        yaxis_title="Predicted Units",
    )

    figure.update_xaxes(
        showgrid=False,
        tickformat="%d %b",
        linecolor="#E4E7EC",
    )

    figure.update_yaxes(
        gridcolor="#EEF1F4",
        rangemode="tozero",
        separatethousands=True,
    )

    return figure


def create_inventory_status_chart(
    inventory_data: pd.DataFrame,
) -> go.Figure:
    """Create an inventory-status donut chart."""

    require_columns(
        data=inventory_data,
        required_columns=("status", "product_id"),
        data_name="inventory_data",
    )

    status_summary = (
        inventory_data.groupby(
            "status",
            as_index=False,
            observed=False,
        )
        .agg(
            store_product_pairs=(
                "product_id",
                "size",
            )
        )
    )

    status_summary["display_status"] = (
        status_summary["status"].map(
            STATUS_DISPLAY_NAMES
        )
    )

    figure = go.Figure(
        data=[
            go.Pie(
                labels=status_summary[
                    "display_status"
                ],
                values=status_summary[
                    "store_product_pairs"
                ],
                hole=0.62,
                sort=False,
                marker={
                    "colors": [
                        STATUS_COLORS.get(
                            status,
                            "#667085",
                        )
                        for status in status_summary[
                            "status"
                        ]
                    ],
                    "line": {
                        "color": "#FFFFFF",
                        "width": 3,
                    },
                },
                textinfo="label+percent",
                textposition="outside",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:,.0f} store-product pairs"
                    "<br>%{percent}<extra></extra>"
                ),
            )
        ]
    )

    figure.update_layout(
        height=410,
        margin={
            "l": 35,
            "r": 35,
            "t": 25,
            "b": 25,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
        },
        font={
            "color": "#344054",
            "family": "Segoe UI, sans-serif",
        },
        annotations=[
            {
                "text": (
                    f"<b>{len(inventory_data):,}</b>"
                    "<br><span style='font-size:11px'>"
                    "Pairs</span>"
                ),
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {
                    "color": "#101828",
                    "size": 18,
                },
            }
        ],
    )

    return figure


def create_shortage_chart(
    total_shortage: int | float,
    remaining_shortage: int | float,
) -> go.Figure:
    """Compare shortage before and after transfer."""

    shortage_data = pd.DataFrame(
        {
            "Stage": ["Before Transfer", "After Transfer"],
            "Shortage": [
                total_shortage,
                remaining_shortage,
            ],
        }
    )

    return px.bar(
        shortage_data,
        x="Stage",
        y="Shortage",
        color="Stage",
        title="Shortage Before and After Transfer",
        color_discrete_sequence=(
            "#D92D20",
            "#12B76A",
        ),
    )


def create_transfer_quantity_chart(
    transfer_data: pd.DataFrame,
) -> go.Figure:
    """Create transferred quantity by route type."""

    route_summary = create_route_summary(
        transfer_data
    )

    figure = go.Figure(
        data=[
            go.Pie(
                labels=route_summary[
                    "display_route"
                ],
                values=route_summary[
                    "transferred_quantity"
                ],
                hole=0.58,
                sort=False,
                marker={
                    "colors": [
                        ROUTE_COLORS.get(
                            route_type,
                            "#667085",
                        )
                        for route_type in route_summary[
                            "route_type"
                        ]
                    ],
                    "line": {
                        "color": "#FFFFFF",
                        "width": 3,
                    },
                },
                textinfo="label+percent",
                textposition="outside",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:,.0f} units"
                    "<br>%{percent}<extra></extra>"
                ),
            )
        ]
    )

    figure.update_layout(
        height=390,
        margin={
            "l": 30,
            "r": 30,
            "t": 20,
            "b": 30,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        font={
            "color": "#344054",
            "family": "Segoe UI, sans-serif",
        },
    )

    return figure


def create_transport_cost_chart(
    transfer_data: pd.DataFrame,
) -> go.Figure:
    """Create transport cost by route type."""

    route_summary = create_route_summary(
        transfer_data
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=route_summary[
                    "display_route"
                ],
                y=route_summary[
                    "transport_cost"
                ],
                marker={
                    "color": [
                        ROUTE_COLORS.get(
                            route_type,
                            "#667085",
                        )
                        for route_type in route_summary[
                            "route_type"
                        ]
                    ],
                    "line": {
                        "width": 0,
                    },
                },
                text=route_summary[
                    "transport_cost"
                ],
                texttemplate="%{text:,.0f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Transport cost: %{y:,.0f} VND"
                    "<extra></extra>"
                ),
            )
        ]
    )

    figure.update_layout(
        height=390,
        margin={
            "l": 25,
            "r": 20,
            "t": 35,
            "b": 35,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        font={
            "color": "#344054",
            "family": "Segoe UI, sans-serif",
        },
        xaxis_title=None,
        yaxis_title="Transport Cost (VND)",
    )

    figure.update_xaxes(
        showgrid=False,
        linecolor="#E4E7EC",
    )

    figure.update_yaxes(
        gridcolor="#EEF1F4",
        rangemode="tozero",
        separatethousands=True,
    )

    return figure


def create_route_summary(
    transfer_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize transfer quantity and cost by route type."""

    require_columns(
        data=transfer_data,
        required_columns=(
            "route_type",
            "quantity",
            "total_transport_cost",
        ),
        data_name="transfer_data",
    )

    route_summary = (
        transfer_data.groupby(
            "route_type",
            as_index=False,
        )
        .agg(
            transferred_quantity=(
                "quantity",
                "sum",
            ),
            transport_cost=(
                "total_transport_cost",
                "sum",
            ),
        )
    )

    route_summary["display_route"] = (
        route_summary["route_type"].map(
            ROUTE_DISPLAY_NAMES
        )
    )

    return route_summary
