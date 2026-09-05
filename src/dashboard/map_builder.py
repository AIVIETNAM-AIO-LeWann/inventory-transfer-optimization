"""Build an interactive store and transfer-route map."""

from html import escape
from numbers import Integral

import folium
import numpy as np
import pandas as pd

from src.dashboard.constants import (
    DEFAULT_MAP_ZOOM,
    DEFAULT_MAX_DISPLAYED_ROUTES,
    ROUTE_COLORS,
    ROUTE_DISPLAY_NAMES,
    STATUS_COLORS,
    STATUS_DISPLAY_NAMES,
    STATUS_MARKER_COLORS,
    STORE_MARKER_ICON,
    STORE_MARKER_ICON_PREFIX,
)
from src.inventory_analyzer import (
    BALANCED_STATUS,
    EXCESS_STATUS,
    SHORTAGE_STATUS,
)


STORE_MAP_COLUMNS = (
    "store_id",
    "store_name",
    "city",
    "latitude",
    "longitude",
)


def validate_map_stores(
    stores: pd.DataFrame,
) -> None:
    """Validate store fields required by the map."""

    if not isinstance(stores, pd.DataFrame):
        raise TypeError(
            "stores must be a pandas DataFrame."
        )

    missing_columns = (
        set(STORE_MAP_COLUMNS) - set(stores.columns)
    )

    if missing_columns:
        raise ValueError(
            "stores is missing map columns: "
            f"{sorted(missing_columns)}"
        )

    if stores.empty:
        raise ValueError(
            "stores must not be empty."
        )

    if stores["store_id"].duplicated().any():
        raise ValueError(
            "stores contains duplicate store_id values."
        )

    try:
        coordinates = stores[
            ["latitude", "longitude"]
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Store coordinates must be numeric."
        ) from error

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "Store coordinates must be finite."
        )

    if not stores["latitude"].between(-90, 90).all():
        raise ValueError(
            "Store latitude must be between -90 and 90."
        )

    if not stores["longitude"].between(
        -180,
        180,
    ).all():
        raise ValueError(
            "Store longitude must be between -180 and 180."
        )


def create_store_map_summary(
    stores: pd.DataFrame,
    inventory_analysis: pd.DataFrame | None = None,
    transfer_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate inventory and transfer information by store."""

    validate_map_stores(stores)

    summary = stores[
        list(STORE_MAP_COLUMNS)
    ].copy()
    summary["shortage_quantity"] = 0.0
    summary["excess_quantity"] = 0.0
    summary["shortage_products"] = 0
    summary["excess_products"] = 0
    summary["inbound_quantity"] = 0.0
    summary["outbound_quantity"] = 0.0

    if inventory_analysis is not None:
        required_columns = {
            "store_id",
            "status",
            "shortage_quantity",
            "excess_quantity",
        }
        missing_columns = (
            required_columns
            - set(inventory_analysis.columns)
        )

        if missing_columns:
            raise ValueError(
                "inventory_analysis is missing columns: "
                f"{sorted(missing_columns)}"
            )

        inventory_summary = (
            inventory_analysis.groupby(
                "store_id",
                as_index=False,
            )
            .agg(
                shortage_quantity=(
                    "shortage_quantity",
                    "sum",
                ),
                excess_quantity=(
                    "excess_quantity",
                    "sum",
                ),
                shortage_products=(
                    "status",
                    lambda values: int(
                        (values == SHORTAGE_STATUS).sum()
                    ),
                ),
                excess_products=(
                    "status",
                    lambda values: int(
                        (values == EXCESS_STATUS).sum()
                    ),
                ),
            )
        )

        summary = summary.drop(
            columns=[
                "shortage_quantity",
                "excess_quantity",
                "shortage_products",
                "excess_products",
            ]
        ).merge(
            inventory_summary,
            on="store_id",
            how="left",
            validate="one_to_one",
        )

        for column in (
            "shortage_quantity",
            "excess_quantity",
            "shortage_products",
            "excess_products",
        ):
            summary[column] = summary[column].fillna(0)

    if transfer_plan is not None and not transfer_plan.empty:
        required_columns = {
            "from_store_id",
            "to_store_id",
            "quantity",
        }
        missing_columns = (
            required_columns - set(transfer_plan.columns)
        )

        if missing_columns:
            raise ValueError(
                "transfer_plan is missing columns: "
                f"{sorted(missing_columns)}"
            )

        outbound = (
            transfer_plan.groupby(
                "from_store_id"
            )["quantity"]
            .sum()
        )
        inbound = (
            transfer_plan.groupby(
                "to_store_id"
            )["quantity"]
            .sum()
        )

        summary["outbound_quantity"] = (
            summary["store_id"]
            .map(outbound)
            .fillna(0)
        )
        summary["inbound_quantity"] = (
            summary["store_id"]
            .map(inbound)
            .fillna(0)
        )

    summary["map_status"] = BALANCED_STATUS
    summary.loc[
        summary["excess_quantity"] > 0,
        "map_status",
    ] = EXCESS_STATUS
    summary.loc[
        summary["shortage_quantity"] > 0,
        "map_status",
    ] = SHORTAGE_STATUS

    return summary


def validate_max_routes(
    max_routes: int,
) -> None:
    """Validate the route display limit."""

    if (
        isinstance(max_routes, bool)
        or not isinstance(max_routes, Integral)
        or max_routes < 0
    ):
        raise ValueError(
            "max_routes must be a nonnegative integer."
        )


def create_store_popup(
    store: pd.Series,
) -> str:
    """Create safe HTML content for one store marker."""

    store_name = escape(str(store["store_name"]))
    store_id = escape(str(store["store_id"]))
    city = escape(str(store["city"]))
    raw_status = str(store["map_status"])
    status = escape(
        STATUS_DISPLAY_NAMES.get(
            raw_status,
            raw_status.title(),
        )
    )
    status_color = STATUS_COLORS.get(
        raw_status,
        "#667085",
    )

    return f"""
    <div style="
        width: 280px;
        color: #101828;
        font-family: Segoe UI, sans-serif;
    ">
        <div style="
            padding-bottom: 10px;
            border-bottom: 1px solid #e4e7ec;
        ">
            <div style="
                font-size: 16px;
                font-weight: 700;
            ">{store_name}</div>
            <div style="
                margin-top: 3px;
                color: #667085;
                font-size: 12px;
            ">{store_id} · {city}</div>
            <span style="
                display: inline-block;
                margin-top: 8px;
                padding: 3px 8px;
                background: {status_color}18;
                border-radius: 999px;
                color: {status_color};
                font-size: 11px;
                font-weight: 700;
            ">{status}</span>
        </div>
        <table style="
            width: 100%;
            margin-top: 8px;
            border-collapse: collapse;
            font-size: 12px;
        ">
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Shortage units
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {float(store['shortage_quantity']):,.0f}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Excess units
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {float(store['excess_quantity']):,.0f}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Shortage products
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {int(store['shortage_products']):,}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Excess products
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {int(store['excess_products']):,}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Inbound units
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {float(store['inbound_quantity']):,.0f}
                </td>
            </tr>
            <tr>
                <td style="padding: 4px 0; color: #667085;">
                    Outbound units
                </td>
                <td style="text-align: right; font-weight: 700;">
                    {float(store['outbound_quantity']):,.0f}
                </td>
            </tr>
        </table>
    </div>
    """


def add_network_legend(
    network_map: folium.Map,
) -> None:
    """Add inventory-status and route legends to a map."""

    legend_html = f"""
    <div style="
        position: fixed;
        z-index: 9999;
        left: 24px;
        bottom: 24px;
        width: 190px;
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dfe3e8;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.16);
        color: #344054;
        font-family: Segoe UI, sans-serif;
        font-size: 12px;
    ">
        <div style="
            margin-bottom: 8px;
            color: #101828;
            font-size: 13px;
            font-weight: 700;
        ">Map Legend</div>
        <div style="margin-bottom: 5px;">
            <span style="color: {STATUS_COLORS[SHORTAGE_STATUS]};">
                ●
            </span>
            Shortage store
        </div>
        <div style="margin-bottom: 5px;">
            <span style="color: {STATUS_COLORS[BALANCED_STATUS]};">
                ●
            </span>
            Balanced store
        </div>
        <div style="margin-bottom: 8px;">
            <span style="color: {STATUS_COLORS[EXCESS_STATUS]};">
                ●
            </span>
            Excess store
        </div>
        <div style="
            margin: 7px 0;
            border-top: 1px solid #e4e7ec;
        "></div>
        <div style="margin-bottom: 5px;">
            <span style="color: {ROUTE_COLORS['intra_city']};">
                ━━
            </span>
            {ROUTE_DISPLAY_NAMES['intra_city']}
        </div>
        <div>
            <span style="color: {ROUTE_COLORS['inter_city']};">
                ━━
            </span>
            {ROUTE_DISPLAY_NAMES['inter_city']}
        </div>
    </div>
    """

    network_map.get_root().html.add_child(
        folium.Element(legend_html)
    )


def add_transfer_routes(
    network_map: folium.Map,
    stores: pd.DataFrame,
    transfer_plan: pd.DataFrame,
    max_routes: int,
) -> None:
    """Add the highest-volume transfer routes to a map."""

    validate_max_routes(max_routes)

    if transfer_plan.empty or max_routes == 0:
        return

    required_columns = {
        "from_store_id",
        "to_store_id",
        "product_id",
        "quantity",
        "route_type",
    }
    missing_columns = (
        required_columns - set(transfer_plan.columns)
    )

    if missing_columns:
        raise ValueError(
            "transfer_plan is missing map columns: "
            f"{sorted(missing_columns)}"
        )

    coordinates = stores.set_index("store_id")
    displayed_routes = transfer_plan.sort_values(
        "quantity",
        ascending=False,
    ).head(int(max_routes))
    route_layer = folium.FeatureGroup(
        name="Transfer Routes"
    )

    for route in displayed_routes.itertuples(
        index=False
    ):
        if (
            route.from_store_id not in coordinates.index
            or route.to_store_id not in coordinates.index
        ):
            raise ValueError(
                "A transfer route references an unknown store."
            )

        source = coordinates.loc[route.from_store_id]
        destination = coordinates.loc[route.to_store_id]
        route_color = ROUTE_COLORS.get(
            route.route_type,
            "#667085",
        )
        source_name = escape(
            str(
                getattr(
                    route,
                    "from_store_name",
                    route.from_store_id,
                )
            )
        )
        destination_name = escape(
            str(
                getattr(
                    route,
                    "to_store_name",
                    route.to_store_id,
                )
            )
        )
        product_name = escape(
            str(
                getattr(
                    route,
                    "product_name",
                    route.product_id,
                )
            )
        )
        distance_km = float(
            getattr(route, "distance_km", 0)
        )
        lead_time_minutes = float(
            getattr(route, "lead_time_minutes", 0)
        )
        route_name = escape(
            ROUTE_DISPLAY_NAMES.get(
                route.route_type,
                str(route.route_type),
            )
        )
        tooltip = (
            '<div style="font-family: Segoe UI, sans-serif;">'
            f"<b>{source_name} → {destination_name}</b><br>"
            f"{product_name}: {int(route.quantity):,} units"
            f"<br>{route_name} · {distance_km:,.1f} km"
            f" · {lead_time_minutes:,.0f} min"
            "</div>"
        )

        folium.PolyLine(
            locations=[
                [source["latitude"], source["longitude"]],
                [
                    destination["latitude"],
                    destination["longitude"],
                ],
            ],
            color=route_color,
            weight=4,
            opacity=0.75,
            tooltip=folium.Tooltip(
                tooltip,
                sticky=True,
            ),
        ).add_to(route_layer)

    route_layer.add_to(network_map)


def build_network_map(
    stores: pd.DataFrame,
    inventory_analysis: pd.DataFrame | None = None,
    transfer_plan: pd.DataFrame | None = None,
    max_routes: int = DEFAULT_MAX_DISPLAYED_ROUTES,
) -> folium.Map:
    """Build a Folium map for stores and transfer routes."""

    validate_max_routes(max_routes)

    store_summary = create_store_map_summary(
        stores=stores,
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
    )
    center = [
        float(store_summary["latitude"].mean()),
        float(store_summary["longitude"].mean()),
    ]
    network_map = folium.Map(
        location=center,
        zoom_start=(
            12
            if len(store_summary) == 1
            else DEFAULT_MAP_ZOOM
        ),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if len(store_summary) > 1:
        network_map.fit_bounds(
            [
                [
                    float(
                        store_summary["latitude"].min()
                    ),
                    float(
                        store_summary["longitude"].min()
                    ),
                ],
                [
                    float(
                        store_summary["latitude"].max()
                    ),
                    float(
                        store_summary["longitude"].max()
                    ),
                ],
            ],
            padding=(30, 30),
            max_zoom=12,
        )
    marker_layer = folium.FeatureGroup(
        name="Stores"
    )

    for _, store in store_summary.iterrows():
        status = str(store["map_status"])
        marker_color = STATUS_MARKER_COLORS.get(
            status,
            "gray",
        )

        folium.Marker(
            location=[
                float(store["latitude"]),
                float(store["longitude"]),
            ],
            tooltip=(
                f"{store['store_name']} - "
                f"{status.title()}"
            ),
            popup=folium.Popup(
                create_store_popup(store),
                max_width=320,
            ),
            icon=folium.Icon(
                color=marker_color,
                icon=STORE_MARKER_ICON,
                prefix=STORE_MARKER_ICON_PREFIX,
            ),
        ).add_to(marker_layer)

    marker_layer.add_to(network_map)

    if transfer_plan is not None:
        add_transfer_routes(
            network_map=network_map,
            stores=stores,
            transfer_plan=transfer_plan,
            max_routes=max_routes,
        )

    folium.LayerControl(
        collapsed=False
    ).add_to(network_map)

    add_network_legend(network_map)

    return network_map
