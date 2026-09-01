"""Create inventory transfer plans using a greedy algorithm."""

from pathlib import Path

import pandas as pd

from src.config import (
    GREEDY_TRANSFER_PLAN_FILE,
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
)
from src.data_loader import load_all_data
from src.inventory_analyzer import (
    EXCESS_STATUS,
    SHORTAGE_STATUS,
    analyze_inventory,
)
from src.route_analyzer import analyze_routes


INVENTORY_REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
    "inventory_days",
    "status",
    "shortage_quantity",
    "excess_quantity",
)

ROUTE_REQUIRED_COLUMNS = (
    "from_store_id",
    "to_store_id",
    "route_type",
    "priority_rank",
    "distance_km",
    "lead_time_minutes",
    "transport_cost_per_unit",
    "is_allowed",
)

TRANSFER_PLAN_COLUMNS = (
    "transfer_id",
    "product_id",
    "from_store_id",
    "to_store_id",
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


def validate_greedy_inputs(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
) -> None:
    """Validate inputs required by the greedy optimizer."""

    if inventory_analysis.empty:
        raise ValueError(
            "inventory_analysis must not be empty."
        )

    if route_analysis.empty:
        raise ValueError(
            "route_analysis must not be empty."
        )

    missing_inventory_columns = (
        set(INVENTORY_REQUIRED_COLUMNS)
        - set(inventory_analysis.columns)
    )

    if missing_inventory_columns:
        raise ValueError(
            "inventory_analysis is missing columns: "
            f"{sorted(missing_inventory_columns)}"
        )

    missing_route_columns = (
        set(ROUTE_REQUIRED_COLUMNS)
        - set(route_analysis.columns)
    )

    if missing_route_columns:
        raise ValueError(
            "route_analysis is missing columns: "
            f"{sorted(missing_route_columns)}"
        )

    if (
        inventory_analysis["shortage_quantity"] < 0
    ).any():
        raise ValueError(
            "shortage_quantity must not be negative."
        )

    if (
        inventory_analysis["excess_quantity"] < 0
    ).any():
        raise ValueError(
            "excess_quantity must not be negative."
        )

    duplicated_routes = route_analysis.duplicated(
        subset=[
            "from_store_id",
            "to_store_id",
        ],
        keep=False,
    )

    if duplicated_routes.any():
        raise ValueError(
            "route_analysis contains duplicate routes."
        )


def optimize_greedy(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Create a transfer plan using greedy allocation."""

    validate_greedy_inputs(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    shortage_rows = inventory_analysis.loc[
        inventory_analysis["status"]
        == SHORTAGE_STATUS
    ].copy()

    excess_rows = inventory_analysis.loc[
        inventory_analysis["status"]
        == EXCESS_STATUS
    ].copy()

    shortage_rows = shortage_rows.sort_values(
        by=[
            "inventory_days",
            "shortage_quantity",
            "store_id",
            "product_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        ignore_index=True,
    )

    allowed_routes = route_analysis.loc[
        route_analysis["is_allowed"]
    ].copy()

    route_lookup = allowed_routes.set_index(
        [
            "from_store_id",
            "to_store_id",
        ],
        drop=False,
    )

    remaining_excess = {
        (
            str(row.store_id),
            str(row.product_id),
        ): int(row.excess_quantity)
        for row in excess_rows.itertuples(index=False)
    }

    transfer_records: list[dict[str, object]] = []

    for shortage in shortage_rows.itertuples(
        index=False
    ):
        destination_store_id = str(
            shortage.store_id
        )

        product_id = str(
            shortage.product_id
        )

        remaining_shortage = int(
            shortage.shortage_quantity
        )

        product_sources = excess_rows.loc[
            excess_rows["product_id"].astype(str)
            == product_id
        ]

        candidate_sources: list[
            dict[str, object]
        ] = []

        for source in product_sources.itertuples(
            index=False
        ):
            source_store_id = str(source.store_id)

            source_key = (
                source_store_id,
                product_id,
            )

            available_excess = remaining_excess.get(
                source_key,
                0,
            )

            if available_excess <= 0:
                continue

            route_key = (
                source_store_id,
                destination_store_id,
            )

            if route_key not in route_lookup.index:
                continue

            route = route_lookup.loc[route_key]

            candidate_sources.append(
                {
                    "from_store_id": source_store_id,
                    "available_excess": (
                        available_excess
                    ),
                    "route_type": route[
                        "route_type"
                    ],
                    "priority_rank": int(
                        route["priority_rank"]
                    ),
                    "distance_km": float(
                        route["distance_km"]
                    ),
                    "lead_time_minutes": float(
                        route["lead_time_minutes"]
                    ),
                    "transport_cost_per_unit": float(
                        route[
                            "transport_cost_per_unit"
                        ]
                    ),
                }
            )

        candidate_sources.sort(
            key=lambda candidate: (
                candidate["priority_rank"],
                candidate[
                    "transport_cost_per_unit"
                ],
                candidate["lead_time_minutes"],
                -candidate["available_excess"],
                candidate["from_store_id"],
            )
        )

        for candidate in candidate_sources:
            if remaining_shortage <= 0:
                break

            source_store_id = str(
                candidate["from_store_id"]
            )

            source_key = (
                source_store_id,
                product_id,
            )

            source_excess_before = (
                remaining_excess[source_key]
            )

            destination_shortage_before = (
                remaining_shortage
            )

            transfer_quantity = min(
                source_excess_before,
                destination_shortage_before,
            )

            if transfer_quantity <= 0:
                continue

            source_excess_after = (
                source_excess_before
                - transfer_quantity
            )

            destination_shortage_after = (
                destination_shortage_before
                - transfer_quantity
            )

            remaining_excess[source_key] = (
                source_excess_after
            )

            remaining_shortage = (
                destination_shortage_after
            )

            transport_cost_per_unit = float(
                candidate[
                    "transport_cost_per_unit"
                ]
            )

            total_transport_cost = (
                transfer_quantity
                * transport_cost_per_unit
            )

            transfer_number = (
                len(transfer_records) + 1
            )

            transfer_records.append(
                {
                    "transfer_id": (
                        f"T{transfer_number:04d}"
                    ),
                    "product_id": product_id,
                    "from_store_id": (
                        source_store_id
                    ),
                    "to_store_id": (
                        destination_store_id
                    ),
                    "quantity": (
                        transfer_quantity
                    ),
                    "route_type": candidate[
                        "route_type"
                    ],
                    "distance_km": round(
                        float(
                            candidate["distance_km"]
                        ),
                        3,
                    ),
                    "lead_time_minutes": round(
                        float(
                            candidate[
                                "lead_time_minutes"
                            ]
                        ),
                        2,
                    ),
                    "transport_cost_per_unit": (
                        round(
                            transport_cost_per_unit,
                            2,
                        )
                    ),
                    "total_transport_cost": (
                        round(
                            total_transport_cost,
                            2,
                        )
                    ),
                    "source_excess_before": (
                        source_excess_before
                    ),
                    "source_excess_after": (
                        source_excess_after
                    ),
                    "destination_shortage_before": (
                        destination_shortage_before
                    ),
                    "destination_shortage_after": (
                        destination_shortage_after
                    ),
                }
            )

    return pd.DataFrame(
        transfer_records,
        columns=TRANSFER_PLAN_COLUMNS,
    )


def create_greedy_summary(
    inventory_analysis: pd.DataFrame,
    transfer_plan: pd.DataFrame,
) -> dict[str, int | float]:
    """Create summary metrics for a greedy transfer plan."""

    total_shortage = int(
        inventory_analysis["shortage_quantity"].sum()
    )

    total_excess = int(
        inventory_analysis["excess_quantity"].sum()
    )

    if transfer_plan.empty:
        transferred_quantity = 0
        total_transport_cost = 0.0
        intra_city_quantity = 0
        inter_city_quantity = 0
    else:
        transferred_quantity = int(
            transfer_plan["quantity"].sum()
        )

        total_transport_cost = float(
            transfer_plan[
                "total_transport_cost"
            ].sum()
        )

        intra_city_quantity = int(
            transfer_plan.loc[
                transfer_plan["route_type"]
                == INTRA_CITY_ROUTE,
                "quantity",
            ].sum()
        )

        inter_city_quantity = int(
            transfer_plan.loc[
                transfer_plan["route_type"]
                == INTER_CITY_ROUTE,
                "quantity",
            ].sum()
        )

    remaining_shortage = max(
        total_shortage - transferred_quantity,
        0,
    )

    remaining_excess = max(
        total_excess - transferred_quantity,
        0,
    )

    shortage_resolution_rate = (
        transferred_quantity / total_shortage
        if total_shortage > 0
        else 1.0
    )

    return {
        "transfer_count": len(transfer_plan),
        "total_shortage": total_shortage,
        "total_excess": total_excess,
        "transferred_quantity": transferred_quantity,
        "remaining_shortage": remaining_shortage,
        "remaining_excess": remaining_excess,
        "intra_city_quantity": intra_city_quantity,
        "inter_city_quantity": inter_city_quantity,
        "total_transport_cost": round(
            total_transport_cost,
            2,
        ),
        "shortage_resolution_rate": round(
            shortage_resolution_rate,
            4,
        ),
    }


def save_transfer_plan(
    transfer_plan: pd.DataFrame,
    output_path: str | Path = (
        GREEDY_TRANSFER_PLAN_FILE
    ),
) -> Path:
    """Save the greedy transfer plan to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transfer_plan.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Run the greedy inventory transfer optimizer."""

    project_data = load_all_data()

    inventory_analysis = analyze_inventory(
        sales=project_data.sales,
        inventory=project_data.inventory,
    )

    route_analysis = analyze_routes(
        stores=project_data.stores,
        distance_matrix=project_data.distance_matrix,
        duration_matrix=project_data.duration_matrix,
        transport_cost_matrix=(
            project_data.transport_cost_matrix
        ),
    )

    transfer_plan = optimize_greedy(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    summary = create_greedy_summary(
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
    )

    output_path = save_transfer_plan(
        transfer_plan
    )

    print("Greedy optimization completed successfully.")
    print(f"Saved to: {output_path}")
    print()

    for metric_name, metric_value in summary.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()