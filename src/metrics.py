"""Calculate and compare optimization performance metrics."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import (
    ALGORITHM_COMPARISON_FILE,
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
)
from src.data_loader import load_all_data
from src.inventory_analyzer import analyze_inventory
from src.optimizers.greedy import optimize_greedy
from src.route_analyzer import analyze_routes


INVENTORY_REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
    "shortage_quantity",
    "excess_quantity",
)

TRANSFER_REQUIRED_COLUMNS = (
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
)


def validate_metrics_inputs(
    inventory_analysis: pd.DataFrame,
    transfer_plan: pd.DataFrame,
    algorithm_name: str,
    execution_time_seconds: float,
) -> None:
    """Validate data used to calculate optimization metrics."""

    if inventory_analysis.empty:
        raise ValueError(
            "inventory_analysis must not be empty."
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

    missing_transfer_columns = (
        set(TRANSFER_REQUIRED_COLUMNS)
        - set(transfer_plan.columns)
    )

    if missing_transfer_columns:
        raise ValueError(
            "transfer_plan is missing columns: "
            f"{sorted(missing_transfer_columns)}"
        )

    if not algorithm_name.strip():
        raise ValueError(
            "algorithm_name must not be blank."
        )

    if execution_time_seconds < 0:
        raise ValueError(
            "execution_time_seconds must not be negative."
        )

    if transfer_plan.empty:
        return

    if transfer_plan["transfer_id"].duplicated().any():
        raise ValueError(
            "transfer_id values must be unique."
        )

    if (transfer_plan["quantity"] <= 0).any():
        raise ValueError(
            "Transfer quantities must be greater than zero."
        )

    if (
        transfer_plan["from_store_id"]
        == transfer_plan["to_store_id"]
    ).any():
        raise ValueError(
            "A store cannot transfer inventory to itself."
        )

    numeric_columns = (
        "quantity",
        "distance_km",
        "lead_time_minutes",
        "transport_cost_per_unit",
        "total_transport_cost",
    )

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(
            transfer_plan[column]
        ):
            raise TypeError(
                f"transfer_plan.{column} must be numeric."
            )

        if (transfer_plan[column] < 0).any():
            raise ValueError(
                f"transfer_plan.{column} "
                "must not contain negative values."
            )

    expected_cost = (
        transfer_plan["quantity"]
        * transfer_plan["transport_cost_per_unit"]
    )

    if not np.allclose(
        transfer_plan["total_transport_cost"],
        expected_cost,
        atol=0.01,
    ):
        raise ValueError(
            "total_transport_cost does not match "
            "quantity multiplied by cost per unit."
        )

    inventory_limits = inventory_analysis.set_index(
        ["store_id", "product_id"]
    )

    source_usage = (
        transfer_plan.groupby(
            ["from_store_id", "product_id"]
        )["quantity"]
        .sum()
    )

    for source_key, used_quantity in (
        source_usage.items()
    ):
        if source_key not in inventory_limits.index:
            raise ValueError(
                f"Unknown source-product pair: {source_key}"
            )

        available_excess = inventory_limits.loc[
            source_key,
            "excess_quantity",
        ]

        if used_quantity > available_excess:
            raise ValueError(
                f"Source {source_key} transfers "
                "more than its available excess."
            )

    destination_usage = (
        transfer_plan.groupby(
            ["to_store_id", "product_id"]
        )["quantity"]
        .sum()
    )

    for destination_key, received_quantity in (
        destination_usage.items()
    ):
        if destination_key not in inventory_limits.index:
            raise ValueError(
                "Unknown destination-product pair: "
                f"{destination_key}"
            )

        required_shortage = inventory_limits.loc[
            destination_key,
            "shortage_quantity",
        ]

        if received_quantity > required_shortage:
            raise ValueError(
                f"Destination {destination_key} receives "
                "more than its shortage."
            )


def calculate_plan_metrics(
    inventory_analysis: pd.DataFrame,
    transfer_plan: pd.DataFrame,
    algorithm_name: str,
    execution_time_seconds: float,
) -> dict[str, str | int | float]:
    """Calculate performance metrics for one transfer plan."""

    validate_metrics_inputs(
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
        algorithm_name=algorithm_name,
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    total_shortage = int(
        inventory_analysis["shortage_quantity"].sum()
    )

    total_excess = int(
        inventory_analysis["excess_quantity"].sum()
    )

    if transfer_plan.empty:
        transferred_quantity = 0
        total_transport_cost = 0.0
        average_cost_per_unit = 0.0
        weighted_average_distance_km = 0.0
        weighted_average_lead_time_minutes = 0.0
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

        average_cost_per_unit = (
            total_transport_cost
            / transferred_quantity
        )

        weighted_average_distance_km = float(
            np.average(
                transfer_plan["distance_km"],
                weights=transfer_plan["quantity"],
            )
        )

        weighted_average_lead_time_minutes = float(
            np.average(
                transfer_plan["lead_time_minutes"],
                weights=transfer_plan["quantity"],
            )
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

    excess_utilization_rate = (
        transferred_quantity / total_excess
        if total_excess > 0
        else 0.0
    )

    intra_city_quantity_rate = (
        intra_city_quantity / transferred_quantity
        if transferred_quantity > 0
        else 0.0
    )

    inter_city_quantity_rate = (
        inter_city_quantity / transferred_quantity
        if transferred_quantity > 0
        else 0.0
    )

    return {
        "algorithm": algorithm_name,
        "transfer_count": len(transfer_plan),
        "total_shortage": total_shortage,
        "total_excess": total_excess,
        "transferred_quantity": transferred_quantity,
        "remaining_shortage": remaining_shortage,
        "remaining_excess": remaining_excess,
        "shortage_resolution_rate": round(
            shortage_resolution_rate,
            4,
        ),
        "excess_utilization_rate": round(
            excess_utilization_rate,
            4,
        ),
        "intra_city_quantity": intra_city_quantity,
        "inter_city_quantity": inter_city_quantity,
        "intra_city_quantity_rate": round(
            intra_city_quantity_rate,
            4,
        ),
        "inter_city_quantity_rate": round(
            inter_city_quantity_rate,
            4,
        ),
        "total_transport_cost": round(
            total_transport_cost,
            2,
        ),
        "average_cost_per_unit": round(
            average_cost_per_unit,
            2,
        ),
        "weighted_average_distance_km": round(
            weighted_average_distance_km,
            2,
        ),
        "weighted_average_lead_time_minutes": round(
            weighted_average_lead_time_minutes,
            2,
        ),
        "execution_time_seconds": round(
            execution_time_seconds,
            6,
        ),
    }


def create_algorithm_comparison(
    metric_records: list[
        dict[str, str | int | float]
    ],
) -> pd.DataFrame:
    """Create and rank an algorithm comparison table."""

    if not metric_records:
        raise ValueError(
            "metric_records must not be empty."
        )

    comparison = pd.DataFrame(metric_records)

    if comparison["algorithm"].duplicated().any():
        raise ValueError(
            "Algorithm names must be unique."
        )

    comparison = comparison.sort_values(
        by=[
            "shortage_resolution_rate",
            "total_transport_cost",
            "execution_time_seconds",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        ignore_index=True,
    )

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    return comparison


def save_algorithm_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path = (
        ALGORITHM_COMPARISON_FILE
    ),
) -> Path:
    """Save algorithm comparison results to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Evaluate the greedy optimizer."""

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

    start_time = perf_counter()

    greedy_plan = optimize_greedy(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    execution_time_seconds = (
        perf_counter() - start_time
    )

    greedy_metrics = calculate_plan_metrics(
        inventory_analysis=inventory_analysis,
        transfer_plan=greedy_plan,
        algorithm_name="Greedy",
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    comparison = create_algorithm_comparison(
        metric_records=[
            greedy_metrics,
        ]
    )

    output_path = save_algorithm_comparison(
        comparison
    )

    print("Optimization metrics calculated successfully.")
    print(f"Saved to: {output_path}")
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()