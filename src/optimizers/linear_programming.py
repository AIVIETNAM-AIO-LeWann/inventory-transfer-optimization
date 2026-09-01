"""Optimize inventory transfers using linear programming."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import pulp

from src.config import (
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
    LINEAR_PROGRAMMING_TRANSFER_PLAN_FILE,
    UNMET_SHORTAGE_PENALTY_PER_UNIT,
)
from src.data_loader import load_all_data
from src.inventory_analyzer import (
    EXCESS_STATUS,
    SHORTAGE_STATUS,
    analyze_inventory,
)
from src.metrics import calculate_plan_metrics
from src.route_analyzer import analyze_routes


INVENTORY_REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
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


def validate_linear_programming_inputs(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
    unmet_penalty_per_unit: float,
) -> None:
    """Validate inputs required by linear programming."""

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

    duplicated_inventory = inventory_analysis.duplicated(
        subset=["store_id", "product_id"],
        keep=False,
    )

    if duplicated_inventory.any():
        raise ValueError(
            "inventory_analysis contains duplicate "
            "store-product pairs."
        )

    duplicated_routes = route_analysis.duplicated(
        subset=["from_store_id", "to_store_id"],
        keep=False,
    )

    if duplicated_routes.any():
        raise ValueError(
            "route_analysis contains duplicate routes."
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

    if unmet_penalty_per_unit <= 0:
        raise ValueError(
            "unmet_penalty_per_unit must be greater than zero."
        )

    allowed_costs = route_analysis.loc[
        route_analysis["is_allowed"],
        "transport_cost_per_unit",
    ]

    if (
        not allowed_costs.empty
        and unmet_penalty_per_unit
        <= float(allowed_costs.max())
    ):
        raise ValueError(
            "unmet_penalty_per_unit must be greater than "
            "every allowed transport cost per unit."
        )


def build_transfer_candidates(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build feasible source-destination-product candidates."""

    shortage_rows = inventory_analysis.loc[
        (
            inventory_analysis["status"]
            == SHORTAGE_STATUS
        )
        & (
            inventory_analysis["shortage_quantity"]
            > 0
        )
    ].copy()

    excess_rows = inventory_analysis.loc[
        (
            inventory_analysis["status"]
            == EXCESS_STATUS
        )
        & (
            inventory_analysis["excess_quantity"]
            > 0
        )
    ].copy()

    allowed_routes = route_analysis.loc[
        route_analysis["is_allowed"]
    ].copy()

    route_lookup = allowed_routes.set_index(
        ["from_store_id", "to_store_id"],
        drop=False,
    )

    candidates: list[dict[str, object]] = []

    for shortage in shortage_rows.itertuples(index=False):
        destination_store_id = str(
            shortage.store_id
        )
        product_id = str(shortage.product_id)

        product_sources = excess_rows.loc[
            excess_rows["product_id"].astype(str)
            == product_id
        ]

        for source in product_sources.itertuples(
            index=False
        ):
            source_store_id = str(source.store_id)

            route_key = (
                source_store_id,
                destination_store_id,
            )

            if route_key not in route_lookup.index:
                continue

            route = route_lookup.loc[route_key]

            candidate_id = len(candidates)

            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "product_id": product_id,
                    "from_store_id": source_store_id,
                    "to_store_id": destination_store_id,
                    "source_excess": int(
                        source.excess_quantity
                    ),
                    "destination_shortage": int(
                        shortage.shortage_quantity
                    ),
                    "route_type": route["route_type"],
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
                        route["transport_cost_per_unit"]
                    ),
                }
            )

    return candidates


def create_solver(
    show_solver_messages: bool = False,
):
    """Create an available CBC solver."""

    available_solvers = set(
        pulp.listSolvers(onlyAvailable=True)
    )

    if "COIN_CMD" not in available_solvers:
        raise RuntimeError(
            "COIN_CMD is not available. "
            "Install CBC with: "
            "python -m pip install \"pulp[cbc]==3.3.2\""
        )

    return pulp.COIN_CMD(
        msg=show_solver_messages
    )

def optimize_linear_programming(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
    unmet_penalty_per_unit: float = (
        UNMET_SHORTAGE_PENALTY_PER_UNIT
    ),
    show_solver_messages: bool = False,
) -> pd.DataFrame:
    """Create a minimum-cost inventory transfer plan."""

    validate_linear_programming_inputs(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
        unmet_penalty_per_unit=(
            unmet_penalty_per_unit
        ),
    )

    candidates = build_transfer_candidates(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    shortage_rows = inventory_analysis.loc[
        (
            inventory_analysis["status"]
            == SHORTAGE_STATUS
        )
        & (
            inventory_analysis["shortage_quantity"]
            > 0
        )
    ].copy()

    model = pulp.LpProblem(
        name="InventoryTransferOptimization",
        sense=pulp.LpMinimize,
    )

    transfer_variables = {
        candidate["candidate_id"]: model.add_variable(
            name=(
                f"transfer_{candidate['candidate_id']:05d}"
            ),
            lowBound=0,
            upBound=min(
                candidate["source_excess"],
                candidate["destination_shortage"],
            ),
            cat=pulp.LpInteger,
        )
        for candidate in candidates
    }

    shortage_limits = {
        (
            str(row.store_id),
            str(row.product_id),
        ): int(row.shortage_quantity)
        for row in shortage_rows.itertuples(index=False)
    }

    unmet_variables = {
        shortage_key: model.add_variable(
            name=f"unmet_{index:05d}",
            lowBound=0,
            upBound=shortage_quantity,
            cat=pulp.LpInteger,
        )
        for index, (
            shortage_key,
            shortage_quantity,
        ) in enumerate(shortage_limits.items())
    }

    source_limits = {
        (
            str(row.store_id),
            str(row.product_id),
        ): int(row.excess_quantity)
        for row in inventory_analysis.loc[
            (
                inventory_analysis["status"]
                == EXCESS_STATUS
            )
            & (
                inventory_analysis["excess_quantity"]
                > 0
            )
        ].itertuples(index=False)
    }

    for source_key, source_limit in (
        source_limits.items()
    ):
        source_candidate_ids = [
            candidate["candidate_id"]
            for candidate in candidates
            if (
                candidate["from_store_id"],
                candidate["product_id"],
            )
            == source_key
        ]

        model += (
            pulp.lpSum(
                transfer_variables[candidate_id]
                for candidate_id
                in source_candidate_ids
            )
            <= source_limit,
            (
                "source_limit_"
                f"{source_key[0]}_{source_key[1]}"
            ),
        )

    for shortage_key, shortage_limit in (
        shortage_limits.items()
    ):
        destination_candidate_ids = [
            candidate["candidate_id"]
            for candidate in candidates
            if (
                candidate["to_store_id"],
                candidate["product_id"],
            )
            == shortage_key
        ]

        model += (
            pulp.lpSum(
                transfer_variables[candidate_id]
                for candidate_id
                in destination_candidate_ids
            )
            + unmet_variables[shortage_key]
            == shortage_limit,
            (
                "shortage_balance_"
                f"{shortage_key[0]}_{shortage_key[1]}"
            ),
        )

    transport_cost = pulp.lpSum(
        transfer_variables[candidate["candidate_id"]]
        * candidate["transport_cost_per_unit"]
        for candidate in candidates
    )

    unmet_shortage_cost = pulp.lpSum(
        unmet_variable * unmet_penalty_per_unit
        for unmet_variable in unmet_variables.values()
    )

    model += (
        transport_cost + unmet_shortage_cost
    )

    solver = create_solver(
        show_solver_messages=show_solver_messages
    )

    model.solve(solver)

    solver_status = pulp.LpStatus.get(
        model.status,
        str(model.status),
    )

    if solver_status != "Optimal":
        raise RuntimeError(
            "Linear programming did not find an "
            f"optimal solution. Status: {solver_status}"
        )

    allocations: list[dict[str, object]] = []

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]

        variable_value = pulp.value(
            transfer_variables[candidate_id]
        )

        quantity = int(
            round(variable_value or 0)
        )

        if quantity <= 0:
            continue

        allocations.append(
            {
                **candidate,
                "quantity": quantity,
            }
        )

    allocations.sort(
        key=lambda allocation: (
            allocation["product_id"],
            allocation["to_store_id"],
            allocation["transport_cost_per_unit"],
            allocation["lead_time_minutes"],
            allocation["from_store_id"],
        )
    )

    remaining_excess = source_limits.copy()
    remaining_shortage = shortage_limits.copy()

    transfer_records: list[dict[str, object]] = []

    for allocation in allocations:
        source_key = (
            allocation["from_store_id"],
            allocation["product_id"],
        )

        destination_key = (
            allocation["to_store_id"],
            allocation["product_id"],
        )

        quantity = int(allocation["quantity"])

        source_excess_before = (
            remaining_excess[source_key]
        )

        destination_shortage_before = (
            remaining_shortage[destination_key]
        )

        source_excess_after = (
            source_excess_before - quantity
        )

        destination_shortage_after = (
            destination_shortage_before - quantity
        )

        remaining_excess[source_key] = (
            source_excess_after
        )

        remaining_shortage[destination_key] = (
            destination_shortage_after
        )

        transport_cost_per_unit = float(
            allocation["transport_cost_per_unit"]
        )

        total_transport_cost = (
            quantity * transport_cost_per_unit
        )

        transfer_number = (
            len(transfer_records) + 1
        )

        transfer_records.append(
            {
                "transfer_id": (
                    f"LP{transfer_number:04d}"
                ),
                "product_id": (
                    allocation["product_id"]
                ),
                "from_store_id": (
                    allocation["from_store_id"]
                ),
                "to_store_id": (
                    allocation["to_store_id"]
                ),
                "quantity": quantity,
                "route_type": (
                    allocation["route_type"]
                ),
                "distance_km": round(
                    float(allocation["distance_km"]),
                    3,
                ),
                "lead_time_minutes": round(
                    float(
                        allocation[
                            "lead_time_minutes"
                        ]
                    ),
                    2,
                ),
                "transport_cost_per_unit": round(
                    transport_cost_per_unit,
                    2,
                ),
                "total_transport_cost": round(
                    total_transport_cost,
                    2,
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


def save_transfer_plan(
    transfer_plan: pd.DataFrame,
    output_path: str | Path = (
        LINEAR_PROGRAMMING_TRANSFER_PLAN_FILE
    ),
) -> Path:
    """Save the linear programming transfer plan."""

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
    """Run linear programming with project data."""

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

    transfer_plan = optimize_linear_programming(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    execution_time_seconds = (
        perf_counter() - start_time
    )

    metrics = calculate_plan_metrics(
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
        algorithm_name="Linear Programming",
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    output_path = save_transfer_plan(
        transfer_plan=transfer_plan
    )

    print(
        "Linear programming completed successfully."
    )
    print(f"Saved to: {output_path}")
    print()
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()