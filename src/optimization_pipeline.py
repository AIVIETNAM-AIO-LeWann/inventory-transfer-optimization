"""Run demand forecasting and inventory optimization together."""

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from src.config import (
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_FORECAST_METHOD,
    DEFAULT_OPTIMIZER,
    GENETIC_ALGORITHM_OPTIMIZER,
    GREEDY_OPTIMIZER,
    LINEAR_PROGRAMMING_OPTIMIZER,
    MOVING_AVERAGE_WINDOW_DAYS,
    SUPPORTED_OPTIMIZERS,
)
from src.data_loader import (
    ProjectData,
    load_all_data,
)
from src.forecasting.demand_forecaster import (
    forecast_demand_for_optimization,
)
from src.inventory_analyzer import (
    analyze_inventory_with_forecast,
)
from src.metrics import calculate_plan_metrics
from src.optimizers.genetic_algorithm import (
    optimize_genetic_algorithm,
)
from src.optimizers.greedy import optimize_greedy
from src.optimizers.linear_programming import (
    optimize_linear_programming,
)
from src.route_analyzer import analyze_routes


OPTIMIZER_DISPLAY_NAMES = {
    GREEDY_OPTIMIZER: "Greedy",
    LINEAR_PROGRAMMING_OPTIMIZER: (
        "Linear Programming"
    ),
    GENETIC_ALGORITHM_OPTIMIZER: (
        "Genetic Algorithm"
    ),
}


@dataclass
class OptimizationPipelineResult:
    """Store all outputs created by the pipeline."""

    requested_horizon_days: int
    replenishment_horizon_days: int
    forecast_method: str
    optimizer_name: str
    daily_forecast: pd.DataFrame
    inventory_analysis: pd.DataFrame
    route_analysis: pd.DataFrame
    transfer_plan: pd.DataFrame
    metrics: dict[str, str | int | float]


def validate_optimizer_name(
    optimizer_name: str,
) -> str:
    """Validate and normalize an optimizer name."""

    if not isinstance(optimizer_name, str):
        raise TypeError(
            "optimizer_name must be a string."
        )

    normalized_name = (
        optimizer_name.strip().lower()
    )

    if normalized_name not in SUPPORTED_OPTIMIZERS:
        supported_names = ", ".join(
            SUPPORTED_OPTIMIZERS
        )

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers: {supported_names}."
        )

    return normalized_name


def run_selected_optimizer(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
    optimizer_name: str,
) -> tuple[str, pd.DataFrame, float]:
    """Run the selected inventory transfer optimizer."""

    selected_optimizer = validate_optimizer_name(
        optimizer_name
    )

    start_time = perf_counter()

    if selected_optimizer == GREEDY_OPTIMIZER:
        transfer_plan = optimize_greedy(
            inventory_analysis=inventory_analysis,
            route_analysis=route_analysis,
        )

    elif (
        selected_optimizer
        == LINEAR_PROGRAMMING_OPTIMIZER
    ):
        transfer_plan = optimize_linear_programming(
            inventory_analysis=inventory_analysis,
            route_analysis=route_analysis,
        )

    elif (
        selected_optimizer
        == GENETIC_ALGORITHM_OPTIMIZER
    ):
        transfer_plan = optimize_genetic_algorithm(
            inventory_analysis=inventory_analysis,
            route_analysis=route_analysis,
        )

    else:
        raise RuntimeError(
            "The selected optimizer has no "
            "implementation."
        )

    execution_time_seconds = (
        perf_counter() - start_time
    )

    return (
        selected_optimizer,
        transfer_plan,
        execution_time_seconds,
    )


def run_optimization_pipeline(
    project_data: ProjectData,
    requested_horizon_days: int = (
        DEFAULT_FORECAST_HORIZON_DAYS
    ),
    forecast_method: str = DEFAULT_FORECAST_METHOD,
    optimizer_name: str = DEFAULT_OPTIMIZER,
    moving_average_window_days: int = (
        MOVING_AVERAGE_WINDOW_DAYS
    ),
) -> OptimizationPipelineResult:
    """Run forecasting, inventory analysis, and optimization."""

    daily_forecast = (
        forecast_demand_for_optimization(
            sales=project_data.sales,
            requested_horizon_days=(
                requested_horizon_days
            ),
            method=forecast_method,
            moving_average_window_days=(
                moving_average_window_days
            ),
        )
    )

    inventory_analysis = (
        analyze_inventory_with_forecast(
            inventory=project_data.inventory,
            daily_forecast=daily_forecast,
        )
    )

    route_analysis = analyze_routes(
        stores=project_data.stores,
        distance_matrix=(
            project_data.distance_matrix
        ),
        duration_matrix=(
            project_data.duration_matrix
        ),
        transport_cost_matrix=(
            project_data.transport_cost_matrix
        ),
    )

    (
        selected_optimizer,
        transfer_plan,
        execution_time_seconds,
    ) = run_selected_optimizer(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
        optimizer_name=optimizer_name,
    )

    optimizer_display_name = (
        OPTIMIZER_DISPLAY_NAMES[
            selected_optimizer
        ]
    )

    metrics = calculate_plan_metrics(
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
        algorithm_name=optimizer_display_name,
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    replenishment_horizon_days = int(
        daily_forecast["forecast_day"].max()
    )

    selected_forecast_method = str(
        daily_forecast["method"].iloc[0]
    )

    return OptimizationPipelineResult(
        requested_horizon_days=(
            requested_horizon_days
        ),
        replenishment_horizon_days=(
            replenishment_horizon_days
        ),
        forecast_method=(
            selected_forecast_method
        ),
        optimizer_name=selected_optimizer,
        daily_forecast=daily_forecast,
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
        transfer_plan=transfer_plan,
        metrics=metrics,
    )


def main() -> None:
    """Run the default optimization pipeline."""

    project_data = load_all_data()

    result = run_optimization_pipeline(
        project_data=project_data,
        requested_horizon_days=(
            DEFAULT_FORECAST_HORIZON_DAYS
        ),
        forecast_method=DEFAULT_FORECAST_METHOD,
        optimizer_name=DEFAULT_OPTIMIZER,
    )

    print(
        "Forecast optimization pipeline "
        "completed successfully."
    )
    print(
        "Requested horizon: "
        f"{result.requested_horizon_days} days"
    )
    print(
        "Replenishment horizon: "
        f"{result.replenishment_horizon_days} days"
    )
    print(
        f"Forecast method: {result.forecast_method}"
    )
    print(f"Optimizer: {result.optimizer_name}")
    print(
        "Forecast rows: "
        f"{len(result.daily_forecast):,}"
    )
    print(
        "Inventory rows: "
        f"{len(result.inventory_analysis):,}"
    )
    print(
        "Transfer rows: "
        f"{len(result.transfer_plan):,}"
    )
    print()
    print(pd.Series(result.metrics).to_string())


if __name__ == "__main__":
    main()