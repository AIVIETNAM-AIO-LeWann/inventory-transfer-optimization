"""Run inventory optimization using machine-learning forecasts."""

from src.dashboard.constants import (
    OPTIMIZER_DISPLAY_NAMES,
)
from src.dashboard.ml_forecaster import (
    forecast_machine_learning_demand,
)
from src.dashboard.model_artifacts import (
    ForecastModelArtifact,
    validate_model_artifact,
)
from src.data_loader import ProjectData
from src.inventory_analyzer import (
    analyze_inventory_with_forecast,
)
from src.metrics import calculate_plan_metrics
from src.optimization_pipeline import (
    OptimizationPipelineResult,
    run_selected_optimizer,
)
from src.route_analyzer import analyze_routes


def run_ml_optimization_pipeline(
    project_data: ProjectData,
    artifact: ForecastModelArtifact,
    dataset_fingerprint: str,
    requested_horizon_days: int,
    optimizer_name: str,
) -> OptimizationPipelineResult:
    """Run forecasting and optimization with a trained model."""

    if not isinstance(
        project_data,
        ProjectData,
    ):
        raise TypeError(
            "project_data must be ProjectData."
        )

    validate_model_artifact(artifact)

    daily_forecast = (
        forecast_machine_learning_demand(
            project_data=project_data,
            artifact=artifact,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
            requested_horizon_days=(
                requested_horizon_days
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
        algorithm_name=(
            optimizer_display_name
        ),
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    replenishment_horizon_days = int(
        daily_forecast["forecast_day"].max()
    )

    return OptimizationPipelineResult(
        requested_horizon_days=(
            requested_horizon_days
        ),
        replenishment_horizon_days=(
            replenishment_horizon_days
        ),
        forecast_method=artifact.method,
        optimizer_name=selected_optimizer,
        daily_forecast=daily_forecast,
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
        transfer_plan=transfer_plan,
        metrics=metrics,
    )