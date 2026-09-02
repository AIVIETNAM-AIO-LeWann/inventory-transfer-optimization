"""Tests for the forecast optimization pipeline."""

import pandas as pd
import pytest

import src.optimization_pipeline as pipeline_module
from src.config import (
    GENETIC_ALGORITHM_OPTIMIZER,
    GREEDY_OPTIMIZER,
    LINEAR_PROGRAMMING_OPTIMIZER,
    MOVING_AVERAGE_METHOD,
)
from src.data_loader import ProjectData
from src.optimization_pipeline import (
    OptimizationPipelineResult,
    run_optimization_pipeline,
    run_selected_optimizer,
    validate_optimizer_name,
)


@pytest.mark.parametrize(
    "optimizer_name",
    [
        GREEDY_OPTIMIZER,
        LINEAR_PROGRAMMING_OPTIMIZER,
        GENETIC_ALGORITHM_OPTIMIZER,
    ],
)
def test_validate_optimizer_name_accepts_supported_names(
    optimizer_name: str,
) -> None:
    """Supported optimizer names should be accepted."""

    result = validate_optimizer_name(
        optimizer_name
    )

    assert result == optimizer_name


def test_validate_optimizer_name_normalizes_input() -> None:
    """Optimizer names should be stripped and lowercased."""

    result = validate_optimizer_name(
        "  GREEDY  "
    )

    assert result == GREEDY_OPTIMIZER


def test_validate_optimizer_name_rejects_unsupported_name() -> None:
    """Unsupported optimizer names should be rejected."""

    with pytest.raises(
        ValueError,
        match="Unsupported optimizer",
    ):
        validate_optimizer_name(
            "unknown_optimizer"
        )


def test_validate_optimizer_name_rejects_non_string() -> None:
    """Optimizer names must be strings."""

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        validate_optimizer_name(123)


@pytest.mark.parametrize(
    (
        "optimizer_name",
        "optimizer_function_name",
    ),
    [
        (
            GREEDY_OPTIMIZER,
            "optimize_greedy",
        ),
        (
            LINEAR_PROGRAMMING_OPTIMIZER,
            "optimize_linear_programming",
        ),
        (
            GENETIC_ALGORITHM_OPTIMIZER,
            "optimize_genetic_algorithm",
        ),
    ],
)
def test_run_selected_optimizer_dispatches_correctly(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_name: str,
    optimizer_function_name: str,
) -> None:
    """The dispatcher should call the selected optimizer."""

    inventory_analysis = pd.DataFrame(
        {
            "data": ["inventory"],
        }
    )

    route_analysis = pd.DataFrame(
        {
            "data": ["route"],
        }
    )

    expected_plan = pd.DataFrame(
        {
            "optimizer": [optimizer_name],
        }
    )

    def fake_optimizer(
        inventory_analysis: pd.DataFrame,
        route_analysis: pd.DataFrame,
    ) -> pd.DataFrame:
        assert not inventory_analysis.empty
        assert not route_analysis.empty

        return expected_plan

    monkeypatch.setattr(
        pipeline_module,
        optimizer_function_name,
        fake_optimizer,
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

    assert selected_optimizer == optimizer_name
    assert transfer_plan is expected_plan
    assert execution_time_seconds >= 0


def test_run_optimization_pipeline_connects_all_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline should pass data through every step."""

    stores = pd.DataFrame(
        {
            "data": ["stores"],
        }
    )

    products = pd.DataFrame(
        {
            "data": ["products"],
        }
    )

    sales = pd.DataFrame(
        {
            "data": ["sales"],
        }
    )

    inventory = pd.DataFrame(
        {
            "data": ["inventory"],
        }
    )

    distance_matrix = pd.DataFrame(
        {
            "data": ["distance"],
        }
    )

    duration_matrix = pd.DataFrame(
        {
            "data": ["duration"],
        }
    )

    transport_cost_matrix = pd.DataFrame(
        {
            "data": ["transport_cost"],
        }
    )

    project_data = ProjectData(
        stores=stores,
        products=products,
        sales=sales,
        inventory=inventory,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=(
            transport_cost_matrix
        ),
    )

    daily_forecast = pd.DataFrame(
        {
            "forecast_day": [1, 7],
            "method": [
                MOVING_AVERAGE_METHOD,
                MOVING_AVERAGE_METHOD,
            ],
        }
    )

    inventory_analysis = pd.DataFrame(
        {
            "data": ["inventory_analysis"],
        }
    )

    route_analysis = pd.DataFrame(
        {
            "data": ["route_analysis"],
        }
    )

    transfer_plan = pd.DataFrame(
        {
            "data": ["transfer_plan"],
        }
    )

    expected_metrics = {
        "algorithm": "Greedy",
        "execution_time_seconds": 0.25,
    }

    received_arguments = {}

    def fake_forecast(
        sales: pd.DataFrame,
        requested_horizon_days: int,
        method: str,
        moving_average_window_days: int,
    ) -> pd.DataFrame:
        received_arguments["sales"] = sales
        received_arguments[
            "requested_horizon_days"
        ] = requested_horizon_days
        received_arguments["method"] = method
        received_arguments[
            "moving_average_window_days"
        ] = moving_average_window_days

        return daily_forecast

    def fake_inventory_analysis(
        inventory: pd.DataFrame,
        daily_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        received_arguments["inventory"] = inventory
        received_arguments[
            "daily_forecast"
        ] = daily_forecast

        return inventory_analysis

    def fake_route_analysis(
        stores: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        duration_matrix: pd.DataFrame,
        transport_cost_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        received_arguments["stores"] = stores
        received_arguments[
            "distance_matrix"
        ] = distance_matrix
        received_arguments[
            "duration_matrix"
        ] = duration_matrix
        received_arguments[
            "transport_cost_matrix"
        ] = transport_cost_matrix

        return route_analysis

    def fake_run_optimizer(
        inventory_analysis: pd.DataFrame,
        route_analysis: pd.DataFrame,
        optimizer_name: str,
    ) -> tuple[str, pd.DataFrame, float]:
        received_arguments[
            "optimizer_inventory_analysis"
        ] = inventory_analysis
        received_arguments[
            "optimizer_route_analysis"
        ] = route_analysis
        received_arguments[
            "optimizer_name"
        ] = optimizer_name

        return (
            GREEDY_OPTIMIZER,
            transfer_plan,
            0.25,
        )

    def fake_calculate_metrics(
        inventory_analysis: pd.DataFrame,
        transfer_plan: pd.DataFrame,
        algorithm_name: str,
        execution_time_seconds: float,
    ) -> dict[str, str | float]:
        received_arguments[
            "metrics_inventory_analysis"
        ] = inventory_analysis
        received_arguments[
            "metrics_transfer_plan"
        ] = transfer_plan
        received_arguments[
            "algorithm_name"
        ] = algorithm_name
        received_arguments[
            "execution_time_seconds"
        ] = execution_time_seconds

        return expected_metrics

    monkeypatch.setattr(
        pipeline_module,
        "forecast_demand_for_optimization",
        fake_forecast,
    )

    monkeypatch.setattr(
        pipeline_module,
        "analyze_inventory_with_forecast",
        fake_inventory_analysis,
    )

    monkeypatch.setattr(
        pipeline_module,
        "analyze_routes",
        fake_route_analysis,
    )

    monkeypatch.setattr(
        pipeline_module,
        "run_selected_optimizer",
        fake_run_optimizer,
    )

    monkeypatch.setattr(
        pipeline_module,
        "calculate_plan_metrics",
        fake_calculate_metrics,
    )

    result = run_optimization_pipeline(
        project_data=project_data,
        requested_horizon_days=5,
        forecast_method=MOVING_AVERAGE_METHOD,
        optimizer_name="  GREEDY  ",
        moving_average_window_days=3,
    )

    assert isinstance(
        result,
        OptimizationPipelineResult,
    )

    assert result.requested_horizon_days == 5
    assert result.replenishment_horizon_days == 7
    assert result.forecast_method == (
        MOVING_AVERAGE_METHOD
    )
    assert result.optimizer_name == (
        GREEDY_OPTIMIZER
    )

    assert result.daily_forecast is daily_forecast
    assert result.inventory_analysis is (
        inventory_analysis
    )
    assert result.route_analysis is route_analysis
    assert result.transfer_plan is transfer_plan
    assert result.metrics is expected_metrics

    assert received_arguments["sales"] is sales
    assert received_arguments["inventory"] is inventory
    assert received_arguments["stores"] is stores

    assert received_arguments[
        "requested_horizon_days"
    ] == 5

    assert received_arguments["method"] == (
        MOVING_AVERAGE_METHOD
    )

    assert received_arguments[
        "moving_average_window_days"
    ] == 3

    assert received_arguments[
        "optimizer_inventory_analysis"
    ] is inventory_analysis

    assert received_arguments[
        "optimizer_route_analysis"
    ] is route_analysis

    assert received_arguments[
        "algorithm_name"
    ] == "Greedy"

    assert received_arguments[
        "execution_time_seconds"
    ] == pytest.approx(0.25)
    