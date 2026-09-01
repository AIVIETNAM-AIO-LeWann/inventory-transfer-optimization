"""Tests for optimization performance metrics."""

import pandas as pd
import pytest

from src.config import (
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
)
from src.metrics import (
    calculate_plan_metrics,
    create_algorithm_comparison,
    save_algorithm_comparison,
)


@pytest.fixture
def sample_inventory_analysis() -> pd.DataFrame:
    """Create inventory limits for metrics tests."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "shortage_quantity": 12,
                "excess_quantity": 0,
            },
            {
                "store_id": "S002",
                "product_id": "P001",
                "shortage_quantity": 0,
                "excess_quantity": 5,
            },
            {
                "store_id": "S003",
                "product_id": "P001",
                "shortage_quantity": 0,
                "excess_quantity": 10,
            },
        ]
    )


@pytest.fixture
def sample_transfer_plan() -> pd.DataFrame:
    """Create a valid transfer plan."""

    return pd.DataFrame(
        [
            {
                "transfer_id": "T0001",
                "product_id": "P001",
                "from_store_id": "S002",
                "to_store_id": "S001",
                "quantity": 5,
                "route_type": INTRA_CITY_ROUTE,
                "distance_km": 10.0,
                "lead_time_minutes": 90.0,
                "transport_cost_per_unit": 100.0,
                "total_transport_cost": 500.0,
            },
            {
                "transfer_id": "T0002",
                "product_id": "P001",
                "from_store_id": "S003",
                "to_store_id": "S001",
                "quantity": 7,
                "route_type": INTER_CITY_ROUTE,
                "distance_km": 100.0,
                "lead_time_minutes": 600.0,
                "transport_cost_per_unit": 50.0,
                "total_transport_cost": 350.0,
            },
        ]
    )


def test_calculate_plan_metrics(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """Metrics should correctly summarize a transfer plan."""

    metrics = calculate_plan_metrics(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=sample_transfer_plan,
        algorithm_name="Greedy",
        execution_time_seconds=0.1234567,
    )

    assert metrics["algorithm"] == "Greedy"
    assert metrics["transfer_count"] == 2
    assert metrics["total_shortage"] == 12
    assert metrics["total_excess"] == 15
    assert metrics["transferred_quantity"] == 12
    assert metrics["remaining_shortage"] == 0
    assert metrics["remaining_excess"] == 3

    assert metrics["shortage_resolution_rate"] == 1.0
    assert metrics["excess_utilization_rate"] == 0.8

    assert metrics["intra_city_quantity"] == 5
    assert metrics["inter_city_quantity"] == 7

    assert metrics["intra_city_quantity_rate"] == 0.4167
    assert metrics["inter_city_quantity_rate"] == 0.5833

    assert metrics["total_transport_cost"] == 850.0
    assert metrics["average_cost_per_unit"] == 70.83

    assert metrics["weighted_average_distance_km"] == 62.5

    assert (
        metrics["weighted_average_lead_time_minutes"]
        == 387.5
    )

    assert metrics["execution_time_seconds"] == 0.123457


def test_calculate_metrics_for_empty_plan(
    sample_inventory_analysis: pd.DataFrame,
) -> None:
    """An empty plan should leave all shortage unresolved."""

    empty_plan = pd.DataFrame(
        columns=[
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
        ]
    )

    metrics = calculate_plan_metrics(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=empty_plan,
        algorithm_name="Empty Plan",
        execution_time_seconds=0.01,
    )

    assert metrics["transfer_count"] == 0
    assert metrics["transferred_quantity"] == 0
    assert metrics["remaining_shortage"] == 12
    assert metrics["remaining_excess"] == 15
    assert metrics["shortage_resolution_rate"] == 0.0
    assert metrics["excess_utilization_rate"] == 0.0
    assert metrics["total_transport_cost"] == 0.0


def test_metrics_rejects_incorrect_transport_cost(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """Total cost must equal quantity multiplied by unit cost."""

    invalid_plan = sample_transfer_plan.copy()

    invalid_plan.loc[
        0,
        "total_transport_cost",
    ] = 999.0

    with pytest.raises(
        ValueError,
        match="total_transport_cost does not match",
    ):
        calculate_plan_metrics(
            inventory_analysis=sample_inventory_analysis,
            transfer_plan=invalid_plan,
            algorithm_name="Invalid Plan",
            execution_time_seconds=0.01,
        )


def test_metrics_rejects_excessive_source_usage(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """A source cannot transfer more than its available excess."""

    invalid_plan = sample_transfer_plan.copy()

    invalid_plan.loc[0, "quantity"] = 6
    invalid_plan.loc[0, "total_transport_cost"] = 600.0

    with pytest.raises(
        ValueError,
        match="more than its available excess",
    ):
        calculate_plan_metrics(
            inventory_analysis=sample_inventory_analysis,
            transfer_plan=invalid_plan,
            algorithm_name="Invalid Plan",
            execution_time_seconds=0.01,
        )


def test_metrics_rejects_duplicate_transfer_ids(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """Every transfer ID must be unique."""

    invalid_plan = sample_transfer_plan.copy()

    invalid_plan.loc[1, "transfer_id"] = "T0001"

    with pytest.raises(
        ValueError,
        match="transfer_id values must be unique",
    ):
        calculate_plan_metrics(
            inventory_analysis=sample_inventory_analysis,
            transfer_plan=invalid_plan,
            algorithm_name="Invalid Plan",
            execution_time_seconds=0.01,
        )


def test_create_algorithm_comparison(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """Algorithms should be ranked by resolution and cost."""

    greedy_metrics = calculate_plan_metrics(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=sample_transfer_plan,
        algorithm_name="Greedy",
        execution_time_seconds=0.10,
    )

    linear_programming_metrics = {
        **greedy_metrics,
        "algorithm": "Linear Programming",
        "total_transport_cost": 700.0,
        "execution_time_seconds": 0.50,
    }

    genetic_algorithm_metrics = {
        **greedy_metrics,
        "algorithm": "Genetic Algorithm",
        "shortage_resolution_rate": 0.90,
        "total_transport_cost": 600.0,
        "execution_time_seconds": 2.0,
    }

    comparison = create_algorithm_comparison(
        metric_records=[
            greedy_metrics,
            linear_programming_metrics,
            genetic_algorithm_metrics,
        ]
    )

    assert comparison["rank"].tolist() == [1, 2, 3]

    assert comparison["algorithm"].tolist() == [
        "Linear Programming",
        "Greedy",
        "Genetic Algorithm",
    ]


def test_comparison_rejects_duplicate_algorithm_names(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
) -> None:
    """Algorithm names in a comparison must be unique."""

    metrics = calculate_plan_metrics(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=sample_transfer_plan,
        algorithm_name="Greedy",
        execution_time_seconds=0.10,
    )

    with pytest.raises(
        ValueError,
        match="Algorithm names must be unique",
    ):
        create_algorithm_comparison(
            metric_records=[
                metrics,
                metrics.copy(),
            ]
        )


def test_save_algorithm_comparison(
    sample_inventory_analysis: pd.DataFrame,
    sample_transfer_plan: pd.DataFrame,
    tmp_path,
) -> None:
    """Algorithm comparison should be saved as CSV."""

    metrics = calculate_plan_metrics(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=sample_transfer_plan,
        algorithm_name="Greedy",
        execution_time_seconds=0.10,
    )

    comparison = create_algorithm_comparison(
        metric_records=[metrics]
    )

    output_path = (
        tmp_path / "algorithm_comparison.csv"
    )

    saved_path = save_algorithm_comparison(
        comparison=comparison,
        output_path=output_path,
    )

    assert saved_path.exists()

    loaded_comparison = pd.read_csv(saved_path)

    assert len(loaded_comparison) == 1
    assert loaded_comparison.loc[0, "rank"] == 1
    assert (
        loaded_comparison.loc[0, "algorithm"]
        == "Greedy"
    )