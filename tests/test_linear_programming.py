"""Tests for the linear programming optimizer."""

import pandas as pd
import pytest

from src.config import (
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
)
from src.inventory_analyzer import (
    EXCESS_STATUS,
    SHORTAGE_STATUS,
)
from src.optimizers.greedy import optimize_greedy
from src.optimizers.linear_programming import (
    build_transfer_candidates,
    optimize_linear_programming,
    save_transfer_plan,
)


@pytest.fixture
def sample_inventory_analysis() -> pd.DataFrame:
    """Create one shortage and two excess inventory rows."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "inventory_days": 2.0,
                "status": SHORTAGE_STATUS,
                "shortage_quantity": 12,
                "excess_quantity": 0,
            },
            {
                "store_id": "S002",
                "product_id": "P001",
                "inventory_days": 25.0,
                "status": EXCESS_STATUS,
                "shortage_quantity": 0,
                "excess_quantity": 5,
            },
            {
                "store_id": "S003",
                "product_id": "P001",
                "inventory_days": 30.0,
                "status": EXCESS_STATUS,
                "shortage_quantity": 0,
                "excess_quantity": 10,
            },
        ]
    )


@pytest.fixture
def sample_route_analysis() -> pd.DataFrame:
    """Create two allowed transfer routes."""

    return pd.DataFrame(
        [
            {
                "from_store_id": "S002",
                "to_store_id": "S001",
                "route_type": INTRA_CITY_ROUTE,
                "priority_rank": 1,
                "distance_km": 10.0,
                "lead_time_minutes": 90.0,
                "transport_cost_per_unit": 100.0,
                "is_allowed": True,
            },
            {
                "from_store_id": "S003",
                "to_store_id": "S001",
                "route_type": INTER_CITY_ROUTE,
                "priority_rank": 2,
                "distance_km": 100.0,
                "lead_time_minutes": 600.0,
                "transport_cost_per_unit": 50.0,
                "is_allowed": True,
            },
        ]
    )


def test_build_transfer_candidates(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Candidates should connect matching products through allowed routes."""

    candidates = build_transfer_candidates(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    assert len(candidates) == 2

    source_ids = {
        candidate["from_store_id"]
        for candidate in candidates
    }

    assert source_ids == {"S002", "S003"}

    assert all(
        candidate["to_store_id"] == "S001"
        for candidate in candidates
    )

    assert all(
        candidate["product_id"] == "P001"
        for candidate in candidates
    )


def test_linear_programming_minimizes_cost(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Linear programming should use the cheapest source first."""

    plan = optimize_linear_programming(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    assert len(plan) == 2
    assert plan["quantity"].sum() == 12
    assert plan["total_transport_cost"].sum() == 700.0

    cheapest_transfer = plan.iloc[0]
    remaining_transfer = plan.iloc[1]

    assert cheapest_transfer["from_store_id"] == "S003"
    assert cheapest_transfer["quantity"] == 10
    assert cheapest_transfer["total_transport_cost"] == 500.0

    assert remaining_transfer["from_store_id"] == "S002"
    assert remaining_transfer["quantity"] == 2
    assert remaining_transfer["total_transport_cost"] == 200.0


def test_linear_programming_respects_inventory_limits(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """The solution must respect excess and shortage quantities."""

    plan = optimize_linear_programming(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    source_totals = plan.groupby(
        "from_store_id"
    )["quantity"].sum()

    destination_totals = plan.groupby(
        "to_store_id"
    )["quantity"].sum()

    assert source_totals["S002"] <= 5
    assert source_totals["S003"] <= 10
    assert destination_totals["S001"] <= 12

    assert plan.iloc[0]["source_excess_before"] == 10
    assert plan.iloc[0]["source_excess_after"] == 0

    assert (
        plan.iloc[-1]["destination_shortage_after"]
        == 0
    )


def test_linear_programming_ignores_disallowed_routes(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Disallowed routes must not appear in the solution."""

    disallowed_routes = sample_route_analysis.copy()
    disallowed_routes["is_allowed"] = False

    plan = optimize_linear_programming(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=disallowed_routes,
    )

    assert plan.empty


def test_linear_programming_rejects_low_penalty(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Unmet-shortage penalty must exceed route costs."""

    with pytest.raises(
        ValueError,
        match=(
            "unmet_penalty_per_unit must be greater"
        ),
    ):
        optimize_linear_programming(
            inventory_analysis=sample_inventory_analysis,
            route_analysis=sample_route_analysis,
            unmet_penalty_per_unit=50.0,
        )


def test_linear_programming_is_cheaper_than_greedy(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """LP should find a cheaper solution for this example."""

    greedy_plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    linear_programming_plan = (
        optimize_linear_programming(
            inventory_analysis=(
                sample_inventory_analysis
            ),
            route_analysis=sample_route_analysis,
        )
    )

    greedy_cost = float(
        greedy_plan["total_transport_cost"].sum()
    )

    linear_programming_cost = float(
        linear_programming_plan[
            "total_transport_cost"
        ].sum()
    )

    assert greedy_cost == 850.0
    assert linear_programming_cost == 700.0
    assert linear_programming_cost < greedy_cost


def test_save_linear_programming_plan(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
    tmp_path,
) -> None:
    """The transfer plan should be saved as CSV."""

    plan = optimize_linear_programming(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    output_path = (
        tmp_path
        / "linear_programming_transfer_plan.csv"
    )

    saved_path = save_transfer_plan(
        transfer_plan=plan,
        output_path=output_path,
    )

    assert saved_path.exists()

    loaded_plan = pd.read_csv(saved_path)

    assert len(loaded_plan) == 2
    assert loaded_plan["quantity"].sum() == 12

    assert (
        loaded_plan["total_transport_cost"].sum()
        == 700.0
    )