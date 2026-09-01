"""Tests for the greedy inventory transfer optimizer."""

import pandas as pd
import pytest

from src.inventory_analyzer import (
    EXCESS_STATUS,
    SHORTAGE_STATUS,
)
from src.optimizers.greedy import (
    create_greedy_summary,
    optimize_greedy,
)
from src.config import (
    INTER_CITY_ROUTE,
    INTRA_CITY_ROUTE,
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
    """Create one intra-city and one inter-city route."""

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


def test_greedy_prioritizes_intra_city_route(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Greedy should use the intra-city source first."""

    plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    assert len(plan) == 2

    first_transfer = plan.iloc[0]
    second_transfer = plan.iloc[1]

    assert first_transfer["transfer_id"] == "T0001"
    assert first_transfer["from_store_id"] == "S002"
    assert first_transfer["to_store_id"] == "S001"
    assert first_transfer["quantity"] == 5
    assert first_transfer["route_type"] == INTRA_CITY_ROUTE

    assert second_transfer["transfer_id"] == "T0002"
    assert second_transfer["from_store_id"] == "S003"
    assert second_transfer["to_store_id"] == "S001"
    assert second_transfer["quantity"] == 7
    assert second_transfer["route_type"] == INTER_CITY_ROUTE


def test_greedy_does_not_exceed_inventory_limits(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Transfers must not exceed shortage or available excess."""

    plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    assert plan["quantity"].sum() == 12

    first_transfer = plan.iloc[0]
    second_transfer = plan.iloc[1]

    assert first_transfer["source_excess_before"] == 5
    assert first_transfer["source_excess_after"] == 0

    assert (
        first_transfer["destination_shortage_before"]
        == 12
    )
    assert (
        first_transfer["destination_shortage_after"]
        == 7
    )

    assert second_transfer["source_excess_before"] == 10
    assert second_transfer["source_excess_after"] == 3

    assert (
        second_transfer["destination_shortage_before"]
        == 7
    )
    assert (
        second_transfer["destination_shortage_after"]
        == 0
    )


def test_greedy_calculates_transport_cost(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Each transfer cost should equal quantity times unit cost."""

    plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    first_transfer = plan.iloc[0]
    second_transfer = plan.iloc[1]

    assert first_transfer["total_transport_cost"] == 500.0
    assert second_transfer["total_transport_cost"] == 350.0

    assert (
        plan["total_transport_cost"].sum()
        == 850.0
    )


def test_greedy_ignores_disallowed_routes(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Greedy should not use routes that are not allowed."""

    disallowed_routes = sample_route_analysis.copy()
    disallowed_routes["is_allowed"] = False

    plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=disallowed_routes,
    )

    assert plan.empty


def test_greedy_requires_the_same_product(
    sample_route_analysis: pd.DataFrame,
) -> None:
    """A source cannot satisfy shortage for another product."""

    inventory_analysis = pd.DataFrame(
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
                "product_id": "P002",
                "inventory_days": 25.0,
                "status": EXCESS_STATUS,
                "shortage_quantity": 0,
                "excess_quantity": 20,
            },
        ]
    )

    plan = optimize_greedy(
        inventory_analysis=inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    assert plan.empty


def test_create_greedy_summary(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Greedy summary should calculate correct totals."""

    plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    summary = create_greedy_summary(
        inventory_analysis=sample_inventory_analysis,
        transfer_plan=plan,
    )

    assert summary["transfer_count"] == 2
    assert summary["total_shortage"] == 12
    assert summary["total_excess"] == 15
    assert summary["transferred_quantity"] == 12
    assert summary["remaining_shortage"] == 0
    assert summary["remaining_excess"] == 3
    assert summary["intra_city_quantity"] == 5
    assert summary["inter_city_quantity"] == 7
    assert summary["total_transport_cost"] == 850.0
    assert summary["shortage_resolution_rate"] == 1.0


def test_greedy_rejects_duplicate_routes(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Duplicate source-destination routes should be rejected."""

    duplicate_routes = pd.concat(
        [
            sample_route_analysis,
            sample_route_analysis.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate routes",
    ):
        optimize_greedy(
            inventory_analysis=sample_inventory_analysis,
            route_analysis=duplicate_routes,
        )