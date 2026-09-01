"""Tests for route analysis functions."""

import pandas as pd
import pytest

from src.config import (
    INTER_CITY_HANDLING_TIME_MINUTES,
    INTER_CITY_ROUTE,
    INTRA_CITY_HANDLING_TIME_MINUTES,
    INTRA_CITY_ROUTE,
)
from src.route_analyzer import (
    analyze_routes,
    classify_route,
    create_route_summary,
    get_rejection_reason,
    get_route_policy,
)


@pytest.fixture
def sample_stores() -> pd.DataFrame:
    """Create stores in the same and different cities."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "store_name": "Hanoi Store 1",
                "city": "Hanoi",
                "latitude": 21.0285,
                "longitude": 105.8542,
            },
            {
                "store_id": "S002",
                "store_name": "Hanoi Store 2",
                "city": "Hanoi",
                "latitude": 21.0368,
                "longitude": 105.8342,
            },
            {
                "store_id": "S003",
                "store_name": "Ho Chi Minh Store",
                "city": "Ho Chi Minh City",
                "latitude": 10.7769,
                "longitude": 106.7009,
            },
        ]
    )


@pytest.fixture
def route_matrices() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create distance, duration, and transport cost matrices."""

    store_ids = ["S001", "S002", "S003"]

    distance_matrix = pd.DataFrame(
        [
            [0.0, 10.0, 1700.0],
            [10.0, 0.0, 1500.0],
            [1700.0, 1500.0, 0.0],
        ],
        index=store_ids,
        columns=store_ids,
    )

    duration_matrix = pd.DataFrame(
        [
            [0.0, 60.0, 1800.0],
            [60.0, 0.0, 1900.0],
            [1800.0, 1900.0, 0.0],
        ],
        index=store_ids,
        columns=store_ids,
    )

    transport_cost_matrix = pd.DataFrame(
        [
            [0.0, 1000.0, 170000.0],
            [1000.0, 0.0, 150000.0],
            [170000.0, 150000.0, 0.0],
        ],
        index=store_ids,
        columns=store_ids,
    )

    return (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    )


def test_classify_route() -> None:
    """Routes should be classified by city."""

    assert (
        classify_route("Hanoi", "Hanoi")
        == INTRA_CITY_ROUTE
    )

    assert (
        classify_route(
            "Hanoi",
            "Ho Chi Minh City",
        )
        == INTER_CITY_ROUTE
    )


def test_get_route_policy() -> None:
    """Each route type should use the correct policy."""

    intra_policy = get_route_policy(
        INTRA_CITY_ROUTE
    )

    inter_policy = get_route_policy(
        INTER_CITY_ROUTE
    )

    assert intra_policy[0] == (
        INTRA_CITY_HANDLING_TIME_MINUTES
    )
    assert intra_policy[2] == 1

    assert inter_policy[0] == (
        INTER_CITY_HANDLING_TIME_MINUTES
    )
    assert inter_policy[2] == 2


def test_get_route_policy_rejects_unknown_type() -> None:
    """Unknown route types should be rejected."""

    with pytest.raises(
        ValueError,
        match="Unsupported route type",
    ):
        get_route_policy("unknown_route")


def test_analyze_routes_calculates_route_information(
    sample_stores: pd.DataFrame,
    route_matrices: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """Route analysis should calculate lead time and permissions."""

    (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    ) = route_matrices

    result = analyze_routes(
        stores=sample_stores,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    assert len(result) == 6

    assert not (
        result["from_store_id"]
        == result["to_store_id"]
    ).any()

    intra_route = result.loc[
        (result["from_store_id"] == "S001")
        & (result["to_store_id"] == "S002")
    ].iloc[0]

    assert intra_route["route_type"] == INTRA_CITY_ROUTE
    assert intra_route["priority_rank"] == 1
    assert intra_route["distance_km"] == 10.0
    assert intra_route["driving_time_minutes"] == 60.0

    assert intra_route["handling_time_minutes"] == (
        INTRA_CITY_HANDLING_TIME_MINUTES
    )

    assert intra_route["lead_time_minutes"] == (
        60.0 + INTRA_CITY_HANDLING_TIME_MINUTES
    )

    assert bool(intra_route["is_allowed"])
    assert intra_route["rejection_reason"] == ""


def test_route_can_be_rejected_by_distance(
    sample_stores: pd.DataFrame,
    route_matrices: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """A route over the distance limit should be rejected."""

    (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    ) = route_matrices

    result = analyze_routes(
        stores=sample_stores,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    route = result.loc[
        (result["from_store_id"] == "S001")
        & (result["to_store_id"] == "S003")
    ].iloc[0]

    assert not bool(route["distance_allowed"])
    assert bool(route["time_allowed"])
    assert not bool(route["is_allowed"])
    assert route["rejection_reason"] == "distance_limit"


def test_route_can_be_rejected_by_lead_time(
    sample_stores: pd.DataFrame,
    route_matrices: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """A route over the lead-time limit should be rejected."""

    (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    ) = route_matrices

    result = analyze_routes(
        stores=sample_stores,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    route = result.loc[
        (result["from_store_id"] == "S002")
        & (result["to_store_id"] == "S003")
    ].iloc[0]

    assert bool(route["distance_allowed"])
    assert not bool(route["time_allowed"])
    assert not bool(route["is_allowed"])
    assert route["rejection_reason"] == "lead_time_limit"


def test_create_route_summary(
    sample_stores: pd.DataFrame,
    route_matrices: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    """Route summary should aggregate routes by type and permission."""

    (
        distance_matrix,
        duration_matrix,
        transport_cost_matrix,
    ) = route_matrices

    route_analysis = analyze_routes(
        stores=sample_stores,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
        transport_cost_matrix=transport_cost_matrix,
    )

    summary = create_route_summary(
        route_analysis=route_analysis
    )

    intra_summary = summary.loc[
        (summary["route_type"] == INTRA_CITY_ROUTE)
        & summary["is_allowed"]
    ].iloc[0]

    inter_summary = summary.loc[
        (summary["route_type"] == INTER_CITY_ROUTE)
        & ~summary["is_allowed"]
    ].iloc[0]

    assert intra_summary["route_count"] == 2
    assert inter_summary["route_count"] == 4


def test_get_rejection_reason() -> None:
    """Rejection reasons should match failed constraints."""

    assert get_rejection_reason(True, True) == ""

    assert (
        get_rejection_reason(False, True)
        == "distance_limit"
    )

    assert (
        get_rejection_reason(True, False)
        == "lead_time_limit"
    )

    assert (
        get_rejection_reason(False, False)
        == "distance_and_lead_time_limit"
    )