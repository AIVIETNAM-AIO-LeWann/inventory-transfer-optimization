"""Tests for the genetic algorithm optimizer."""

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
from src.optimizers.genetic_algorithm import (
    create_inventory_limits,
    evaluate_individual,
    optimize_genetic_algorithm,
    repair_individual,
    save_transfer_plan,
    validate_ga_settings,
)
from src.optimizers.greedy import optimize_greedy
from src.optimizers.linear_programming import (
    build_transfer_candidates,
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


@pytest.mark.parametrize(
    (
        "population_size",
        "generations",
        "crossover_probability",
        "mutation_probability",
        "tournament_size",
        "expected_message",
    ),
    [
        (
            1,
            10,
            0.7,
            0.1,
            2,
            "population_size",
        ),
        (
            10,
            0,
            0.7,
            0.1,
            2,
            "generations",
        ),
        (
            10,
            10,
            1.5,
            0.1,
            2,
            "crossover_probability",
        ),
        (
            10,
            10,
            0.7,
            -0.1,
            2,
            "mutation_probability",
        ),
        (
            10,
            10,
            0.7,
            0.1,
            11,
            "tournament_size",
        ),
    ],
)
def test_validate_ga_settings_rejects_invalid_values(
    population_size: int,
    generations: int,
    crossover_probability: float,
    mutation_probability: float,
    tournament_size: int,
    expected_message: str,
) -> None:
    """Invalid GA settings should be rejected."""

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_ga_settings(
            population_size=population_size,
            generations=generations,
            crossover_probability=(
                crossover_probability
            ),
            mutation_probability=(
                mutation_probability
            ),
            tournament_size=tournament_size,
        )


def test_repair_individual_respects_limits(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Repair should prevent excessive transfers."""

    candidates = build_transfer_candidates(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    (
        source_limits,
        shortage_limits,
    ) = create_inventory_limits(
        inventory_analysis=sample_inventory_analysis
    )

    individual = [8, 10]

    repaired = repair_individual(
        individual=individual,
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
    )

    assert repaired == [5, 7]
    assert sum(repaired) == 12


def test_evaluate_individual_penalizes_shortage(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """Unresolved shortage should strongly increase fitness."""

    candidates = build_transfer_candidates(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    complete_fitness = evaluate_individual(
        individual=[5, 7],
        candidates=candidates,
        total_shortage=12,
        unmet_penalty_per_unit=1_000_000.0,
    )

    incomplete_fitness = evaluate_individual(
        individual=[5, 6],
        candidates=candidates,
        total_shortage=12,
        unmet_penalty_per_unit=1_000_000.0,
    )

    assert complete_fitness == (850.0,)
    assert incomplete_fitness == (1_000_800.0,)
    assert complete_fitness < incomplete_fitness


def test_genetic_algorithm_finds_low_cost_plan(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """GA should find a complete low-cost solution."""

    plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
        population_size=20,
        generations=10,
        crossover_probability=0.7,
        mutation_probability=0.1,
        tournament_size=3,
        seed=2026,
    )

    assert plan["quantity"].sum() == 12
    assert plan["total_transport_cost"].sum() == 700.0

    source_totals = plan.groupby(
        "from_store_id"
    )["quantity"].sum()

    assert source_totals["S002"] == 2
    assert source_totals["S003"] == 10


def test_genetic_algorithm_is_reproducible(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """The same random seed should produce the same plan."""

    first_plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
        population_size=20,
        generations=10,
        seed=2026,
    )

    second_plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
        population_size=20,
        generations=10,
        seed=2026,
    )

    pd.testing.assert_frame_equal(
        first_plan,
        second_plan,
    )


def test_genetic_algorithm_ignores_disallowed_routes(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """GA must not use disallowed routes."""

    disallowed_routes = sample_route_analysis.copy()
    disallowed_routes["is_allowed"] = False

    plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=disallowed_routes,
        population_size=10,
        generations=5,
        tournament_size=2,
        seed=2026,
    )

    assert plan.empty


def test_genetic_algorithm_is_cheaper_than_greedy(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
) -> None:
    """GA should improve on Greedy for the sample problem."""

    greedy_plan = optimize_greedy(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
    )

    genetic_plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
        population_size=20,
        generations=10,
        seed=2026,
    )

    greedy_cost = float(
        greedy_plan["total_transport_cost"].sum()
    )

    genetic_cost = float(
        genetic_plan["total_transport_cost"].sum()
    )

    assert greedy_cost == 850.0
    assert genetic_cost == 700.0
    assert genetic_cost < greedy_cost


def test_save_genetic_algorithm_plan(
    sample_inventory_analysis: pd.DataFrame,
    sample_route_analysis: pd.DataFrame,
    tmp_path,
) -> None:
    """The genetic transfer plan should be saved as CSV."""

    plan = optimize_genetic_algorithm(
        inventory_analysis=sample_inventory_analysis,
        route_analysis=sample_route_analysis,
        population_size=20,
        generations=10,
        seed=2026,
    )

    output_path = (
        tmp_path
        / "genetic_algorithm_transfer_plan.csv"
    )

    saved_path = save_transfer_plan(
        transfer_plan=plan,
        output_path=output_path,
    )

    assert saved_path.exists()

    loaded_plan = pd.read_csv(saved_path)

    assert loaded_plan["quantity"].sum() == 12

    assert (
        loaded_plan["total_transport_cost"].sum()
        == 700.0
    )