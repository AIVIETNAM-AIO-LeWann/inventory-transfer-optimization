"""Optimize inventory transfers using a genetic algorithm."""

from pathlib import Path
from time import perf_counter
import random

import pandas as pd
from deap import base, creator, tools

from src.config import (
    GA_CROSSOVER_PROBABILITY,
    GA_GENERATIONS,
    GA_MUTATION_PROBABILITY,
    GA_POPULATION_SIZE,
    GA_TOURNAMENT_SIZE,
    GENETIC_ALGORITHM_TRANSFER_PLAN_FILE,
    RANDOM_SEED,
    UNMET_SHORTAGE_PENALTY_PER_UNIT,
)
from src.data_loader import load_all_data
from src.inventory_analyzer import (
    EXCESS_STATUS,
    SHORTAGE_STATUS,
    analyze_inventory,
)
from src.metrics import calculate_plan_metrics
from src.optimizers.linear_programming import (
    TRANSFER_PLAN_COLUMNS,
    build_transfer_candidates,
    validate_linear_programming_inputs,
)
from src.route_analyzer import analyze_routes


def validate_ga_settings(
    population_size: int,
    generations: int,
    crossover_probability: float,
    mutation_probability: float,
    tournament_size: int,
) -> None:
    """Validate genetic algorithm settings."""

    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2."
        )

    if generations <= 0:
        raise ValueError(
            "generations must be greater than zero."
        )

    if not 0 <= crossover_probability <= 1:
        raise ValueError(
            "crossover_probability must be "
            "between 0 and 1."
        )

    if not 0 <= mutation_probability <= 1:
        raise ValueError(
            "mutation_probability must be "
            "between 0 and 1."
        )

    if not 2 <= tournament_size <= population_size:
        raise ValueError(
            "tournament_size must be between "
            "2 and population_size."
        )


def create_inventory_limits(
    inventory_analysis: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
]:
    """Create source excess and destination shortage limits."""

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

    shortage_limits = {
        (
            str(row.store_id),
            str(row.product_id),
        ): int(row.shortage_quantity)
        for row in inventory_analysis.loc[
            (
                inventory_analysis["status"]
                == SHORTAGE_STATUS
            )
            & (
                inventory_analysis["shortage_quantity"]
                > 0
            )
        ].itertuples(index=False)
    }

    return source_limits, shortage_limits


def repair_individual(
    individual: list[int],
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
) -> list[int]:
    """Repair an individual so all constraints are satisfied."""

    remaining_excess = source_limits.copy()
    remaining_shortage = shortage_limits.copy()

    for index, candidate in enumerate(candidates):
        source_key = (
            str(candidate["from_store_id"]),
            str(candidate["product_id"]),
        )

        destination_key = (
            str(candidate["to_store_id"]),
            str(candidate["product_id"]),
        )

        requested_quantity = max(
            int(round(individual[index])),
            0,
        )

        maximum_quantity = min(
            remaining_excess.get(source_key, 0),
            remaining_shortage.get(
                destination_key,
                0,
            ),
        )

        repaired_quantity = min(
            requested_quantity,
            maximum_quantity,
        )

        individual[index] = repaired_quantity

        remaining_excess[source_key] -= (
            repaired_quantity
        )

        remaining_shortage[destination_key] -= (
            repaired_quantity
        )

    return individual


def create_random_genome(
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
) -> list[int]:
    """Create a random feasible transfer genome."""

    genome = [0] * len(candidates)

    remaining_excess = source_limits.copy()
    remaining_shortage = shortage_limits.copy()

    candidate_indexes = list(range(len(candidates)))
    random.shuffle(candidate_indexes)

    for index in candidate_indexes:
        candidate = candidates[index]

        source_key = (
            str(candidate["from_store_id"]),
            str(candidate["product_id"]),
        )

        destination_key = (
            str(candidate["to_store_id"]),
            str(candidate["product_id"]),
        )

        maximum_quantity = min(
            remaining_excess.get(source_key, 0),
            remaining_shortage.get(
                destination_key,
                0,
            ),
        )

        if maximum_quantity <= 0:
            continue

        quantity = random.randint(
            0,
            maximum_quantity,
        )

        genome[index] = quantity

        remaining_excess[source_key] -= quantity
        remaining_shortage[destination_key] -= quantity

    return genome


def create_heuristic_genome(
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
    prioritize_route_type: bool = False,
) -> list[int]:
    """Create a feasible genome using a greedy heuristic."""

    genome = [0] * len(candidates)

    remaining_excess = source_limits.copy()
    remaining_shortage = shortage_limits.copy()

    if prioritize_route_type:
        candidate_indexes = sorted(
            range(len(candidates)),
            key=lambda index: (
                candidates[index]["priority_rank"],
                candidates[index][
                    "transport_cost_per_unit"
                ],
                candidates[index]["lead_time_minutes"],
            ),
        )
    else:
        candidate_indexes = sorted(
            range(len(candidates)),
            key=lambda index: (
                candidates[index][
                    "transport_cost_per_unit"
                ],
                candidates[index]["lead_time_minutes"],
                candidates[index]["priority_rank"],
            ),
        )

    for index in candidate_indexes:
        candidate = candidates[index]

        source_key = (
            str(candidate["from_store_id"]),
            str(candidate["product_id"]),
        )

        destination_key = (
            str(candidate["to_store_id"]),
            str(candidate["product_id"]),
        )

        quantity = min(
            remaining_excess.get(source_key, 0),
            remaining_shortage.get(
                destination_key,
                0,
            ),
        )

        if quantity <= 0:
            continue

        genome[index] = quantity

        remaining_excess[source_key] -= quantity
        remaining_shortage[destination_key] -= quantity

    return genome


def evaluate_individual(
    individual: list[int],
    candidates: list[dict[str, object]],
    total_shortage: int,
    unmet_penalty_per_unit: float,
) -> tuple[float]:
    """Calculate the fitness value of one individual."""

    transferred_quantity = sum(
        int(quantity)
        for quantity in individual
    )

    transport_cost = sum(
        int(individual[index])
        * float(
            candidate["transport_cost_per_unit"]
        )
        for index, candidate
        in enumerate(candidates)
    )

    remaining_shortage = max(
        total_shortage - transferred_quantity,
        0,
    )

    objective_value = (
        transport_cost
        + remaining_shortage
        * unmet_penalty_per_unit
    )

    return (float(objective_value),)


def mutate_individual(
    individual: list[int],
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
) -> tuple[list[int]]:
    """Randomly mutate genes and repair the individual."""

    if not individual:
        return (individual,)

    gene_probability = max(
        1.0 / len(individual),
        0.01,
    )

    for index, candidate in enumerate(candidates):
        if random.random() >= gene_probability:
            continue

        maximum_gene_value = min(
            int(candidate["source_excess"]),
            int(candidate["destination_shortage"]),
        )

        individual[index] = random.randint(
            0,
            maximum_gene_value,
        )

    repair_individual(
        individual=individual,
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
    )

    return (individual,)


def crossover_individuals(
    first_individual: list[int],
    second_individual: list[int],
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
) -> tuple[list[int], list[int]]:
    """Perform two-point crossover and repair children."""

    if len(first_individual) >= 2:
        tools.cxTwoPoint(
            first_individual,
            second_individual,
        )

    repair_individual(
        individual=first_individual,
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
    )

    repair_individual(
        individual=second_individual,
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
    )

    return first_individual, second_individual


def ensure_deap_types() -> None:
    """Create DEAP fitness and individual classes once."""

    if not hasattr(
        creator,
        "InventoryTransferFitness",
    ):
        creator.create(
            "InventoryTransferFitness",
            base.Fitness,
            weights=(-1.0,),
        )

    if not hasattr(
        creator,
        "InventoryTransferIndividual",
    ):
        creator.create(
            "InventoryTransferIndividual",
            list,
            fitness=(
                creator.InventoryTransferFitness
            ),
        )


def evolve_best_individual(
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
    population_size: int,
    generations: int,
    crossover_probability: float,
    mutation_probability: float,
    tournament_size: int,
    unmet_penalty_per_unit: float,
    seed: int,
) -> list[int]:
    """Run evolution and return the best genome."""

    if not candidates:
        return []

    ensure_deap_types()

    previous_random_state = random.getstate()
    random.seed(seed)

    try:
        toolbox = base.Toolbox()

        total_shortage = sum(
            shortage_limits.values()
        )

        toolbox.register(
            "evaluate",
            evaluate_individual,
            candidates=candidates,
            total_shortage=total_shortage,
            unmet_penalty_per_unit=(
                unmet_penalty_per_unit
            ),
        )

        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=tournament_size,
        )

        toolbox.register(
            "mate",
            crossover_individuals,
            candidates=candidates,
            source_limits=source_limits,
            shortage_limits=shortage_limits,
        )

        toolbox.register(
            "mutate",
            mutate_individual,
            candidates=candidates,
            source_limits=source_limits,
            shortage_limits=shortage_limits,
        )

        individual_class = (
            creator.InventoryTransferIndividual
        )

        population = [
            individual_class(
                create_random_genome(
                    candidates=candidates,
                    source_limits=source_limits,
                    shortage_limits=shortage_limits,
                )
            )
            for _ in range(population_size)
        ]

        population[0] = individual_class(
            create_heuristic_genome(
                candidates=candidates,
                source_limits=source_limits,
                shortage_limits=shortage_limits,
                prioritize_route_type=False,
            )
        )

        if population_size > 1:
            population[1] = individual_class(
                create_heuristic_genome(
                    candidates=candidates,
                    source_limits=source_limits,
                    shortage_limits=shortage_limits,
                    prioritize_route_type=True,
                )
            )

        for individual in population:
            individual.fitness.values = (
                toolbox.evaluate(individual)
            )

        for _ in range(generations):
            elite = toolbox.clone(
                tools.selBest(
                    population,
                    1,
                )[0]
            )

            offspring = toolbox.select(
                population,
                population_size - 1,
            )

            offspring = [
                toolbox.clone(individual)
                for individual in offspring
            ]

            for first_index in range(
                0,
                len(offspring) - 1,
                2,
            ):
                first_child = offspring[first_index]
                second_child = offspring[
                    first_index + 1
                ]

                if (
                    random.random()
                    < crossover_probability
                ):
                    toolbox.mate(
                        first_child,
                        second_child,
                    )

                    del first_child.fitness.values
                    del second_child.fitness.values

            for individual in offspring:
                if (
                    random.random()
                    < mutation_probability
                ):
                    toolbox.mutate(individual)

                    if individual.fitness.valid:
                        del individual.fitness.values

            invalid_individuals = [
                individual
                for individual in offspring
                if not individual.fitness.valid
            ]

            for individual in invalid_individuals:
                individual.fitness.values = (
                    toolbox.evaluate(individual)
                )

            population = offspring + [elite]

        best_individual = tools.selBest(
            population,
            1,
        )[0]

        return [
            int(quantity)
            for quantity in best_individual
        ]

    finally:
        random.setstate(previous_random_state)


def create_transfer_plan_from_genome(
    genome: list[int],
    candidates: list[dict[str, object]],
    source_limits: dict[tuple[str, str], int],
    shortage_limits: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """Convert a GA genome into a transfer plan."""

    allocations = [
        {
            **candidate,
            "quantity": int(genome[index]),
        }
        for index, candidate in enumerate(candidates)
        if int(genome[index]) > 0
    ]

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
            str(allocation["from_store_id"]),
            str(allocation["product_id"]),
        )

        destination_key = (
            str(allocation["to_store_id"]),
            str(allocation["product_id"]),
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
                    f"GA{transfer_number:04d}"
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


def optimize_genetic_algorithm(
    inventory_analysis: pd.DataFrame,
    route_analysis: pd.DataFrame,
    population_size: int = GA_POPULATION_SIZE,
    generations: int = GA_GENERATIONS,
    crossover_probability: float = (
        GA_CROSSOVER_PROBABILITY
    ),
    mutation_probability: float = (
        GA_MUTATION_PROBABILITY
    ),
    tournament_size: int = GA_TOURNAMENT_SIZE,
    unmet_penalty_per_unit: float = (
        UNMET_SHORTAGE_PENALTY_PER_UNIT
    ),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create an inventory transfer plan using GA."""

    validate_linear_programming_inputs(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
        unmet_penalty_per_unit=(
            unmet_penalty_per_unit
        ),
    )

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

    candidates = build_transfer_candidates(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    if not candidates:
        return pd.DataFrame(
            columns=TRANSFER_PLAN_COLUMNS
        )

    (
        source_limits,
        shortage_limits,
    ) = create_inventory_limits(
        inventory_analysis=inventory_analysis
    )

    best_genome = evolve_best_individual(
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
        population_size=population_size,
        generations=generations,
        crossover_probability=(
            crossover_probability
        ),
        mutation_probability=(
            mutation_probability
        ),
        tournament_size=tournament_size,
        unmet_penalty_per_unit=(
            unmet_penalty_per_unit
        ),
        seed=seed,
    )

    return create_transfer_plan_from_genome(
        genome=best_genome,
        candidates=candidates,
        source_limits=source_limits,
        shortage_limits=shortage_limits,
    )


def save_transfer_plan(
    transfer_plan: pd.DataFrame,
    output_path: str | Path = (
        GENETIC_ALGORITHM_TRANSFER_PLAN_FILE
    ),
) -> Path:
    """Save the genetic algorithm transfer plan."""

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
    """Run the genetic algorithm with project data."""

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

    transfer_plan = optimize_genetic_algorithm(
        inventory_analysis=inventory_analysis,
        route_analysis=route_analysis,
    )

    execution_time_seconds = (
        perf_counter() - start_time
    )

    metrics = calculate_plan_metrics(
        inventory_analysis=inventory_analysis,
        transfer_plan=transfer_plan,
        algorithm_name="Genetic Algorithm",
        execution_time_seconds=(
            execution_time_seconds
        ),
    )

    output_path = save_transfer_plan(
        transfer_plan=transfer_plan
    )

    print(
        "Genetic algorithm completed successfully."
    )
    print(f"Saved to: {output_path}")
    print()
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()