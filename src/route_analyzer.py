"""Analyze store-to-store inventory transfer routes."""

from pathlib import Path

import pandas as pd

from src.config import (
    INTER_CITY_HANDLING_TIME_MINUTES,
    INTER_CITY_MAX_LEAD_TIME_MINUTES,
    INTER_CITY_ROUTE,
    INTRA_CITY_HANDLING_TIME_MINUTES,
    INTRA_CITY_MAX_LEAD_TIME_MINUTES,
    INTRA_CITY_ROUTE,
    MAX_TRANSFER_DISTANCE_KM,
    ROUTE_ANALYSIS_FILE,
)
from src.data_loader import load_all_data
from src.validator import (
    validate_route_matrix,
    validate_stores,
)


ROUTE_ANALYSIS_COLUMNS = (
    "from_store_id",
    "to_store_id",
    "from_city",
    "to_city",
    "route_type",
    "priority_rank",
    "distance_km",
    "driving_time_minutes",
    "handling_time_minutes",
    "lead_time_minutes",
    "maximum_lead_time_minutes",
    "transport_cost_per_unit",
    "distance_allowed",
    "time_allowed",
    "is_allowed",
    "rejection_reason",
)


def validate_route_policy() -> None:
    """Validate route policy configuration."""

    if MAX_TRANSFER_DISTANCE_KM <= 0:
        raise ValueError(
            "MAX_TRANSFER_DISTANCE_KM "
            "must be greater than zero."
        )

    if INTRA_CITY_HANDLING_TIME_MINUTES < 0:
        raise ValueError(
            "INTRA_CITY_HANDLING_TIME_MINUTES "
            "must not be negative."
        )

    if INTER_CITY_HANDLING_TIME_MINUTES < 0:
        raise ValueError(
            "INTER_CITY_HANDLING_TIME_MINUTES "
            "must not be negative."
        )

    if INTRA_CITY_MAX_LEAD_TIME_MINUTES <= 0:
        raise ValueError(
            "INTRA_CITY_MAX_LEAD_TIME_MINUTES "
            "must be greater than zero."
        )

    if INTER_CITY_MAX_LEAD_TIME_MINUTES <= 0:
        raise ValueError(
            "INTER_CITY_MAX_LEAD_TIME_MINUTES "
            "must be greater than zero."
        )


def classify_route(
    from_city: str,
    to_city: str,
) -> str:
    """Classify a route as intra-city or inter-city."""

    if from_city == to_city:
        return INTRA_CITY_ROUTE

    return INTER_CITY_ROUTE


def get_route_policy(
    route_type: str,
) -> tuple[int, int, int]:
    """
    Return handling time, maximum lead time, and priority.

    A lower priority rank means the optimizer should consider
    the route earlier.
    """

    if route_type == INTRA_CITY_ROUTE:
        return (
            INTRA_CITY_HANDLING_TIME_MINUTES,
            INTRA_CITY_MAX_LEAD_TIME_MINUTES,
            1,
        )

    if route_type == INTER_CITY_ROUTE:
        return (
            INTER_CITY_HANDLING_TIME_MINUTES,
            INTER_CITY_MAX_LEAD_TIME_MINUTES,
            2,
        )

    raise ValueError(
        f"Unsupported route type: {route_type}"
    )


def get_rejection_reason(
    distance_allowed: bool,
    time_allowed: bool,
) -> str:
    """Return the reason why a route is not allowed."""

    if distance_allowed and time_allowed:
        return ""

    if not distance_allowed and not time_allowed:
        return "distance_and_lead_time_limit"

    if not distance_allowed:
        return "distance_limit"

    return "lead_time_limit"


def analyze_routes(
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    duration_matrix: pd.DataFrame,
    transport_cost_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze every directed store-to-store transfer route."""

    validate_route_policy()
    validate_stores(stores)

    validate_route_matrix(
        matrix=distance_matrix,
        stores=stores,
        matrix_name="distance_matrix",
    )

    validate_route_matrix(
        matrix=duration_matrix,
        stores=stores,
        matrix_name="duration_matrix",
    )

    validate_route_matrix(
        matrix=transport_cost_matrix,
        stores=stores,
        matrix_name="transport_cost_matrix",
    )

    route_records: list[dict[str, object]] = []

    for from_store in stores.itertuples(index=False):
        for to_store in stores.itertuples(index=False):
            if from_store.store_id == to_store.store_id:
                continue

            route_type = classify_route(
                from_city=from_store.city,
                to_city=to_store.city,
            )

            (
                handling_time_minutes,
                maximum_lead_time_minutes,
                priority_rank,
            ) = get_route_policy(route_type)

            distance_km = float(
                distance_matrix.loc[
                    from_store.store_id,
                    to_store.store_id,
                ]
            )

            driving_time_minutes = float(
                duration_matrix.loc[
                    from_store.store_id,
                    to_store.store_id,
                ]
            )

            transport_cost_per_unit = float(
                transport_cost_matrix.loc[
                    from_store.store_id,
                    to_store.store_id,
                ]
            )

            lead_time_minutes = (
                driving_time_minutes
                + handling_time_minutes
            )

            distance_allowed = (
                distance_km
                <= MAX_TRANSFER_DISTANCE_KM
            )

            time_allowed = (
                lead_time_minutes
                <= maximum_lead_time_minutes
            )

            is_allowed = (
                distance_allowed and time_allowed
            )

            rejection_reason = get_rejection_reason(
                distance_allowed=distance_allowed,
                time_allowed=time_allowed,
            )

            route_records.append(
                {
                    "from_store_id": (
                        from_store.store_id
                    ),
                    "to_store_id": (
                        to_store.store_id
                    ),
                    "from_city": from_store.city,
                    "to_city": to_store.city,
                    "route_type": route_type,
                    "priority_rank": priority_rank,
                    "distance_km": round(
                        distance_km,
                        3,
                    ),
                    "driving_time_minutes": round(
                        driving_time_minutes,
                        2,
                    ),
                    "handling_time_minutes": (
                        handling_time_minutes
                    ),
                    "lead_time_minutes": round(
                        lead_time_minutes,
                        2,
                    ),
                    "maximum_lead_time_minutes": (
                        maximum_lead_time_minutes
                    ),
                    "transport_cost_per_unit": round(
                        transport_cost_per_unit,
                        2,
                    ),
                    "distance_allowed": (
                        distance_allowed
                    ),
                    "time_allowed": time_allowed,
                    "is_allowed": is_allowed,
                    "rejection_reason": (
                        rejection_reason
                    ),
                }
            )

    route_analysis = pd.DataFrame(
        route_records,
        columns=ROUTE_ANALYSIS_COLUMNS,
    )

    route_analysis = route_analysis.sort_values(
        by=[
            "priority_rank",
            "transport_cost_per_unit",
            "lead_time_minutes",
        ],
        ascending=True,
        ignore_index=True,
    )

    return route_analysis


def create_route_summary(
    route_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize route information by route type."""

    summary = (
        route_analysis.groupby(
            ["route_type", "is_allowed"],
            as_index=False,
        )
        .agg(
            route_count=(
                "from_store_id",
                "size",
            ),
            average_distance_km=(
                "distance_km",
                "mean",
            ),
            average_driving_time_minutes=(
                "driving_time_minutes",
                "mean",
            ),
            average_lead_time_minutes=(
                "lead_time_minutes",
                "mean",
            ),
            average_transport_cost_per_unit=(
                "transport_cost_per_unit",
                "mean",
            ),
        )
    )

    numeric_columns = (
        "average_distance_km",
        "average_driving_time_minutes",
        "average_lead_time_minutes",
        "average_transport_cost_per_unit",
    )

    summary[list(numeric_columns)] = summary[
        list(numeric_columns)
    ].round(2)

    return summary


def save_route_analysis(
    route_analysis: pd.DataFrame,
    output_path: str | Path = ROUTE_ANALYSIS_FILE,
) -> Path:
    """Save route analysis results to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_analysis.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Run route analysis using project data."""

    project_data = load_all_data()

    route_analysis = analyze_routes(
        stores=project_data.stores,
        distance_matrix=project_data.distance_matrix,
        duration_matrix=project_data.duration_matrix,
        transport_cost_matrix=(
            project_data.transport_cost_matrix
        ),
    )

    summary = create_route_summary(
        route_analysis
    )

    output_path = save_route_analysis(
        route_analysis
    )

    allowed_route_count = int(
        route_analysis["is_allowed"].sum()
    )

    print("Route analysis completed successfully.")
    print(f"Analyzed routes: {len(route_analysis)}")
    print(f"Allowed routes: {allowed_route_count}")
    print(f"Saved to: {output_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()