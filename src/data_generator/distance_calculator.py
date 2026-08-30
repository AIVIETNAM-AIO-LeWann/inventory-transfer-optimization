"""Calculate route distances, durations, and transport costs."""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    DISTANCE_MATRIX_FILE,
    DURATION_MATRIX_FILE,
    EARTH_RADIUS_KM,
    FALLBACK_AVERAGE_SPEED_KPH,
    MIN_TRANSPORT_COST_PER_UNIT,
    OSRM_REQUEST_TIMEOUT_SECONDS,
    OSRM_TABLE_URL,
    ROUTING_METHOD,
    STORES_FILE,
    TRANSPORT_COST_MATRIX_FILE,
    TRANSPORT_COST_PER_KM_PER_UNIT,
    validate_config,
)


REQUIRED_STORE_COLUMNS = (
    "store_id",
    "latitude",
    "longitude",
)


def validate_store_data(
    stores: pd.DataFrame,
) -> None:
    """Validate store coordinates used for routing."""

    missing_columns = (set(REQUIRED_STORE_COLUMNS) - set(stores.columns))

    if missing_columns:
        raise ValueError(
            "Store data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if stores.empty:
        raise ValueError(
            "Store data must not be empty."
        )

    if stores[list(REQUIRED_STORE_COLUMNS)].isna().any().any():
        raise ValueError(
            "Store data must not contain missing values."
        )

    if stores["store_id"].duplicated().any():
        raise ValueError(
            "Store IDs must be unique."
        )

    if not pd.api.types.is_numeric_dtype(
        stores["latitude"]
    ):
        raise ValueError(
            "Latitude values must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        stores["longitude"]
    ):
        raise ValueError(
            "Longitude values must be numeric."
        )

    if not stores["latitude"].between(-90, 90).all():
        raise ValueError(
            "Latitude values must be between -90 and 90."
        )

    if not stores["longitude"].between(-180, 180).all():
        raise ValueError(
            "Longitude values must be between -180 and 180."
        )


def calculate_haversine_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Calculate straight-line distance between two GPS points."""

    latitude_1_rad = np.radians(latitude_1)
    longitude_1_rad = np.radians(longitude_1)
    latitude_2_rad = np.radians(latitude_2)
    longitude_2_rad = np.radians(longitude_2)

    latitude_difference = (
        latitude_2_rad - latitude_1_rad
    )

    longitude_difference = (
        longitude_2_rad - longitude_1_rad
    )

    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_1_rad)
        * np.cos(latitude_2_rad)
        * np.sin(longitude_difference / 2) ** 2
    )

    haversine_value = np.clip(
        haversine_value,
        0.0,
        1.0,
    )

    central_angle = 2 * np.arctan2(
        np.sqrt(haversine_value),
        np.sqrt(1 - haversine_value),
    )

    return float(
        EARTH_RADIUS_KM * central_angle
    )


def generate_haversine_distance_matrix(
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a symmetric straight-line distance matrix."""

    validate_store_data(stores)

    prepared_stores = stores.reset_index(drop=True)

    store_ids = (
        prepared_stores["store_id"]
        .astype(str)
        .tolist()
    )

    store_count = len(prepared_stores)

    distance_values = np.zeros(
        (store_count, store_count),
        dtype=float,
    )

    for source_index in range(store_count):
        source = prepared_stores.iloc[source_index]

        for destination_index in range(
            source_index + 1,
            store_count,
        ):
            destination = prepared_stores.iloc[
                destination_index
            ]

            distance_km = calculate_haversine_distance(
                latitude_1=float(source["latitude"]),
                longitude_1=float(source["longitude"]),
                latitude_2=float(destination["latitude"]),
                longitude_2=float(destination["longitude"]),
            )

            distance_values[
                source_index,
                destination_index,
            ] = distance_km

            distance_values[
                destination_index,
                source_index,
            ] = distance_km

    distance_matrix = pd.DataFrame(
        distance_values,
        index=store_ids,
        columns=store_ids,
    ).round(3)

    distance_matrix.index.name = "store_id"

    return distance_matrix


def generate_fallback_duration_matrix(
    distance_matrix: pd.DataFrame,
    average_speed_kph: float = (
        FALLBACK_AVERAGE_SPEED_KPH
    ),
) -> pd.DataFrame:
    """Estimate duration in minutes from distance and speed."""

    if average_speed_kph <= 0:
        raise ValueError(
            "average_speed_kph must be greater than zero."
        )

    duration_values = (
        distance_matrix.to_numpy(dtype=float)
        / average_speed_kph
        * 60
    )

    np.fill_diagonal(duration_values, 0.0)

    duration_matrix = pd.DataFrame(
        duration_values,
        index=distance_matrix.index.copy(),
        columns=distance_matrix.columns.copy(),
    ).round(2)

    duration_matrix.index.name = "store_id"

    return duration_matrix


def build_osrm_coordinate_text(
    stores: pd.DataFrame,
) -> str:
    """
    Convert store coordinates to OSRM format.

    OSRM requires longitude,latitude rather than
    latitude,longitude.
    """

    coordinates = []

    for store in stores.itertuples(index=False):
        coordinate = (
            f"{float(store.longitude):.6f},"
            f"{float(store.latitude):.6f}"
        )

        coordinates.append(coordinate)

    return ";".join(coordinates)


def validate_osrm_response(
    response_data: dict,
    expected_store_count: int,
) -> None:
    """Validate the response returned by OSRM."""

    if response_data.get("code") != "Ok":
        message = response_data.get(
            "message",
            "Unknown OSRM error",
        )

        raise RuntimeError(
            f"OSRM returned an error: {message}"
        )

    distances = response_data.get("distances")
    durations = response_data.get("durations")

    if distances is None:
        raise RuntimeError(
            "OSRM response does not contain distances."
        )

    if durations is None:
        raise RuntimeError(
            "OSRM response does not contain durations."
        )

    if len(distances) != expected_store_count:
        raise RuntimeError(
            "OSRM distance matrix has an unexpected size."
        )

    if len(durations) != expected_store_count:
        raise RuntimeError(
            "OSRM duration matrix has an unexpected size."
        )

    for row in distances:
        if len(row) != expected_store_count:
            raise RuntimeError(
                "OSRM distance matrix is not square."
            )

        if any(value is None for value in row):
            raise RuntimeError(
                "OSRM could not find one or more routes."
            )

    for row in durations:
        if len(row) != expected_store_count:
            raise RuntimeError(
                "OSRM duration matrix is not square."
            )

        if any(value is None for value in row):
            raise RuntimeError(
                "OSRM could not calculate one or more "
                "travel durations."
            )


def generate_osrm_route_matrices(
    stores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Request road distance and duration matrices from OSRM.

    Returns:
        Distance matrix in kilometers.
        Duration matrix in minutes.
    """

    validate_store_data(stores)

    prepared_stores = stores.reset_index(drop=True)

    store_ids = (
        prepared_stores["store_id"]
        .astype(str)
        .tolist()
    )

    coordinate_text = build_osrm_coordinate_text(
        prepared_stores
    )

    request_url = (
        f"{OSRM_TABLE_URL}/{coordinate_text}"
    )

    response = requests.get(
        request_url,
        params={
            "annotations": "distance,duration",
        },
        timeout=OSRM_REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    response_data = response.json()

    validate_osrm_response(
        response_data=response_data,
        expected_store_count=len(prepared_stores),
    )

    # OSRM returns meters and seconds.
    distance_values = (
        np.asarray(
            response_data["distances"],
            dtype=float,
        )
        / 1_000
    )

    duration_values = (
        np.asarray(
            response_data["durations"],
            dtype=float,
        )
        / 60
    )

    np.fill_diagonal(distance_values, 0.0)
    np.fill_diagonal(duration_values, 0.0)

    distance_matrix = pd.DataFrame(
        distance_values,
        index=store_ids,
        columns=store_ids,
    ).round(3)

    duration_matrix = pd.DataFrame(
        duration_values,
        index=store_ids,
        columns=store_ids,
    ).round(2)

    distance_matrix.index.name = "store_id"
    duration_matrix.index.name = "store_id"

    return distance_matrix, duration_matrix


def validate_route_matrix(
    matrix: pd.DataFrame,
    matrix_name: str,
) -> None:
    """Validate a distance or duration matrix."""

    if matrix.empty:
        raise ValueError(
            f"{matrix_name} must not be empty."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{matrix_name} must be square."
        )

    if list(matrix.index) != list(matrix.columns):
        raise ValueError(
            f"{matrix_name} row and column IDs "
            "must match."
        )

    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{matrix_name} contains invalid values."
        )

    if (values < 0).any():
        raise ValueError(
            f"{matrix_name} must not contain "
            "negative values."
        )

    if not np.allclose(
        np.diag(values),
        0.0,
    ):
        raise ValueError(
            f"{matrix_name} diagonal must be zero."
        )


def generate_route_matrices(
    stores: pd.DataFrame,
    routing_method: str = ROUTING_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Generate route matrices with automatic Haversine fallback.

    Returns:
        Distance matrix.
        Duration matrix.
        Name of the routing source used.
    """

    normalized_method = routing_method.strip().lower()

    if normalized_method not in {
        "osrm",
        "haversine",
    }:
        raise ValueError(
            "routing_method must be either "
            "'osrm' or 'haversine'."
        )

    if normalized_method == "osrm":
        try:
            distance_matrix, duration_matrix = (
                generate_osrm_route_matrices(stores)
            )

            routing_source = "OSRM"

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as error:
            print(
                "OSRM routing failed. "
                "Using Haversine fallback."
            )
            print(f"Reason: {error}")

            distance_matrix = (
                generate_haversine_distance_matrix(
                    stores
                )
            )

            duration_matrix = (
                generate_fallback_duration_matrix(
                    distance_matrix
                )
            )

            routing_source = "Haversine fallback"

    else:
        distance_matrix = (
            generate_haversine_distance_matrix(stores)
        )

        duration_matrix = (
            generate_fallback_duration_matrix(
                distance_matrix
            )
        )

        routing_source = "Haversine"

    validate_route_matrix(
        distance_matrix,
        "Distance matrix",
    )

    validate_route_matrix(
        duration_matrix,
        "Duration matrix",
    )

    return (
        distance_matrix,
        duration_matrix,
        routing_source,
    )


def generate_transport_cost_matrix(
    distance_matrix: pd.DataFrame,
    cost_per_km_per_unit: float = (
        TRANSPORT_COST_PER_KM_PER_UNIT
    ),
    minimum_cost_per_unit: float = (
        MIN_TRANSPORT_COST_PER_UNIT
    ),
) -> pd.DataFrame:
    """Convert road distances to transport cost per unit."""

    validate_route_matrix(
        distance_matrix,
        "Distance matrix",
    )

    if cost_per_km_per_unit < 0:
        raise ValueError(
            "cost_per_km_per_unit must not be negative."
        )

    if minimum_cost_per_unit < 0:
        raise ValueError(
            "minimum_cost_per_unit must not be negative."
        )

    distance_values = distance_matrix.to_numpy(
        dtype=float
    )

    cost_values = (
        distance_values * cost_per_km_per_unit
    )

    off_diagonal_mask = ~np.eye(
        len(distance_matrix),
        dtype=bool,
    )

    cost_values[off_diagonal_mask] = np.maximum(
        cost_values[off_diagonal_mask],
        minimum_cost_per_unit,
    )

    cost_values[~off_diagonal_mask] = 0.0

    cost_matrix = pd.DataFrame(
        cost_values,
        index=distance_matrix.index.copy(),
        columns=distance_matrix.columns.copy(),
    ).round(2)

    cost_matrix.index.name = "store_id"

    return cost_matrix


def save_matrix(
    matrix: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a matrix to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix.to_csv(
        destination,
        index=True,
        index_label="store_id",
        encoding="utf-8-sig",
    )

    return destination.resolve()


def load_store_data() -> pd.DataFrame:
    """Load store data required for routing."""

    if not STORES_FILE.exists():
        raise FileNotFoundError(
            f"Store data was not found at {STORES_FILE}. "
            "Run the store generator first."
        )

    return pd.read_csv(STORES_FILE)


def main() -> None:
    """Generate and save route and cost matrices."""

    validate_config()

    stores = load_store_data()

    (
        distance_matrix,
        duration_matrix,
        routing_source,
    ) = generate_route_matrices(stores)

    transport_cost_matrix = (
        generate_transport_cost_matrix(
            distance_matrix
        )
    )

    distance_output_path = save_matrix(
        distance_matrix,
        DISTANCE_MATRIX_FILE,
    )

    duration_output_path = save_matrix(
        duration_matrix,
        DURATION_MATRIX_FILE,
    )

    cost_output_path = save_matrix(
        transport_cost_matrix,
        TRANSPORT_COST_MATRIX_FILE,
    )

    print(f"Routing source: {routing_source}")
    print(
        f"Generated matrices for {len(stores)} stores."
    )
    print(
        f"Distance matrix: {distance_output_path}"
    )
    print(
        f"Duration matrix: {duration_output_path}"
    )
    print(
        f"Transport cost matrix: {cost_output_path}"
    )


if __name__ == "__main__":
    main()
