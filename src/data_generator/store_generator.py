"""Generate sample store data for the inventory optimization system."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    STORES_FILE,
    STORE_CITIES,
    validate_config,
)


STORE_BRANDS = (
    "AIO Mart",
    "Lotus Market",
    "Green Basket",
    "Urban Retail",
    "Central Market",
    "Family Store",
    "Fresh Hub",
    "City Mart",
)


STORE_LOCATIONS = {
    "Hanoi": (
        "Ba Dinh",
        "Hoan Kiem",
        "Tay Ho",
        "Long Bien",
        "Cau Giay",
        "Dong Da",
        "Hai Ba Trung",
        "Hoang Mai",
        "Thanh Xuan",
        "My Dinh",
    ),
    "Da Nang": (
        "Hai Chau",
        "Thanh Khe",
        "Son Tra",
        "Ngu Hanh Son",
        "Lien Chieu",
        "Cam Le",
        "Hoa Vang",
    ),
    "Ho Chi Minh City": (
        "District 1",
        "District 2",
        "District 3",
        "District 6",
        "District 7",
        "Thu Duc",
        "Binh Thanh",
        "Phu Nhuan",
        "Tan Binh",
        "Go Vap",
        "Binh Tan",
        "Tan Phu",
        "Nha Be",
    ),
}


STORE_COORDINATES = {
    "Hanoi": {
        "Ba Dinh": (21.0358, 105.8347),
        "Hoan Kiem": (21.0285, 105.8542),
        "Tay Ho": (21.0687, 105.8230),
        "Long Bien": (21.0542, 105.8914),
        "Cau Giay": (21.0362, 105.7906),
        "Dong Da": (21.0180, 105.8290),
        "Hai Ba Trung": (21.0091, 105.8490),
        "Hoang Mai": (20.9743, 105.8522),
        "Thanh Xuan": (20.9938, 105.8115),
        "My Dinh": (21.0285, 105.7731),
    },
    "Da Nang": {
        "Hai Chau": (16.0471, 108.2068),
        "Thanh Khe": (16.0636, 108.1847),
        "Son Tra": (16.0810, 108.2370),
        "Ngu Hanh Son": (16.0032, 108.2450),
        "Lien Chieu": (16.0757, 108.1505),
        "Cam Le": (16.0150, 108.2000),
        "Hoa Vang": (16.0500, 108.1200),
    },
    "Ho Chi Minh City": {
        "District 1": (10.7756, 106.7009),
        "District 2": (10.7873, 106.7498),
        "District 3": (10.7844, 106.6844),
        "District 6": (10.7462, 106.6352),
        "District 7": (10.7340, 106.7216),
        "Thu Duc": (10.8494, 106.7537),
        "Binh Thanh": (10.8106, 106.7091),
        "Phu Nhuan": (10.7992, 106.6802),
        "Tan Binh": (10.8015, 106.6526),
        "Go Vap": (10.8387, 106.6653),
        "Binh Tan": (10.7653, 106.6038),
        "Tan Phu": (10.7901, 106.6284),
        "Nha Be": (10.6953, 106.7042),
    },
}


STORE_COLUMNS = (
    "store_id",
    "store_name",
    "city",
    "latitude",
    "longitude",
)


def validate_store_generation_settings() -> None:
    """Validate settings required by the store generator."""

    missing_location_cities = (
        set(STORE_CITIES)
        - set(STORE_LOCATIONS)
    )

    if missing_location_cities:
        raise ValueError(
            "Missing location names for cities: "
            f"{sorted(missing_location_cities)}"
        )

    missing_coordinate_cities = (
        set(STORE_CITIES)
        - set(STORE_COORDINATES)
    )

    if missing_coordinate_cities:
        raise ValueError(
            "Missing coordinates for cities: "
            f"{sorted(missing_coordinate_cities)}"
        )

    for city, city_config in (
        STORE_CITIES.items()
    ):
        store_count = city_config["count"]
        available_locations = (
            STORE_LOCATIONS[city]
        )
        available_coordinates = (
            STORE_COORDINATES[city]
        )

        if store_count <= 0:
            raise ValueError(
                f"Store count for {city} must "
                "be greater than zero."
            )

        if store_count > len(
            available_locations
        ):
            raise ValueError(
                f"{city} requires {store_count} "
                "unique locations, but only "
                f"{len(available_locations)} "
                "are available."
            )

        missing_locations = (
            set(available_locations)
            - set(available_coordinates)
        )

        if missing_locations:
            raise ValueError(
                f"Missing coordinates for "
                f"{city}: "
                f"{sorted(missing_locations)}"
            )

        extra_locations = (
            set(available_coordinates)
            - set(available_locations)
        )

        if extra_locations:
            raise ValueError(
                f"Coordinates contain unknown "
                f"locations for {city}: "
                f"{sorted(extra_locations)}"
            )

        for location in (
            available_locations
        ):
            latitude, longitude = (
                available_coordinates[
                    location
                ]
            )

            if (
                isinstance(latitude, bool)
                or not isinstance(
                    latitude,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"Latitude for {city} - "
                    f"{location} must be numeric."
                )

            if (
                isinstance(longitude, bool)
                or not isinstance(
                    longitude,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"Longitude for {city} - "
                    f"{location} must be numeric."
                )

            if not np.isfinite(latitude):
                raise ValueError(
                    f"Latitude for {city} - "
                    f"{location} must be finite."
                )

            if not np.isfinite(longitude):
                raise ValueError(
                    f"Longitude for {city} - "
                    f"{location} must be finite."
                )

            if not -90 <= latitude <= 90:
                raise ValueError(
                    f"Invalid latitude for "
                    f"{city} - {location}."
                )

            if not -180 <= longitude <= 180:
                raise ValueError(
                    f"Invalid longitude for "
                    f"{city} - {location}."
                )


def generate_stores(
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate stores using fixed location coordinates."""

    validate_config()
    validate_store_generation_settings()

    rng = np.random.default_rng(seed)

    store_records: list[
        dict[str, object]
    ] = []

    store_number = 1

    for city, city_config in (
        STORE_CITIES.items()
    ):
        store_count = city_config["count"]

        available_locations = (
            STORE_LOCATIONS[city]
        )

        selected_location_indexes = (
            rng.choice(
                len(available_locations),
                size=store_count,
                replace=False,
            )
        )

        for location_index in (
            selected_location_indexes
        ):
            store_id = (
                f"S{store_number:03d}"
            )

            brand = str(
                rng.choice(STORE_BRANDS)
            )

            location = (
                available_locations[
                    int(location_index)
                ]
            )

            latitude, longitude = (
                STORE_COORDINATES[
                    city
                ][location]
            )

            store_records.append(
                {
                    "store_id": store_id,
                    "store_name": (
                        f"{brand} - {location}"
                    ),
                    "city": city,
                    "latitude": round(
                        float(latitude),
                        6,
                    ),
                    "longitude": round(
                        float(longitude),
                        6,
                    ),
                }
            )

            store_number += 1

    stores = pd.DataFrame(
        store_records,
        columns=STORE_COLUMNS,
    )

    validate_generated_stores(
        stores
    )

    return stores

def validate_generated_stores(stores: pd.DataFrame) -> None:
    """Validate a generated store DataFrame."""

    expected_count = sum(
        city_config["count"]
        for city_config in STORE_CITIES.values()
    )

    missing_columns = set(STORE_COLUMNS) - set(stores.columns)

    if missing_columns:
        raise ValueError(
            f"Generated store data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(stores) != expected_count:
        raise ValueError(
            f"Expected {expected_count} stores, "
            f"but generated {len(stores)}."
        )

    if stores["store_id"].duplicated().any():
        raise ValueError(
            "Generated store IDs must be unique."
        )

    if stores["store_name"].duplicated().any():
        raise ValueError(
            "Generated store names must be unique."
        )

    if not stores["latitude"].between(-90, 90).all():
        raise ValueError(
            "Generated latitude values must be between -90 and 90."
        )

    if not stores["longitude"].between(-180, 180).all():
        raise ValueError(
            "Generated longitude values must be between -180 and 180."
        )

    if stores[list(STORE_COLUMNS)].isna().any().any():
        raise ValueError(
            "Generated store data must not contain missing values."
        )


def save_stores(
    stores: pd.DataFrame,
    output_path: str | Path = STORES_FILE,
) -> Path:
    """
    Save generated store data to a CSV file.

    Args:
        stores:
            Store data to save.
        output_path:
            Destination CSV path.

    Returns:
        The resolved output path.
    """

    validate_generated_stores(stores)

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stores.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Generate, validate, and save sample store data."""

    stores = generate_stores()
    output_path = save_stores(stores)

    print(f"Generated {len(stores)} stores.")
    print(f"Saved store data to: {output_path}")
    print()
    print(stores.to_string(index=False))


if __name__ == "__main__":
    main()