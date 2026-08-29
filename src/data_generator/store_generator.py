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


STORE_COLUMNS = (
    "store_id",
    "store_name",
    "city",
    "latitude",
    "longitude",
)


def validate_store_generation_settings() -> None:
    """Validate settings required by the store generator."""

    missing_cities = set(STORE_CITIES) - set(STORE_LOCATIONS)

    if missing_cities:
        raise ValueError(
            "Missing location names for cities: "
            f"{sorted(missing_cities)}"
        )

    for city, city_config in STORE_CITIES.items():
        store_count = city_config["count"]
        latitude_range = city_config["lat_range"]
        longitude_range = city_config["lon_range"]

        if store_count <= 0:
            raise ValueError(
                f"Store count for {city} must be greater than zero."
            )

        if latitude_range[0] >= latitude_range[1]:
            raise ValueError(
                f"Invalid latitude range for {city}."
            )

        if longitude_range[0] >= longitude_range[1]:
            raise ValueError(
                f"Invalid longitude range for {city}."
            )

        if store_count > len(STORE_LOCATIONS[city]):
            raise ValueError(
                f"{city} requires {store_count} unique locations, "
                f"but only {len(STORE_LOCATIONS[city])} are available."
            )


def generate_stores(
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate sample stores using configured cities and coordinate ranges.

    Args:
        seed:
            Random seed used to produce reproducible store data.

    Returns:
        A DataFrame containing generated store information.
    """

    validate_config()
    validate_store_generation_settings()

    # A local random generator avoids changing global random state.
    rng = np.random.default_rng(seed)

    store_records: list[dict[str, object]] = []
    store_number = 1

    for city, city_config in STORE_CITIES.items():
        store_count = city_config["count"]
        latitude_min, latitude_max = city_config["lat_range"]
        longitude_min, longitude_max = city_config["lon_range"]

        available_locations = STORE_LOCATIONS[city]

        # Select different locations within the same city.
        selected_location_indexes = rng.choice(
            len(available_locations),
            size=store_count,
            replace=False,
        )

        for location_index in selected_location_indexes:
            store_id = f"S{store_number:03d}"
            brand = str(rng.choice(STORE_BRANDS))
            location = available_locations[int(location_index)]

            latitude = rng.uniform(
                latitude_min,
                latitude_max,
            )

            longitude = rng.uniform(
                longitude_min,
                longitude_max,
            )

            store_records.append(
                {
                    "store_id": store_id,
                    "store_name": f"{brand} - {location}",
                    "city": city,
                    "latitude": round(float(latitude), 6),
                    "longitude": round(float(longitude), 6),
                }
            )

            store_number += 1

    stores = pd.DataFrame(
        store_records,
        columns=STORE_COLUMNS,
    )

    validate_generated_stores(stores)

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