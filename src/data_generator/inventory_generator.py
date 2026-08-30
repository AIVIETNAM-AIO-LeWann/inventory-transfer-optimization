"""Generate sample inventory data based on historical sales."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXCESS_RATIO,
    INVENTORY_FILE,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    RANDOM_SEED,
    SALES_FILE,
    SHORTAGE_RATIO,
    validate_config,
)


REQUIRED_SALES_COLUMNS = (
    "date",
    "store_id",
    "product_id",
    "quantity_sold",
)

INVENTORY_COLUMNS = (
    "store_id",
    "product_id",
    "current_stock",
    "last_updated",
)


def validate_inventory_input_data( sales: pd.DataFrame, ) -> None:
    """Validate sales data used to generate inventory."""

    missing_columns = (set(REQUIRED_SALES_COLUMNS) - set(sales.columns))

    if missing_columns:
        raise ValueError(
            "Sales data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if sales.empty:
        raise ValueError(
            "Sales data must not be empty."
        )

    if sales[list(REQUIRED_SALES_COLUMNS)].isna().any().any():
        raise ValueError(
            "Sales data must not contain missing values."
        )

    duplicate_columns = [
        "date",
        "store_id",
        "product_id",
    ]

    if sales.duplicated(subset=duplicate_columns).any():
        raise ValueError(
            "Sales data must not contain duplicate "
            "date-store-product records."
        )

    if (sales["quantity_sold"] < 0).any():
        raise ValueError(
            "Sales quantities must not be negative."
        )

    parsed_dates = pd.to_datetime(
        sales["date"],
        errors="coerce",
    )

    if parsed_dates.isna().any():
        raise ValueError(
            "Sales data contains invalid dates."
        )


def calculate_inventory_status_counts( active_store_count: int, shortage_ratio: float = SHORTAGE_RATIO, excess_ratio: float = EXCESS_RATIO, ) -> tuple[int, int]:
    """
    Calculate the number of shortage and excess stores.

    At least one shortage store and one excess store are created
    when a product has sales in two or more stores.
    """

    if active_store_count < 0:
        raise ValueError(
            "active_store_count must not be negative."
        )

    if not 0 <= shortage_ratio <= 1:
        raise ValueError(
            "shortage_ratio must be between 0 and 1."
        )

    if not 0 <= excess_ratio <= 1:
        raise ValueError(
            "excess_ratio must be between 0 and 1."
        )

    if shortage_ratio + excess_ratio > 1:
        raise ValueError(
            "The sum of shortage_ratio and excess_ratio "
            "must not exceed 1."
        )

    if active_store_count < 2:
        return 0, 0

    shortage_count = max(
        1,
        round(active_store_count * shortage_ratio),
    )

    excess_count = max(
        1,
        round(active_store_count * excess_ratio),
    )

    while shortage_count + excess_count > active_store_count:
        if shortage_count >= excess_count and shortage_count > 1:
            shortage_count -= 1
        elif excess_count > 1:
            excess_count -= 1
        else:
            break

    return shortage_count, excess_count


def generate_stock_level( average_daily_sales: float, status: str, min_days: int, max_days: int, rng: np.random.Generator, ) -> int:
    """
    Generate a stock level for a requested inventory status.

    Supported statuses are shortage, balanced, and excess.
    """

    if average_daily_sales < 0:
        raise ValueError(
            "average_daily_sales must not be negative."
        )

    if average_daily_sales == 0:
        return 0

    minimum_stock = int(
        np.ceil(average_daily_sales * min_days)
    )

    maximum_stock = int(
        np.floor(average_daily_sales * max_days)
    )

    if status == "shortage":
        # Generate approximately 30% to 80% of minimum stock.
        lower_bound = int(
            np.floor(minimum_stock * 0.30)
        )

        upper_bound = min(
            minimum_stock - 1,
            int(np.floor(minimum_stock * 0.80)),
        )

        lower_bound = max(0, lower_bound)
        upper_bound = max(lower_bound, upper_bound)

        return int(
            rng.integers(
                lower_bound,
                upper_bound + 1,
            )
        )

    if status == "balanced":
        lower_bound = minimum_stock
        upper_bound = max(
            lower_bound,
            maximum_stock,
        )

        return int(
            rng.integers(
                lower_bound,
                upper_bound + 1,
            )
        )

    if status == "excess":
        minimum_excess_stock = maximum_stock + 1

        extra_stock_lower = max(
            1,
            int(
                np.ceil(
                    average_daily_sales
                    * max_days
                    * 0.20
                )
            ),
        )

        extra_stock_upper = max(
            extra_stock_lower,
            int(
                np.ceil(
                    average_daily_sales
                    * max_days
                    * 0.70
                )
            ),
        )

        extra_stock = int(
            rng.integers(
                extra_stock_lower,
                extra_stock_upper + 1,
            )
        )

        return minimum_excess_stock + extra_stock

    raise ValueError(
        f"Unsupported inventory status: {status}"
    )


def generate_inventory(
    sales: pd.DataFrame,
    min_days: int = MIN_INVENTORY_DAYS,
    max_days: int = MAX_INVENTORY_DAYS,
    shortage_ratio: float = SHORTAGE_RATIO,
    excess_ratio: float = EXCESS_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate current inventory levels from historical sales.

    For every product, stores with positive demand are divided into
    shortage, excess, and balanced groups.

    Args:
        sales:
            Historical daily sales data.
        min_days:
            Inventory coverage below this value is a shortage.
        max_days:
            Inventory coverage above this value is excess.
        shortage_ratio:
            Ratio of active stores assigned shortage inventory.
        excess_ratio:
            Ratio of active stores assigned excess inventory.
        seed:
            Random seed used for reproducibility.

    Returns:
        A DataFrame containing current inventory levels.
    """

    validate_config()
    validate_inventory_input_data(sales)

    if min_days <= 0:
        raise ValueError(
            "min_days must be greater than zero."
        )

    if max_days <= min_days:
        raise ValueError(
            "max_days must be greater than min_days."
        )

    # Work on a copy to avoid modifying the caller's DataFrame.
    prepared_sales = sales.copy()

    prepared_sales["date"] = pd.to_datetime(
        prepared_sales["date"]
    )

    average_sales = (
        prepared_sales
        .groupby(
            ["store_id", "product_id"],
            as_index=False,
        )["quantity_sold"]
        .mean()
        .rename(
            columns={
                "quantity_sold": "average_daily_sales",
            }
        )
    )

    last_updated = prepared_sales["date"].max()
    rng = np.random.default_rng(seed)

    inventory_records: list[dict[str, object]] = []

    product_ids = sorted(
        average_sales["product_id"].unique()
    )

    for product_id in product_ids:
        product_sales = average_sales.loc[
            average_sales["product_id"] == product_id
        ].copy()

        active_product_sales = product_sales.loc[
            product_sales["average_daily_sales"] > 0
        ]

        active_store_ids = (
            active_product_sales["store_id"]
            .to_numpy(copy=True)
        )

        shortage_count, excess_count = (
            calculate_inventory_status_counts(
                active_store_count=len(active_store_ids),
                shortage_ratio=shortage_ratio,
                excess_ratio=excess_ratio,
            )
        )

        shuffled_store_ids = rng.permutation(
            active_store_ids
        )

        shortage_store_ids = set(
            shuffled_store_ids[:shortage_count]
        )

        excess_store_ids = set(
            shuffled_store_ids[
                shortage_count:
                shortage_count + excess_count
            ]
        )

        for row in product_sales.itertuples(index=False):
            store_id = row.store_id
            average_daily_sales = float(
                row.average_daily_sales
            )

            if average_daily_sales == 0:
                status = "balanced"
            elif store_id in shortage_store_ids:
                status = "shortage"
            elif store_id in excess_store_ids:
                status = "excess"
            else:
                status = "balanced"

            current_stock = generate_stock_level(
                average_daily_sales=average_daily_sales,
                status=status,
                min_days=min_days,
                max_days=max_days,
                rng=rng,
            )

            inventory_records.append(
                {
                    "store_id": store_id,
                    "product_id": product_id,
                    "current_stock": current_stock,
                    "last_updated": last_updated,
                }
            )

    inventory = pd.DataFrame(
        inventory_records,
        columns=INVENTORY_COLUMNS,
    )

    expected_count = len(
        average_sales[
            ["store_id", "product_id"]
        ].drop_duplicates()
    )

    validate_generated_inventory(
        inventory=inventory,
        expected_count=expected_count,
    )

    return inventory


def validate_generated_inventory( inventory: pd.DataFrame, expected_count: int | None = None, ) -> None:
    """Validate a generated inventory DataFrame."""

    missing_columns = (
        set(INVENTORY_COLUMNS) - set(inventory.columns)
    )

    if missing_columns:
        raise ValueError(
            "Generated inventory data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if inventory.empty:
        raise ValueError(
            "Generated inventory data must not be empty."
        )

    if (
        expected_count is not None
        and len(inventory) != expected_count
    ):
        raise ValueError(
            f"Expected {expected_count} inventory records, "
            f"but generated {len(inventory)}."
        )

    if inventory.duplicated(
        subset=["store_id", "product_id"]
    ).any():
        raise ValueError(
            "Inventory data must contain only one record "
            "for each store-product pair."
        )

    if inventory[list(INVENTORY_COLUMNS)].isna().any().any():
        raise ValueError(
            "Generated inventory data must not contain "
            "missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        inventory["current_stock"]
    ):
        raise ValueError(
            "current_stock must be numeric."
        )

    if (inventory["current_stock"] < 0).any():
        raise ValueError(
            "Current stock must not be negative."
        )

    if (
        inventory["current_stock"]
        % 1
        != 0
    ).any():
        raise ValueError(
            "Current stock must contain integer values."
        )


def save_inventory( inventory: pd.DataFrame, output_path: str | Path = INVENTORY_FILE, ) -> Path:
    """Validate and save inventory data to a CSV file."""

    validate_generated_inventory(inventory)

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    return destination.resolve()


def load_sales_data() -> pd.DataFrame:
    """Load historical sales data required by the generator."""

    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales data was not found at {SALES_FILE}. "
            "Run the sales generator first."
        )

    return pd.read_csv(
        SALES_FILE,
        parse_dates=["date"],
    )


def main() -> None:
    """Load sales, generate inventory, and save the result."""

    sales = load_sales_data()
    inventory = generate_inventory(sales)
    output_path = save_inventory(inventory)

    print(f"Generated {len(inventory):,} inventory records.")
    print(f"Saved inventory data to: {output_path}")
    print(
        "Total inventory units: "
        f"{inventory['current_stock'].sum():,}"
    )


if __name__ == "__main__":
    main()