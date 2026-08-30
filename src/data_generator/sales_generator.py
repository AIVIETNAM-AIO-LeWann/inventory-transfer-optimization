"""Generate sample sales data for the inventory optimization system."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MAX_DAILY_DEMAND,
    MIN_DAILY_DEMAND,
    PRODUCTS_FILE,
    RANDOM_SEED,
    SALES_DAYS,
    SALES_FILE,
    SALES_START_DATE,
    STORES_FILE,
    WEEKEND_SALES_MULTIPLIER,
    validate_config,
)


CITY_CATEGORY_FACTORS = {
    "Hanoi": {
        "Electronics": 1.20,
        "Clothing": 1.30,
        "Home Goods": 0.90,
        "Food": 1.10,
        "Beauty": 1.10,
    },
    "Da Nang": {
        "Electronics": 0.80,
        "Clothing": 1.00,
        "Home Goods": 1.25,
        "Food": 1.10,
        "Beauty": 1.20,
    },
    "Ho Chi Minh City": {
        "Electronics": 1.30,
        "Clothing": 1.10,
        "Home Goods": 1.00,
        "Food": 1.30,
        "Beauty": 1.15,
    },
}


REQUIRED_STORE_COLUMNS = (
    "store_id",
    "city",
)

REQUIRED_PRODUCT_COLUMNS = (
    "product_id",
    "category",
    "cost",
    "price",
)

SALES_COLUMNS = (
    "date",
    "store_id",
    "product_id",
    "quantity_sold",
    "revenue",
    "cost_of_goods_sold",
)


def validate_sales_input_data( stores: pd.DataFrame, products: pd.DataFrame, ) -> None:
    """Validate store and product data used to generate sales."""

    missing_store_columns = (
        set(REQUIRED_STORE_COLUMNS) - set(stores.columns)
    )

    if missing_store_columns:
        raise ValueError(
            "Store data is missing columns: "
            f"{sorted(missing_store_columns)}"
        )

    missing_product_columns = (
        set(REQUIRED_PRODUCT_COLUMNS) - set(products.columns)
    )

    if missing_product_columns:
        raise ValueError(
            "Product data is missing columns: "
            f"{sorted(missing_product_columns)}"
        )

    if stores.empty:
        raise ValueError(
            "Store data must not be empty."
        )

    if products.empty:
        raise ValueError(
            "Product data must not be empty."
        )

    if stores["store_id"].duplicated().any():
        raise ValueError(
            "Store IDs must be unique."
        )

    if products["product_id"].duplicated().any():
        raise ValueError(
            "Product IDs must be unique."
        )

    if stores[list(REQUIRED_STORE_COLUMNS)].isna().any().any():
        raise ValueError(
            "Store data must not contain missing values."
        )

    if products[list(REQUIRED_PRODUCT_COLUMNS)].isna().any().any():
        raise ValueError(
            "Product data must not contain missing values."
        )

    if (products["cost"] <= 0).any():
        raise ValueError(
            "Product costs must be greater than zero."
        )

    if (products["price"] <= products["cost"]).any():
        raise ValueError(
            "Product prices must be greater than product costs."
        )


def get_seasonality_factor( category: str, month: int, ) -> float:
    """Return a seasonal demand multiplier for a product category."""

    factor = 1.0

    # Year-end and Tet-related demand
    if month in (1, 2, 12):
        if category == "Electronics":
            factor *= 1.35
        elif category == "Home Goods":
            factor *= 1.20
        elif category == "Food":
            factor *= 1.15

    # Summer demand
    if month in (6, 7, 8):
        if category == "Clothing":
            factor *= 1.20
        elif category == "Beauty":
            factor *= 1.25
        elif category == "Food":
            factor *= 1.10

    return factor


def generate_sales( stores: pd.DataFrame, products: pd.DataFrame, days: int = SALES_DAYS, start_date: str = SALES_START_DATE, seed: int = RANDOM_SEED, ) -> pd.DataFrame:
    """
    Generate daily sales for every store-product combination.

    Args:
        stores:
            Store data containing store IDs and cities.
        products:
            Product data containing IDs, categories, costs, and prices.
        days:
            Number of sales days to generate.
        start_date:
            First date of the generated sales period.
        seed:
            Random seed used for reproducibility.

    Returns:
        A DataFrame containing generated sales records.
    """

    validate_config()
    validate_sales_input_data(stores, products)

    if days <= 0:
        raise ValueError(
            "days must be greater than zero."
        )

    try:
        first_date = pd.Timestamp(start_date)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid start date: {start_date}"
        ) from error

    dates = pd.date_range(
        start=first_date,
        periods=days,
        freq="D",
    )

    rng = np.random.default_rng(seed)

    store_rows = list(
        stores[list(REQUIRED_STORE_COLUMNS)].itertuples(
            index=False,
        )
    )

    product_rows = list(
        products[list(REQUIRED_PRODUCT_COLUMNS)].itertuples(
            index=False,
        )
    )

    # Each store has a stable demand level across the sales period.
    store_demand_factors = {
        store.store_id: rng.uniform(0.80, 1.20)
        for store in store_rows
    }

    # Each product has its own stable base daily demand.
    product_base_demands = {
        product.product_id: rng.uniform(
            MIN_DAILY_DEMAND,
            MAX_DAILY_DEMAND,
        )
        for product in product_rows
    }

    sales_records: list[dict[str, object]] = []

    for sale_date in dates:
        weekend_factor = (
            WEEKEND_SALES_MULTIPLIER
            if sale_date.dayofweek >= 5
            else 1.0
        )

        for store in store_rows:
            store_factor = store_demand_factors[store.store_id]

            city_factors = CITY_CATEGORY_FACTORS.get(
                store.city,
                {},
            )

            for product in product_rows:
                base_demand = product_base_demands[
                    product.product_id
                ]

                city_category_factor = city_factors.get(
                    product.category,
                    1.0,
                )

                seasonality_factor = get_seasonality_factor(
                    category=product.category,
                    month=sale_date.month,
                )

                expected_quantity = (
                    base_demand
                    * store_factor
                    * city_category_factor
                    * weekend_factor
                    * seasonality_factor
                )

                quantity_sold = int(
                    rng.poisson(expected_quantity)
                )

                revenue = int(
                    quantity_sold * product.price
                )

                cost_of_goods_sold = int(
                    quantity_sold * product.cost
                )

                sales_records.append(
                    {
                        "date": sale_date,
                        "store_id": store.store_id,
                        "product_id": product.product_id,
                        "quantity_sold": quantity_sold,
                        "revenue": revenue,
                        "cost_of_goods_sold": (
                            cost_of_goods_sold
                        ),
                    }
                )

    sales = pd.DataFrame(
        sales_records,
        columns=SALES_COLUMNS,
    )

    expected_count = (
        days
        * len(stores)
        * len(products)
    )

    validate_generated_sales( sales=sales, expected_count=expected_count, expected_days_per_pair=days, )

    return sales


def validate_generated_sales( sales: pd.DataFrame, expected_count: int | None = None, expected_days_per_pair: int | None = None, ) -> None:
    """Validate a generated sales DataFrame."""

    missing_columns = set(SALES_COLUMNS) - set(sales.columns)

    if missing_columns:
        raise ValueError(
            "Generated sales data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if sales.empty:
        raise ValueError(
            "Generated sales data must not be empty."
        )

    if (
        expected_count is not None
        and len(sales) != expected_count
    ):
        raise ValueError(
            f"Expected {expected_count} sales records, "
            f"but generated {len(sales)}."
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

    if (sales["revenue"] < 0).any():
        raise ValueError(
            "Sales revenue must not be negative."
        )

    if (sales["cost_of_goods_sold"] < 0).any():
        raise ValueError(
            "Cost of goods sold must not be negative."
        )

    if sales[list(SALES_COLUMNS)].isna().any().any():
        raise ValueError(
            "Generated sales data must not contain missing values."
        )

    if expected_days_per_pair is not None:
        pair_record_counts = sales.groupby(
            ["store_id", "product_id"]
        ).size()

        if not pair_record_counts.eq(
            expected_days_per_pair
        ).all():
            raise ValueError(
                "Every store-product pair must contain "
                f"{expected_days_per_pair} daily records."
            )


def save_sales( sales: pd.DataFrame, output_path: str | Path = SALES_FILE, ) -> Path:
    """Validate and save sales data to a CSV file."""

    validate_generated_sales(sales)

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sales.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    return destination.resolve()


def load_generation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load store and product data required by the sales generator."""

    missing_files = [
        path
        for path in (STORES_FILE, PRODUCTS_FILE)
        if not path.exists()
    ]

    if missing_files:
        missing_file_names = [
            path.name
            for path in missing_files
        ]

        raise FileNotFoundError(
            "Missing required input files: "
            f"{missing_file_names}. "
            "Run the store and product generators first."
        )

    stores = pd.read_csv(STORES_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    return stores, products


def main() -> None:
    """Load inputs, generate sales, and save the result."""

    stores, products = load_generation_inputs()

    sales = generate_sales(
        stores=stores,
        products=products,
    )

    output_path = save_sales(sales)

    print(f"Generated {len(sales):,} sales records.")
    print(f"Saved sales data to: {output_path}")
    print(
        "Date range: "
        f"{sales['date'].min().date()} to "
        f"{sales['date'].max().date()}"
    )
    print(
        f"Total units sold: "
        f"{sales['quantity_sold'].sum():,}"
    )
    print(
        f"Total revenue: "
        f"{sales['revenue'].sum():,.0f} VND"
    )


if __name__ == "__main__":
    main()