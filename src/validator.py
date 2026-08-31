"""Validate input data used by the optimization system."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import PRODUCT_CATEGORIES


STORE_COLUMNS = (
    "store_id",
    "store_name",
    "city",
    "latitude",
    "longitude",
)

PRODUCT_COLUMNS = (
    "product_id",
    "product_name",
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

INVENTORY_COLUMNS = (
    "store_id",
    "product_id",
    "current_stock",
    "last_updated",
)


def validate_table(
    table: pd.DataFrame,
    table_name: str,
    required_columns: Sequence[str],
) -> None:
    """Validate the basic structure of a data table."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"{table_name} must be a pandas DataFrame."
        )

    if table.empty:
        raise ValueError(
            f"{table_name} must not be empty."
        )

    missing_columns = (
        set(required_columns) - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if table[list(required_columns)].isna().any().any():
        raise ValueError(
            f"{table_name} contains missing values."
        )


def validate_unique_key(
    table: pd.DataFrame,
    key_columns: Sequence[str],
    table_name: str,
) -> None:
    """Validate that key columns do not contain duplicates."""

    duplicated_rows = table.duplicated(
        subset=list(key_columns),
        keep=False,
    )

    if duplicated_rows.any():
        raise ValueError(
            f"{table_name} contains duplicate keys "
            f"for columns {list(key_columns)}."
        )


def validate_text_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    """Validate that text columns do not contain blank strings."""

    for column in columns:
        blank_values = (
            table[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if blank_values.any():
            raise ValueError(
                f"{table_name}.{column} "
                "must not contain blank values."
            )


def validate_non_negative_numeric_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    """Validate numeric columns and reject negative values."""

    for column in columns:
        if not pd.api.types.is_numeric_dtype(
            table[column]
        ):
            raise TypeError(
                f"{table_name}.{column} must be numeric."
            )

        if (table[column] < 0).any():
            raise ValueError(
                f"{table_name}.{column} "
                "must not contain negative values."
            )


def validate_integer_column(
    table: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    """Validate that a numeric column contains whole numbers."""

    values = table[column].to_numpy(dtype=float)

    if not np.allclose(values, np.round(values)):
        raise ValueError(
            f"{table_name}.{column} "
            "must contain whole numbers."
        )


def validate_date_column(
    table: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    """Validate that every value can be converted to a date."""

    converted_dates = pd.to_datetime(
        table[column],
        errors="coerce",
    )

    if converted_dates.isna().any():
        raise ValueError(
            f"{table_name}.{column} "
            "contains invalid date values."
        )


def validate_reference_ids(
    values: pd.Series,
    valid_values: pd.Series,
    value_name: str,
) -> None:
    """Validate that IDs exist in their parent table."""

    provided_ids = set(values.astype(str))
    valid_ids = set(valid_values.astype(str))

    unknown_ids = sorted(provided_ids - valid_ids)

    if unknown_ids:
        raise ValueError(
            f"{value_name} contains unknown IDs: "
            f"{unknown_ids[:10]}"
        )


def validate_stores(
    stores: pd.DataFrame,
) -> None:
    """Validate store data."""

    validate_table(
        table=stores,
        table_name="stores",
        required_columns=STORE_COLUMNS,
    )

    validate_unique_key(
        table=stores,
        key_columns=("store_id",),
        table_name="stores",
    )

    validate_text_columns(
        table=stores,
        columns=("store_id", "store_name", "city"),
        table_name="stores",
    )

    for column in ("latitude", "longitude"):
        if not pd.api.types.is_numeric_dtype(
            stores[column]
        ):
            raise TypeError(
                f"stores.{column} must be numeric."
            )

    if not stores["latitude"].between(-90, 90).all():
        raise ValueError(
            "stores.latitude must be between -90 and 90."
        )

    if not stores["longitude"].between(
        -180,
        180,
    ).all():
        raise ValueError(
            "stores.longitude must be between -180 and 180."
        )


def validate_products(
    products: pd.DataFrame,
) -> None:
    """Validate product data."""

    validate_table(
        table=products,
        table_name="products",
        required_columns=PRODUCT_COLUMNS,
    )

    validate_unique_key(
        table=products,
        key_columns=("product_id",),
        table_name="products",
    )

    validate_text_columns(
        table=products,
        columns=(
            "product_id",
            "product_name",
            "category",
        ),
        table_name="products",
    )

    validate_non_negative_numeric_columns(
        table=products,
        columns=("cost", "price"),
        table_name="products",
    )

    invalid_categories = (
        set(products["category"])
        - set(PRODUCT_CATEGORIES)
    )

    if invalid_categories:
        raise ValueError(
            "products.category contains unsupported values: "
            f"{sorted(invalid_categories)}"
        )

    if (products["price"] < products["cost"]).any():
        raise ValueError(
            "Product prices must not be lower than costs."
        )


def validate_sales(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    """Validate sales data and its relationships."""

    validate_table(
        table=sales,
        table_name="sales",
        required_columns=SALES_COLUMNS,
    )

    validate_unique_key(
        table=sales,
        key_columns=(
            "date",
            "store_id",
            "product_id",
        ),
        table_name="sales",
    )

    validate_date_column(
        table=sales,
        column="date",
        table_name="sales",
    )

    validate_non_negative_numeric_columns(
        table=sales,
        columns=(
            "quantity_sold",
            "revenue",
            "cost_of_goods_sold",
        ),
        table_name="sales",
    )

    validate_integer_column(
        table=sales,
        column="quantity_sold",
        table_name="sales",
    )

    validate_reference_ids(
        values=sales["store_id"],
        valid_values=stores["store_id"],
        value_name="sales.store_id",
    )

    validate_reference_ids(
        values=sales["product_id"],
        valid_values=products["product_id"],
        value_name="sales.product_id",
    )


def validate_inventory(
    inventory: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    """Validate inventory data and its relationships."""

    validate_table(
        table=inventory,
        table_name="inventory",
        required_columns=INVENTORY_COLUMNS,
    )

    validate_unique_key(
        table=inventory,
        key_columns=("store_id", "product_id"),
        table_name="inventory",
    )

    validate_date_column(
        table=inventory,
        column="last_updated",
        table_name="inventory",
    )

    validate_non_negative_numeric_columns(
        table=inventory,
        columns=("current_stock",),
        table_name="inventory",
    )

    validate_integer_column(
        table=inventory,
        column="current_stock",
        table_name="inventory",
    )

    validate_reference_ids(
        values=inventory["store_id"],
        valid_values=stores["store_id"],
        value_name="inventory.store_id",
    )

    validate_reference_ids(
        values=inventory["product_id"],
        valid_values=products["product_id"],
        value_name="inventory.product_id",
    )


def validate_route_matrix(
    matrix: pd.DataFrame,
    stores: pd.DataFrame,
    matrix_name: str,
) -> None:
    """Validate a store-to-store route matrix."""

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(
            f"{matrix_name} must be a pandas DataFrame."
        )

    if matrix.empty:
        raise ValueError(
            f"{matrix_name} must not be empty."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{matrix_name} must be square."
        )

    row_ids = matrix.index.astype(str).tolist()
    column_ids = matrix.columns.astype(str).tolist()

    if row_ids != column_ids:
        raise ValueError(
            f"{matrix_name} row and column IDs must match."
        )

    expected_store_ids = set(
        stores["store_id"].astype(str)
    )

    if set(row_ids) != expected_store_ids:
        raise ValueError(
            f"{matrix_name} store IDs do not match stores."
        )

    try:
        values = matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{matrix_name} must contain numeric values."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            f"{matrix_name} contains invalid values."
        )

    if (values < 0).any():
        raise ValueError(
            f"{matrix_name} contains negative values."
        )

    if not np.allclose(
        np.diag(values),
        0.0,
    ):
        raise ValueError(
            f"{matrix_name} diagonal must be zero."
        )


def validate_all_data(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    duration_matrix: pd.DataFrame,
    transport_cost_matrix: pd.DataFrame,
) -> None:
    """Validate all project datasets."""

    validate_stores(stores)
    validate_products(products)

    validate_sales(
        sales=sales,
        stores=stores,
        products=products,
    )

    validate_inventory(
        inventory=inventory,
        stores=stores,
        products=products,
    )

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