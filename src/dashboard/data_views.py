"""Prepare enriched data tables for dashboard views."""

import pandas as pd


STORE_LOOKUP_COLUMNS = (
    "store_id",
    "store_name",
    "city",
)

PRODUCT_LOOKUP_COLUMNS = (
    "product_id",
    "product_name",
    "category",
)

FORECAST_REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
    "forecast_date",
    "forecast_day",
    "predicted_quantity",
    "method",
)

INVENTORY_REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
    "status",
)

TRANSFER_REQUIRED_COLUMNS = (
    "product_id",
    "from_store_id",
    "to_store_id",
)


def validate_view_data(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    data_name: str,
    allow_empty: bool = False,
) -> None:
    """Validate a DataFrame used by a dashboard view."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{data_name} must be a pandas DataFrame."
        )

    if not isinstance(allow_empty, bool):
        raise TypeError(
            "allow_empty must be a boolean."
        )

    missing_columns = (
        set(required_columns)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        if allow_empty:
            return

        raise ValueError(
            f"{data_name} must not be empty."
        )

    if data[
        list(required_columns)
    ].isna().any().any():
        raise ValueError(
            f"{data_name} contains missing values "
            "in required columns."
        )


def validate_lookup_data(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    id_column: str,
    data_name: str,
) -> None:
    """Validate store or product lookup data."""

    validate_view_data(
        data=data,
        required_columns=required_columns,
        data_name=data_name,
        allow_empty=False,
    )

    if data[id_column].duplicated().any():
        raise ValueError(
            f"{data_name} contains duplicate "
            f"{id_column} values."
        )


def validate_known_ids(
    values: pd.Series,
    valid_values: pd.Series,
    value_name: str,
) -> None:
    """Validate IDs against a reference table."""

    unknown_values = (
        set(values)
        - set(valid_values)
    )

    if unknown_values:
        display_values = sorted(
            str(value)
            for value in unknown_values
        )

        raise ValueError(
            f"Unknown {value_name} values: "
            f"{display_values}"
        )


def create_store_lookup(
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Create a validated store lookup table."""

    validate_lookup_data(
        data=stores,
        required_columns=STORE_LOOKUP_COLUMNS,
        id_column="store_id",
        data_name="stores",
    )

    return stores[
        list(STORE_LOOKUP_COLUMNS)
    ].copy()


def create_product_lookup(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Create a validated product lookup table."""

    validate_lookup_data(
        data=products,
        required_columns=PRODUCT_LOOKUP_COLUMNS,
        id_column="product_id",
        data_name="products",
    )

    return products[
        list(PRODUCT_LOOKUP_COLUMNS)
    ].copy()


def enrich_forecast_data(
    forecast: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Add store and product information to forecasts."""

    validate_view_data(
        data=forecast,
        required_columns=FORECAST_REQUIRED_COLUMNS,
        data_name="forecast",
    )

    store_lookup = create_store_lookup(
        stores
    )

    product_lookup = create_product_lookup(
        products
    )

    validate_known_ids(
        values=forecast["store_id"],
        valid_values=store_lookup["store_id"],
        value_name="store_id",
    )

    validate_known_ids(
        values=forecast["product_id"],
        valid_values=(
            product_lookup["product_id"]
        ),
        value_name="product_id",
    )

    result = forecast.copy()

    existing_metadata_columns = [
        column
        for column in (
            "store_name",
            "city",
            "product_name",
            "category",
        )
        if column in result.columns
    ]

    if existing_metadata_columns:
        result = result.drop(
            columns=existing_metadata_columns
        )

    result["forecast_date"] = (
        result["forecast_date"].map(
            lambda value: pd.to_datetime(
                value,
                errors="coerce",
            )
        )
    )

    if result["forecast_date"].isna().any():
        raise ValueError(
            "forecast contains invalid "
            "forecast_date values."
        )

    result["forecast_date"] = (
        result["forecast_date"].dt.normalize()
    )

    result = result.merge(
        store_lookup,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    return result


def enrich_inventory_data(
    inventory_analysis: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Add store and product information to inventory."""

    validate_view_data(
        data=inventory_analysis,
        required_columns=(
            INVENTORY_REQUIRED_COLUMNS
        ),
        data_name="inventory_analysis",
    )

    store_lookup = create_store_lookup(
        stores
    )

    product_lookup = create_product_lookup(
        products
    )

    validate_known_ids(
        values=inventory_analysis["store_id"],
        valid_values=store_lookup["store_id"],
        value_name="store_id",
    )

    validate_known_ids(
        values=inventory_analysis["product_id"],
        valid_values=(
            product_lookup["product_id"]
        ),
        value_name="product_id",
    )

    result = inventory_analysis.copy()

    existing_metadata_columns = [
        column
        for column in (
            "store_name",
            "city",
            "product_name",
            "category",
        )
        if column in result.columns
    ]

    if existing_metadata_columns:
        result = result.drop(
            columns=existing_metadata_columns
        )

    result = result.merge(
        store_lookup,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    return result


def enrich_transfer_data(
    transfer_plan: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Add names and cities to a transfer plan."""

    validate_view_data(
        data=transfer_plan,
        required_columns=TRANSFER_REQUIRED_COLUMNS,
        data_name="transfer_plan",
        allow_empty=True,
    )

    store_lookup = create_store_lookup(
        stores
    )

    product_lookup = create_product_lookup(
        products
    )

    if not transfer_plan.empty:
        validate_known_ids(
            values=transfer_plan[
                "from_store_id"
            ],
            valid_values=(
                store_lookup["store_id"]
            ),
            value_name="from_store_id",
        )

        validate_known_ids(
            values=transfer_plan[
                "to_store_id"
            ],
            valid_values=(
                store_lookup["store_id"]
            ),
            value_name="to_store_id",
        )

        validate_known_ids(
            values=transfer_plan["product_id"],
            valid_values=(
                product_lookup["product_id"]
            ),
            value_name="product_id",
        )

    source_store_lookup = (
        store_lookup.rename(
            columns={
                "store_id": "from_store_id",
                "store_name": (
                    "from_store_name"
                ),
                "city": "from_city",
            }
        )
    )

    destination_store_lookup = (
        store_lookup.rename(
            columns={
                "store_id": "to_store_id",
                "store_name": "to_store_name",
                "city": "to_city",
            }
        )
    )

    result = transfer_plan.copy()

    existing_metadata_columns = [
        column
        for column in (
            "product_name",
            "category",
            "from_store_name",
            "from_city",
            "to_store_name",
            "to_city",
        )
        if column in result.columns
    ]

    if existing_metadata_columns:
        result = result.drop(
            columns=existing_metadata_columns
        )

    result = result.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        source_store_lookup,
        on="from_store_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        destination_store_lookup,
        on="to_store_id",
        how="left",
        validate="many_to_one",
    )

    return result