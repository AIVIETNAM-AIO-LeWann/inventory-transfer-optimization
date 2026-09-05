"""Prepare chronological datasets for forecasting models."""

from dataclasses import dataclass

import pandas as pd

from src.config import (
    FORECAST_LAG_DAYS,
    FORECAST_ROLLING_WINDOWS,
    MODEL_TEST_RATIO,
    MODEL_TRAIN_RATIO,
    MODEL_VALIDATION_RATIO,
)
from src.data_loader import load_all_data
from src.forecasting.feature_engineering import (
    CALENDAR_FEATURE_COLUMNS,
    create_time_series_features,
)


TARGET_COLUMN = "quantity_sold"

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

CATEGORICAL_FEATURE_COLUMNS = (
    "store_id",
    "product_id",
    "city",
    "category",
)

LAG_FEATURE_COLUMNS = tuple(
    f"lag_{lag_day}"
    for lag_day in FORECAST_LAG_DAYS
)

ROLLING_FEATURE_COLUMNS = tuple(
    f"rolling_mean_{window}"
    for window in FORECAST_ROLLING_WINDOWS
)

NUMERICAL_FEATURE_COLUMNS = (
    *CALENDAR_FEATURE_COLUMNS,
    *LAG_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
    "cost",
    "price",
)

MODEL_FEATURE_COLUMNS = (
    *CATEGORICAL_FEATURE_COLUMNS,
    *NUMERICAL_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class TrainingDataset:
    """Store chronological model datasets and column names."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    numerical_feature_columns: tuple[str, ...]
    target_column: str


def validate_reference_data(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    id_column: str,
    data_name: str,
) -> None:
    """Validate store or product reference data."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{data_name} must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            f"{data_name} must not be empty."
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

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            f"{data_name} contains missing values."
        )

    if data[id_column].duplicated().any():
        raise ValueError(
            f"{data_name} contains duplicate "
            f"{id_column} values."
        )


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate train, validation, and test ratios."""

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    invalid_ratios = [
        ratio
        for ratio in ratios
        if (
            isinstance(ratio, bool)
            or not isinstance(
                ratio,
                (int, float),
            )
            or not 0 < ratio < 1
        )
    ]

    if invalid_ratios:
        raise ValueError(
            "Split ratios must be numbers "
            "between zero and one."
        )

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(
            "Split ratios must sum to 1.0."
        )


def prepare_model_data(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Create model features and merge reference data."""

    validate_reference_data(
        data=stores,
        required_columns=REQUIRED_STORE_COLUMNS,
        id_column="store_id",
        data_name="stores",
    )

    validate_reference_data(
        data=products,
        required_columns=REQUIRED_PRODUCT_COLUMNS,
        id_column="product_id",
        data_name="products",
    )

    feature_data = create_time_series_features(
        sales=sales,
        drop_incomplete_rows=True,
    )

    store_features = stores[
        list(REQUIRED_STORE_COLUMNS)
    ].copy()

    product_features = products[
        list(REQUIRED_PRODUCT_COLUMNS)
    ].copy()

    feature_data["store_id"] = (
        feature_data["store_id"].astype(str)
    )
    feature_data["product_id"] = (
        feature_data["product_id"].astype(str)
    )

    store_features["store_id"] = (
        store_features["store_id"].astype(str)
    )
    product_features["product_id"] = (
        product_features["product_id"].astype(str)
    )

    missing_store_ids = (
        set(feature_data["store_id"])
        - set(store_features["store_id"])
    )

    if missing_store_ids:
        raise ValueError(
            "Sales contains unknown store IDs: "
            f"{sorted(missing_store_ids)}"
        )

    missing_product_ids = (
        set(feature_data["product_id"])
        - set(product_features["product_id"])
    )

    if missing_product_ids:
        raise ValueError(
            "Sales contains unknown product IDs: "
            f"{sorted(missing_product_ids)}"
        )

    model_data = feature_data.merge(
        store_features,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    model_data = model_data.merge(
        product_features,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    for column in ("cost", "price"):
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    if model_data[
        list(MODEL_FEATURE_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Prepared model data contains "
            "missing feature values."
        )

    output_columns = (
        "date",
        *MODEL_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

    return model_data[
        list(output_columns)
    ].sort_values(
        by=[
            "date",
            "store_id",
            "product_id",
        ],
        ignore_index=True,
    )


def split_model_data_by_date(
    model_data: pd.DataFrame,
    train_ratio: float = MODEL_TRAIN_RATIO,
    validation_ratio: float = (
        MODEL_VALIDATION_RATIO
    ),
    test_ratio: float = MODEL_TEST_RATIO,
) -> TrainingDataset:
    """Split model data chronologically by unique dates."""

    validate_split_ratios(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    if not isinstance(model_data, pd.DataFrame):
        raise TypeError(
            "model_data must be a pandas DataFrame."
        )

    if model_data.empty:
        raise ValueError(
            "model_data must not be empty."
        )

    required_columns = {
        "date",
        TARGET_COLUMN,
        *MODEL_FEATURE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(model_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "model_data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    working_data = model_data.copy()

    working_data["date"] = pd.to_datetime(
        working_data["date"],
        errors="coerce",
    ).dt.normalize()

    if working_data["date"].isna().any():
        raise ValueError(
            "model_data contains invalid dates."
        )

    unique_dates = sorted(
        working_data["date"].unique()
    )

    number_of_dates = len(unique_dates)

    train_date_count = int(
        number_of_dates * train_ratio
    )

    validation_date_count = int(
        number_of_dates * validation_ratio
    )

    test_date_count = (
        number_of_dates
        - train_date_count
        - validation_date_count
    )

    if min(
        train_date_count,
        validation_date_count,
        test_date_count,
    ) <= 0:
        raise ValueError(
            "Not enough dates to create all "
            "three model datasets."
        )

    train_dates = unique_dates[
        :train_date_count
    ]

    validation_start = train_date_count
    validation_end = (
        train_date_count
        + validation_date_count
    )

    validation_dates = unique_dates[
        validation_start:validation_end
    ]

    test_dates = unique_dates[
        validation_end:
    ]

    sort_columns = [
        "date",
        "store_id",
        "product_id",
    ]

    train_data = working_data.loc[
        working_data["date"].isin(train_dates)
    ].sort_values(
        sort_columns,
        ignore_index=True,
    )

    validation_data = working_data.loc[
        working_data["date"].isin(
            validation_dates
        )
    ].sort_values(
        sort_columns,
        ignore_index=True,
    )

    test_data = working_data.loc[
        working_data["date"].isin(test_dates)
    ].sort_values(
        sort_columns,
        ignore_index=True,
    )

    return TrainingDataset(
        train=train_data,
        validation=validation_data,
        test=test_data,
        feature_columns=MODEL_FEATURE_COLUMNS,
        categorical_feature_columns=(
            CATEGORICAL_FEATURE_COLUMNS
        ),
        numerical_feature_columns=(
            NUMERICAL_FEATURE_COLUMNS
        ),
        target_column=TARGET_COLUMN,
    )


def build_training_dataset(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> TrainingDataset:
    """Prepare and split the complete training dataset."""

    model_data = prepare_model_data(
        sales=sales,
        stores=stores,
        products=products,
    )

    return split_model_data_by_date(
        model_data=model_data,
    )


def main() -> None:
    """Build training datasets from project data."""

    project_data = load_all_data()

    dataset = build_training_dataset(
        sales=project_data.sales,
        stores=project_data.stores,
        products=project_data.products,
    )

    print("Training dataset created successfully.")
    print(
        f"Training rows: {len(dataset.train):,}"
    )
    print(
        "Validation rows: "
        f"{len(dataset.validation):,}"
    )
    print(
        f"Test rows: {len(dataset.test):,}"
    )

    print()
    print(
        "Training date range: "
        f"{dataset.train['date'].min().date()} "
        f"to {dataset.train['date'].max().date()}"
    )
    print(
        "Validation date range: "
        f"{dataset.validation['date'].min().date()} "
        f"to {dataset.validation['date'].max().date()}"
    )
    print(
        "Test date range: "
        f"{dataset.test['date'].min().date()} "
        f"to {dataset.test['date'].max().date()}"
    )

    print()
    print(
        "Model features: "
        f"{len(dataset.feature_columns)}"
    )
    print(dataset.feature_columns)


if __name__ == "__main__":
    main()