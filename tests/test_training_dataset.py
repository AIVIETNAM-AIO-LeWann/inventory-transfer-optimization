"""Tests for forecasting training datasets."""

import pandas as pd
import pytest

from src.forecasting.training_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_training_dataset,
    prepare_model_data,
    split_model_data_by_date,
    validate_reference_data,
    validate_split_ratios,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create thirty-five days of sales for two products."""

    dates = pd.date_range(
        "2026-01-01",
        periods=35,
        freq="D",
    )

    rows = []

    for day_number, date in enumerate(
        dates,
        start=1,
    ):
        for product_id, offset in (
            ("P001", 0),
            ("P002", 100),
        ):
            quantity_sold = (
                day_number + offset
            )

            rows.append(
                {
                    "date": date,
                    "store_id": "S001",
                    "product_id": product_id,
                    "quantity_sold": quantity_sold,
                    "revenue": quantity_sold * 100,
                    "cost_of_goods_sold": (
                        quantity_sold * 60
                    ),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_stores() -> pd.DataFrame:
    """Create sample store reference data."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "city": "Hanoi",
            }
        ]
    )


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """Create sample product reference data."""

    return pd.DataFrame(
        [
            {
                "product_id": "P001",
                "category": "Food",
                "cost": 60,
                "price": 100,
            },
            {
                "product_id": "P002",
                "category": "Electronics",
                "cost": 120,
                "price": 200,
            },
        ]
    )


def test_validate_reference_data_accepts_valid_data(
    sample_stores: pd.DataFrame,
) -> None:
    """Valid reference data should be accepted."""

    validate_reference_data(
        data=sample_stores,
        required_columns=(
            "store_id",
            "city",
        ),
        id_column="store_id",
        data_name="stores",
    )


def test_reference_data_must_be_dataframe() -> None:
    """Reference data should be a DataFrame."""

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        validate_reference_data(
            data=[],
            required_columns=(
                "store_id",
                "city",
            ),
            id_column="store_id",
            data_name="stores",
        )


def test_reference_data_must_not_be_empty() -> None:
    """Empty reference data should be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_reference_data(
            data=pd.DataFrame(),
            required_columns=(
                "store_id",
                "city",
            ),
            id_column="store_id",
            data_name="stores",
        )


def test_missing_reference_column_is_rejected(
    sample_stores: pd.DataFrame,
) -> None:
    """Missing reference columns should be rejected."""

    stores = sample_stores.drop(
        columns=["city"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_reference_data(
            data=stores,
            required_columns=(
                "store_id",
                "city",
            ),
            id_column="store_id",
            data_name="stores",
        )


def test_missing_reference_value_is_rejected(
    sample_stores: pd.DataFrame,
) -> None:
    """Missing reference values should be rejected."""

    stores = sample_stores.copy()
    stores.loc[0, "city"] = None

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        validate_reference_data(
            data=stores,
            required_columns=(
                "store_id",
                "city",
            ),
            id_column="store_id",
            data_name="stores",
        )


def test_duplicate_reference_id_is_rejected(
    sample_stores: pd.DataFrame,
) -> None:
    """Duplicate reference IDs should be rejected."""

    stores = pd.concat(
        [
            sample_stores,
            sample_stores,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate store_id",
    ):
        validate_reference_data(
            data=stores,
            required_columns=(
                "store_id",
                "city",
            ),
            id_column="store_id",
            data_name="stores",
        )


def test_prepare_model_data(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Model data should contain complete features."""

    model_data = prepare_model_data(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    )

    expected_rows = 2 * (35 - 28)

    assert len(model_data) == expected_rows

    assert model_data["date"].min() == (
        pd.Timestamp("2026-01-29")
    )

    assert all(
        column in model_data.columns
        for column in MODEL_FEATURE_COLUMNS
    )

    assert TARGET_COLUMN in model_data.columns

    assert set(model_data["city"]) == {
        "Hanoi"
    }

    assert set(model_data["category"]) == {
        "Food",
        "Electronics",
    }

    assert not model_data[
        list(MODEL_FEATURE_COLUMNS)
    ].isna().any().any()

    assert "revenue" not in model_data.columns
    assert (
        "cost_of_goods_sold"
        not in model_data.columns
    )


def test_unknown_store_is_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Unknown sales store IDs should be rejected."""

    sales = sample_sales.copy()

    sales.loc[
        sales["product_id"] == "P001",
        "store_id",
    ] = "S999"

    with pytest.raises(
        ValueError,
        match="unknown store IDs",
    ):
        prepare_model_data(
            sales=sales,
            stores=sample_stores,
            products=sample_products,
        )


def test_unknown_product_is_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Unknown sales product IDs should be rejected."""

    sales = sample_sales.copy()

    sales.loc[
        sales["product_id"] == "P001",
        "product_id",
    ] = "P999"

    with pytest.raises(
        ValueError,
        match="unknown product IDs",
    ):
        prepare_model_data(
            sales=sales,
            stores=sample_stores,
            products=sample_products,
        )


def test_invalid_numeric_product_feature_is_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Invalid product numbers should be rejected."""

    products = sample_products.copy()

    products["price"] = (
        products["price"].astype(object)
    )

    products.loc[0, "price"] = "invalid"

    with pytest.raises(
        ValueError,
        match="missing feature values",
    ):
        prepare_model_data(
            sales=sample_sales,
            stores=sample_stores,
            products=products,
        )


def test_validate_split_ratios_accepts_valid_values() -> None:
    """Valid split ratios should be accepted."""

    validate_split_ratios(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )


@pytest.mark.parametrize(
    (
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    ),
    [
        (0.0, 0.50, 0.50),
        (0.70, 0.20, 0.20),
        ("0.70", 0.15, 0.15),
        (True, 0.50, 0.50),
    ],
)
def test_invalid_split_ratios_are_rejected(
    train_ratio,
    validation_ratio,
    test_ratio,
) -> None:
    """Invalid split ratios should be rejected."""

    with pytest.raises(ValueError):
        validate_split_ratios(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )


def test_split_model_data_chronologically(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Datasets should be separated chronologically."""

    model_data = prepare_model_data(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    )

    dataset = split_model_data_by_date(
        model_data=model_data,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    assert (
        dataset.train["date"].nunique()
        == 3
    )
    assert (
        dataset.validation["date"].nunique()
        == 1
    )
    assert (
        dataset.test["date"].nunique()
        == 3
    )

    assert len(dataset.train) == 6
    assert len(dataset.validation) == 2
    assert len(dataset.test) == 6

    assert (
        dataset.train["date"].max()
        < dataset.validation["date"].min()
    )

    assert (
        dataset.validation["date"].max()
        < dataset.test["date"].min()
    )

    train_dates = set(
        dataset.train["date"]
    )
    validation_dates = set(
        dataset.validation["date"]
    )
    test_dates = set(
        dataset.test["date"]
    )

    assert train_dates.isdisjoint(
        validation_dates
    )
    assert train_dates.isdisjoint(
        test_dates
    )
    assert validation_dates.isdisjoint(
        test_dates
    )


def test_split_model_data_must_be_dataframe() -> None:
    """Model data should be a DataFrame."""

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        split_model_data_by_date(
            model_data=[],
        )


def test_empty_model_data_is_rejected() -> None:
    """Empty model data should be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        split_model_data_by_date(
            model_data=pd.DataFrame(),
        )


def test_missing_model_column_is_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Missing model columns should be rejected."""

    model_data = prepare_model_data(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    ).drop(
        columns=["lag_1"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        split_model_data_by_date(
            model_data=model_data,
        )


def test_invalid_model_date_is_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Invalid model dates should be rejected."""

    model_data = prepare_model_data(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    )

    model_data["date"] = (
        model_data["date"].astype(object)
    )

    model_data.loc[1, "date"] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid dates",
    ):
        split_model_data_by_date(
            model_data=model_data,
        )


def test_insufficient_dates_are_rejected(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """Every split should contain at least one date."""

    model_data = prepare_model_data(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    )

    first_two_dates = (
        model_data["date"]
        .drop_duplicates()
        .head(2)
    )

    small_model_data = model_data.loc[
        model_data["date"].isin(
            first_two_dates
        )
    ]

    with pytest.raises(
        ValueError,
        match="Not enough dates",
    ):
        split_model_data_by_date(
            model_data=small_model_data,
            train_ratio=0.50,
            validation_ratio=0.25,
            test_ratio=0.25,
        )


def test_build_training_dataset(
    sample_sales: pd.DataFrame,
    sample_stores: pd.DataFrame,
    sample_products: pd.DataFrame,
) -> None:
    """The complete dataset builder should return all splits."""

    dataset = build_training_dataset(
        sales=sample_sales,
        stores=sample_stores,
        products=sample_products,
    )

    assert not dataset.train.empty
    assert not dataset.validation.empty
    assert not dataset.test.empty

    assert dataset.feature_columns == (
        MODEL_FEATURE_COLUMNS
    )

    assert (
        dataset.categorical_feature_columns
        == CATEGORICAL_FEATURE_COLUMNS
    )

    assert (
        dataset.numerical_feature_columns
        == NUMERICAL_FEATURE_COLUMNS
    )

    assert dataset.target_column == (
        TARGET_COLUMN
    )