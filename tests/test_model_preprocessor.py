"""Tests for forecasting model preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.forecasting.model_preprocessor import (
    build_model_preprocessor,
    preprocess_training_dataset,
    split_features_and_target,
    validate_feature_groups,
    validate_transformed_matrix,
)
from src.forecasting.training_dataset import (
    TrainingDataset,
)


@pytest.fixture
def sample_training_dataset() -> TrainingDataset:
    """Create a small chronological training dataset."""

    train = pd.DataFrame(
        {
            "city": [
                "Hanoi",
                "Da Nang",
                "Hanoi",
                "Da Nang",
            ],
            "lag_1": [
                10.0,
                20.0,
                12.0,
                18.0,
            ],
            "quantity_sold": [
                11.0,
                19.0,
                13.0,
                17.0,
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "city": [
                "Hue",
                "Hanoi",
            ],
            "lag_1": [
                15.0,
                14.0,
            ],
            "quantity_sold": [
                16.0,
                15.0,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "city": [
                "Da Nang",
                "Hanoi",
            ],
            "lag_1": [
                21.0,
                16.0,
            ],
            "quantity_sold": [
                20.0,
                17.0,
            ],
        }
    )

    return TrainingDataset(
        train=train,
        validation=validation,
        test=test,
        feature_columns=(
            "city",
            "lag_1",
        ),
        categorical_feature_columns=(
            "city",
        ),
        numerical_feature_columns=(
            "lag_1",
        ),
        target_column="quantity_sold",
    )


def test_validate_feature_groups_accepts_valid_values() -> None:
    """Valid feature groups should be accepted."""

    validate_feature_groups(
        categorical_columns=("city",),
        numerical_columns=("lag_1",),
    )


@pytest.mark.parametrize(
    (
        "categorical_columns",
        "numerical_columns",
    ),
    [
        ((), ("lag_1",)),
        (("city",), ()),
        (("city",), ("city",)),
        (("city", "city"), ("lag_1",)),
    ],
)
def test_invalid_feature_groups_are_rejected(
    categorical_columns: tuple[str, ...],
    numerical_columns: tuple[str, ...],
) -> None:
    """Empty or duplicate feature groups should fail."""

    with pytest.raises(ValueError):
        validate_feature_groups(
            categorical_columns=(
                categorical_columns
            ),
            numerical_columns=(
                numerical_columns
            ),
        )


def test_build_model_preprocessor() -> None:
    """The builder should return a ColumnTransformer."""

    preprocessor = build_model_preprocessor(
        categorical_columns=("city",),
        numerical_columns=("lag_1",),
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )

    assert preprocessor.remainder == "drop"


def test_split_features_and_target() -> None:
    """Features and target should be separated."""

    data = pd.DataFrame(
        {
            "city": ["Hanoi", "Da Nang"],
            "lag_1": [10.0, 20.0],
            "quantity_sold": [11.0, 19.0],
        }
    )

    features, target = (
        split_features_and_target(
            data=data,
            feature_columns=(
                "city",
                "lag_1",
            ),
            target_column="quantity_sold",
        )
    )

    assert list(features.columns) == [
        "city",
        "lag_1",
    ]

    assert np.array_equal(
        target,
        np.array([11.0, 19.0]),
    )


def test_split_data_must_be_dataframe() -> None:
    """Input data should be a DataFrame."""

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        split_features_and_target(
            data=[],
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_split_data_must_not_be_empty() -> None:
    """Empty model data should be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        split_features_and_target(
            data=pd.DataFrame(),
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_missing_model_column_is_rejected() -> None:
    """Missing model columns should be rejected."""

    data = pd.DataFrame(
        {
            "lag_1": [10.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing model columns",
    ):
        split_features_and_target(
            data=data,
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_missing_feature_value_is_rejected() -> None:
    """Missing feature values should be rejected."""

    data = pd.DataFrame(
        {
            "lag_1": [np.nan],
            "quantity_sold": [10.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="features contain missing",
    ):
        split_features_and_target(
            data=data,
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_invalid_target_value_is_rejected() -> None:
    """Nonnumeric target values should be rejected."""

    data = pd.DataFrame(
        {
            "lag_1": [10.0],
            "quantity_sold": ["invalid"],
        }
    )

    with pytest.raises(
        ValueError,
        match="target contains invalid",
    ):
        split_features_and_target(
            data=data,
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_negative_target_is_rejected() -> None:
    """Negative demand values should be rejected."""

    data = pd.DataFrame(
        {
            "lag_1": [10.0],
            "quantity_sold": [-1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="negative values",
    ):
        split_features_and_target(
            data=data,
            feature_columns=("lag_1",),
            target_column="quantity_sold",
        )


def test_validate_transformed_matrix_accepts_valid_matrix() -> None:
    """A finite two-dimensional matrix should pass."""

    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    validate_transformed_matrix(
        matrix=matrix,
        matrix_name="X_train",
    )


@pytest.mark.parametrize(
    (
        "matrix",
        "expected_message",
    ),
    [
        (
            np.array([1.0, 2.0]),
            "two-dimensional",
        ),
        (
            np.empty((0, 2)),
            "must not be empty",
        ),
        (
            np.array([[1.0, np.nan]]),
            "non-finite",
        ),
        (
            np.array([[1.0, np.inf]]),
            "non-finite",
        ),
    ],
)
def test_invalid_transformed_matrix_is_rejected(
    matrix: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid numerical matrices should be rejected."""

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        validate_transformed_matrix(
            matrix=matrix,
            matrix_name="X_train",
        )


def test_preprocess_training_dataset_shapes(
    sample_training_dataset: TrainingDataset,
) -> None:
    """All transformed matrices should have valid shapes."""

    processed = preprocess_training_dataset(
        dataset=sample_training_dataset,
    )

    assert processed.X_train.shape == (
        4,
        3,
    )

    assert processed.X_validation.shape == (
        2,
        3,
    )

    assert processed.X_test.shape == (
        2,
        3,
    )

    assert processed.y_train.shape == (4,)
    assert processed.y_validation.shape == (2,)
    assert processed.y_test.shape == (2,)

    assert np.array_equal(
        processed.y_train,
        np.array(
            [
                11.0,
                19.0,
                13.0,
                17.0,
            ]
        ),
    )


def test_preprocessor_handles_unknown_category(
    sample_training_dataset: TrainingDataset,
) -> None:
    """Validation categories absent from train should work."""

    processed = preprocess_training_dataset(
        dataset=sample_training_dataset,
    )

    feature_names = list(
        processed.feature_names
    )

    city_feature_indexes = [
        index
        for index, feature_name in enumerate(
            feature_names
        )
        if feature_name.startswith("city_")
    ]

    unknown_city_values = (
        processed.X_validation[
            0,
            city_feature_indexes,
        ]
    )

    assert np.array_equal(
        unknown_city_values,
        np.zeros(
            len(city_feature_indexes)
        ),
    )


def test_preprocessed_feature_names(
    sample_training_dataset: TrainingDataset,
) -> None:
    """Transformed feature names should be available."""

    processed = preprocess_training_dataset(
        dataset=sample_training_dataset,
    )

    assert processed.feature_names == (
        "city_Da Nang",
        "city_Hanoi",
        "lag_1",
    )

    assert len(processed.feature_names) == (
        processed.X_train.shape[1]
    )


def test_preprocess_requires_training_dataset() -> None:
    """The preprocessing function should reject other types."""

    with pytest.raises(
        TypeError,
        match="TrainingDataset",
    ):
        preprocess_training_dataset(
            dataset={},
        )