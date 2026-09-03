"""Tests for the Decision Tree forecasting model."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from src.forecasting.decision_tree import (
    DecisionTreeResult,
    build_decision_tree_model,
    create_feature_importance,
    predict_nonnegative_demand,
    train_decision_tree,
    validate_decision_tree_settings,
    validate_supervised_arrays,
)
from src.forecasting.model_preprocessor import (
    PreprocessedDataset,
)


@pytest.fixture
def sample_processed_data() -> PreprocessedDataset:
    """Create small preprocessed forecasting data."""

    X_train = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 2.0],
            [6.0, 2.0],
            [7.0, 2.0],
        ]
    )

    y_train = np.array(
        [
            1.0,
            3.0,
            5.0,
            7.0,
            10.0,
            12.0,
            14.0,
            16.0,
        ]
    )

    X_validation = np.array(
        [
            [8.0, 2.0],
            [9.0, 2.0],
        ]
    )

    y_validation = np.array(
        [
            18.0,
            20.0,
        ]
    )

    X_test = np.array(
        [
            [10.0, 3.0],
            [11.0, 3.0],
        ]
    )

    y_test = np.array(
        [
            23.0,
            25.0,
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[],
        remainder="passthrough",
    )

    return PreprocessedDataset(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_names=(
            "lag_1",
            "is_weekend",
        ),
    )


def test_validate_decision_tree_settings() -> None:
    """Valid Decision Tree settings should pass."""

    validate_decision_tree_settings(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )


@pytest.mark.parametrize(
    (
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    ),
    [
        (0, 2, 1),
        (True, 2, 1),
        (5, 1, 1),
        (5, 2, 0),
        (5.5, 2, 1),
    ],
)
def test_invalid_decision_tree_settings(
    max_depth,
    min_samples_split,
    min_samples_leaf,
) -> None:
    """Invalid Decision Tree settings should fail."""

    with pytest.raises(ValueError):
        validate_decision_tree_settings(
            max_depth=max_depth,
            min_samples_split=(
                min_samples_split
            ),
            min_samples_leaf=(
                min_samples_leaf
            ),
        )


def test_build_decision_tree_model() -> None:
    """The model builder should apply its settings."""

    model = build_decision_tree_model(
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=2026,
    )

    assert isinstance(
        model,
        DecisionTreeRegressor,
    )

    assert model.max_depth == 5
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 2
    assert model.random_state == 2026
    assert model.criterion == "squared_error"


def test_invalid_random_state_is_rejected() -> None:
    """Random state should be an integer."""

    with pytest.raises(
        ValueError,
        match="random_state",
    ):
        build_decision_tree_model(
            random_state=True,
        )


def test_validate_supervised_arrays() -> None:
    """Valid supervised arrays should pass."""

    validate_supervised_arrays(
        features=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        target=np.array(
            [
                5.0,
                6.0,
            ]
        ),
        dataset_name="Training",
    )


@pytest.mark.parametrize(
    (
        "features",
        "target",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            [[1.0, 2.0]],
            np.array([1.0]),
            TypeError,
            "NumPy array",
        ),
        (
            np.array([[1.0, 2.0]]),
            [1.0],
            TypeError,
            "NumPy array",
        ),
        (
            np.array([1.0, 2.0]),
            np.array([1.0]),
            ValueError,
            "two-dimensional",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([[1.0]]),
            ValueError,
            "one-dimensional",
        ),
        (
            np.empty((0, 2)),
            np.empty(0),
            ValueError,
            "must not be empty",
        ),
        (
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            np.array([1.0]),
            ValueError,
            "same number of rows",
        ),
        (
            np.array([[np.nan, 2.0]]),
            np.array([1.0]),
            ValueError,
            "features contain non-finite",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([np.inf]),
            ValueError,
            "target contains non-finite",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([-1.0]),
            ValueError,
            "negative values",
        ),
    ],
)
def test_invalid_supervised_arrays(
    features,
    target,
    expected_exception,
    expected_message: str,
) -> None:
    """Invalid supervised arrays should fail."""

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        validate_supervised_arrays(
            features=features,
            target=target,
            dataset_name="Training",
        )


def test_train_decision_tree(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """Decision Tree should train and predict."""

    result = train_decision_tree(
        processed_data=sample_processed_data,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2026,
    )

    assert isinstance(
        result,
        DecisionTreeResult,
    )

    assert isinstance(
        result.model,
        DecisionTreeRegressor,
    )

    assert result.model.get_depth() <= 3

    assert result.validation_predictions.shape == (
        2,
    )

    assert result.test_predictions.shape == (
        2,
    )

    assert (
        result.validation_predictions >= 0
    ).all()

    assert (
        result.test_predictions >= 0
    ).all()

    assert len(result.feature_importance) == 2


def test_train_requires_preprocessed_dataset() -> None:
    """Training should require PreprocessedDataset."""

    with pytest.raises(
        TypeError,
        match="PreprocessedDataset",
    ):
        train_decision_tree(
            processed_data={},
        )


def test_different_feature_counts_are_rejected(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """All splits should have the same features."""

    invalid_data = replace(
        sample_processed_data,
        X_validation=np.array(
            [
                [8.0, 2.0, 1.0],
                [9.0, 2.0, 1.0],
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="same feature count",
    ):
        train_decision_tree(
            processed_data=invalid_data,
        )


def test_prediction_clips_negative_values() -> None:
    """Negative model predictions should become zero."""

    features = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    target = np.array(
        [
            -2.0,
            -1.0,
        ]
    )

    model = DecisionTreeRegressor(
        random_state=2026,
    )

    model.fit(
        features,
        target,
    )

    predictions = predict_nonnegative_demand(
        model=model,
        features=features,
    )

    assert np.array_equal(
        predictions,
        np.array([0.0, 0.0]),
    )


def test_unfitted_model_cannot_predict() -> None:
    """An unfitted model should not predict."""

    model = DecisionTreeRegressor(
        random_state=2026,
    )

    with pytest.raises(NotFittedError):
        predict_nonnegative_demand(
            model=model,
            features=np.array([[1.0]]),
        )


def test_create_feature_importance(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """Feature importance should be sorted."""

    result = train_decision_tree(
        processed_data=sample_processed_data,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    importance = result.feature_importance

    assert list(importance.columns) == [
        "feature",
        "importance",
    ]

    assert importance[
        "importance"
    ].is_monotonic_decreasing

    assert importance[
        "importance"
    ].sum() == pytest.approx(1.0)

    assert set(importance["feature"]) == {
        "lag_1",
        "is_weekend",
    }


def test_feature_name_count_must_match_model() -> None:
    """Every trained feature should have a name."""

    features = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ]
    )

    target = np.array(
        [
            1.0,
            2.0,
        ]
    )

    model = DecisionTreeRegressor(
        random_state=2026,
    )

    model.fit(
        features,
        target,
    )

    with pytest.raises(
        ValueError,
        match="count must match",
    ):
        create_feature_importance(
            model=model,
            feature_names=("lag_1",),
        )