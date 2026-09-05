"""Tests for the AdaBoost forecasting model."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from src.forecasting.adaboost import (
    AdaBoostResult,
    build_adaboost_model,
    create_adaboost_feature_importance,
    predict_adaboost_demand,
    train_adaboost,
    validate_adaboost_settings,
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


def test_validate_adaboost_settings() -> None:
    """Valid AdaBoost settings should pass."""

    validate_adaboost_settings(
        n_estimators=10,
        learning_rate=0.1,
        base_max_depth=3,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
        loss="linear",
    )


@pytest.mark.parametrize(
    (
        "n_estimators",
        "learning_rate",
        "base_max_depth",
        "base_min_samples_split",
        "base_min_samples_leaf",
        "loss",
    ),
    [
        (0, 0.1, 3, 2, 1, "linear"),
        (True, 0.1, 3, 2, 1, "linear"),
        (10, 0.0, 3, 2, 1, "linear"),
        (10, True, 3, 2, 1, "linear"),
        (10, 0.1, 0, 2, 1, "linear"),
        (10, 0.1, 3, 1, 1, "linear"),
        (10, 0.1, 3, 2, 0, "linear"),
        (10, 0.1, 3, 2, 1, "invalid"),
    ],
)
def test_invalid_adaboost_settings(
    n_estimators,
    learning_rate,
    base_max_depth,
    base_min_samples_split,
    base_min_samples_leaf,
    loss,
) -> None:
    """Invalid AdaBoost settings should fail."""

    with pytest.raises(ValueError):
        validate_adaboost_settings(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            base_max_depth=base_max_depth,
            base_min_samples_split=(
                base_min_samples_split
            ),
            base_min_samples_leaf=(
                base_min_samples_leaf
            ),
            loss=loss,
        )


def test_build_adaboost_model() -> None:
    """The model builder should apply settings."""

    model = build_adaboost_model(
        n_estimators=10,
        learning_rate=0.1,
        base_max_depth=3,
        base_min_samples_split=4,
        base_min_samples_leaf=2,
        loss="square",
        random_state=2026,
    )

    assert isinstance(
        model,
        AdaBoostRegressor,
    )

    assert isinstance(
        model.estimator,
        DecisionTreeRegressor,
    )

    assert model.n_estimators == 10
    assert model.learning_rate == 0.1
    assert model.loss == "square"
    assert model.random_state == 2026

    assert model.estimator.max_depth == 3
    assert model.estimator.min_samples_split == 4
    assert model.estimator.min_samples_leaf == 2


def test_invalid_random_state_is_rejected() -> None:
    """Random state should be an integer."""

    with pytest.raises(
        ValueError,
        match="random_state",
    ):
        build_adaboost_model(
            random_state=True,
        )


def test_train_adaboost(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """AdaBoost should train and predict."""

    result = train_adaboost(
        processed_data=sample_processed_data,
        n_estimators=5,
        learning_rate=0.1,
        base_max_depth=3,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
        loss="linear",
        random_state=2026,
    )

    assert isinstance(
        result,
        AdaBoostResult,
    )

    assert isinstance(
        result.model,
        AdaBoostRegressor,
    )

    assert 1 <= len(
        result.model.estimators_
    ) <= 5

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
        train_adaboost(
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
        train_adaboost(
            processed_data=invalid_data,
            n_estimators=5,
        )


def test_predict_adaboost_demand(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """A fitted AdaBoost model should predict."""

    result = train_adaboost(
        processed_data=sample_processed_data,
        n_estimators=5,
        learning_rate=0.1,
        base_max_depth=3,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
    )

    predictions = predict_adaboost_demand(
        model=result.model,
        features=(
            sample_processed_data.X_validation
        ),
    )

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_prediction_clips_negative_values() -> None:
    """Negative predictions should become zero."""

    features = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    target = np.array(
        [
            -3.0,
            -2.0,
            -1.0,
        ]
    )

    model = build_adaboost_model(
        n_estimators=5,
        learning_rate=0.1,
        base_max_depth=2,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
    )

    model.fit(
        features,
        target,
    )

    predictions = predict_adaboost_demand(
        model=model,
        features=features,
    )

    assert (predictions == 0).all()


def test_unfitted_model_cannot_predict() -> None:
    """An unfitted model should not predict."""

    model = build_adaboost_model(
        n_estimators=5,
    )

    with pytest.raises(NotFittedError):
        predict_adaboost_demand(
            model=model,
            features=np.array([[1.0]]),
        )


@pytest.mark.parametrize(
    (
        "features",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            [[1.0]],
            TypeError,
            "NumPy array",
        ),
        (
            np.array([1.0]),
            ValueError,
            "two-dimensional",
        ),
        (
            np.empty((0, 1)),
            ValueError,
            "must not be empty",
        ),
        (
            np.array([[np.nan]]),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_invalid_prediction_features(
    features,
    expected_exception,
    expected_message: str,
) -> None:
    """Invalid prediction features should fail."""

    model = build_adaboost_model(
        n_estimators=5,
        base_max_depth=2,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
    )

    model.fit(
        np.array(
            [
                [0.0],
                [1.0],
            ]
        ),
        np.array(
            [
                1.0,
                2.0,
            ]
        ),
    )

    with pytest.raises(
        expected_exception,
        match=expected_message,
    ):
        predict_adaboost_demand(
            model=model,
            features=features,
        )


def test_create_adaboost_feature_importance(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """Feature importance should be sorted."""

    result = train_adaboost(
        processed_data=sample_processed_data,
        n_estimators=5,
        learning_rate=0.1,
        base_max_depth=3,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
    )

    importance = result.feature_importance

    assert isinstance(
        importance,
        pd.DataFrame,
    )

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

    model = build_adaboost_model(
        n_estimators=5,
        base_max_depth=2,
        base_min_samples_split=2,
        base_min_samples_leaf=1,
    )

    model.fit(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
            ]
        ),
        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="count must match",
    ):
        create_adaboost_feature_importance(
            model=model,
            feature_names=("lag_1",),
        )


def test_feature_importance_requires_adaboost() -> None:
    """Feature importance should require AdaBoost."""

    with pytest.raises(
        TypeError,
        match="AdaBoostRegressor",
    ):
        create_adaboost_feature_importance(
            model="invalid",
            feature_names=("lag_1",),
        )