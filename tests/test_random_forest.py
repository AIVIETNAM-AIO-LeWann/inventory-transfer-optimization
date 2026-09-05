"""Tests for the Random Forest forecasting model."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

from src.forecasting.model_preprocessor import (
    PreprocessedDataset,
)
from src.forecasting.random_forest import (
    RandomForestResult,
    build_random_forest_model,
    create_random_forest_feature_importance,
    predict_random_forest_demand,
    train_random_forest,
    validate_random_forest_settings,
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


def test_validate_random_forest_settings() -> None:
    """Valid Random Forest settings should pass."""

    validate_random_forest_settings(
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=1,
    )


@pytest.mark.parametrize(
    (
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "n_jobs",
    ),
    [
        (0, 5, 2, 1, "sqrt", 1),
        (True, 5, 2, 1, "sqrt", 1),
        (10, 0, 2, 1, "sqrt", 1),
        (10, 5, 1, 1, "sqrt", 1),
        (10, 5, 2, 0, "sqrt", 1),
        (10, 5, 2, 1, "invalid", 1),
        (10, 5, 2, 1, "sqrt", 0),
        (10, 5, 2, 1, "sqrt", True),
    ],
)
def test_invalid_random_forest_settings(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    n_jobs,
) -> None:
    """Invalid Random Forest settings should fail."""

    with pytest.raises(ValueError):
        validate_random_forest_settings(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=(
                min_samples_split
            ),
            min_samples_leaf=(
                min_samples_leaf
            ),
            max_features=max_features,
            n_jobs=n_jobs,
        )


def test_build_random_forest_model() -> None:
    """The model builder should apply its settings."""

    model = build_random_forest_model(
        n_estimators=10,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="log2",
        n_jobs=1,
        random_state=2026,
    )

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators == 10
    assert model.max_depth == 5
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 2
    assert model.max_features == "log2"
    assert model.n_jobs == 1
    assert model.random_state == 2026
    assert model.bootstrap is True


def test_invalid_random_state_is_rejected() -> None:
    """Random state should be an integer."""

    with pytest.raises(
        ValueError,
        match="random_state",
    ):
        build_random_forest_model(
            random_state=True,
        )


def test_train_random_forest(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """Random Forest should train and predict."""

    result = train_random_forest(
        processed_data=sample_processed_data,
        n_estimators=5,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=1,
        random_state=2026,
    )

    assert isinstance(
        result,
        RandomForestResult,
    )

    assert isinstance(
        result.model,
        RandomForestRegressor,
    )

    assert len(result.model.estimators_) == 5

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
        train_random_forest(
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
        train_random_forest(
            processed_data=invalid_data,
            n_estimators=5,
            n_jobs=1,
        )


def test_predict_random_forest_demand(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """A fitted Random Forest should predict."""

    result = train_random_forest(
        processed_data=sample_processed_data,
        n_estimators=5,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
    )

    predictions = predict_random_forest_demand(
        model=result.model,
        features=(
            sample_processed_data.X_validation
        ),
    )

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_prediction_clips_negative_values() -> None:
    """Negative model predictions should become zero."""

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

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=2026,
        n_jobs=1,
    )

    model.fit(
        features,
        target,
    )

    predictions = predict_random_forest_demand(
        model=model,
        features=features,
    )

    assert (predictions == 0).all()


def test_unfitted_model_cannot_predict() -> None:
    """An unfitted model should not predict."""

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=2026,
        n_jobs=1,
    )

    with pytest.raises(NotFittedError):
        predict_random_forest_demand(
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

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=2026,
        n_jobs=1,
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
        predict_random_forest_demand(
            model=model,
            features=features,
        )


def test_create_random_forest_feature_importance(
    sample_processed_data: PreprocessedDataset,
) -> None:
    """Feature importance should be sorted."""

    result = train_random_forest(
        processed_data=sample_processed_data,
        n_estimators=5,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
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

    features = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )

    target = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=2026,
        n_jobs=1,
    )

    model.fit(
        features,
        target,
    )

    with pytest.raises(
        ValueError,
        match="count must match",
    ):
        create_random_forest_feature_importance(
            model=model,
            feature_names=("lag_1",),
        )


def test_feature_importance_requires_random_forest() -> None:
    """Feature importance should require the right model."""

    with pytest.raises(
        TypeError,
        match="RandomForestRegressor",
    ):
        create_random_forest_feature_importance(
            model="invalid",
            feature_names=("lag_1",),
        )