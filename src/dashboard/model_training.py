"""Train forecasting models for the active dashboard dataset."""

from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

from src.dashboard.model_artifacts import (
    ADABOOST_METHOD,
    RANDOM_FOREST_METHOD,
    SUPPORTED_MACHINE_LEARNING_METHODS,
    ForecastModelArtifact,
    save_model_artifact,
    validate_dataset_fingerprint,
    validate_model_method,
)
from src.data_loader import ProjectData
from src.forecasting.adaboost import (
    train_adaboost,
)
from src.forecasting.model_preprocessor import (
    PreprocessedDataset,
    preprocess_training_dataset,
)
from src.forecasting.random_forest import (
    train_random_forest,
)
from src.forecasting.training_dataset import (
    build_training_dataset,
)


def validate_prediction_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Validate actual and predicted demand arrays."""

    if not isinstance(actual, np.ndarray):
        raise TypeError(
            "actual must be a NumPy array."
        )

    if not isinstance(predicted, np.ndarray):
        raise TypeError(
            "predicted must be a NumPy array."
        )

    if actual.ndim != 1:
        raise ValueError(
            "actual must be one-dimensional."
        )

    if predicted.ndim != 1:
        raise ValueError(
            "predicted must be one-dimensional."
        )

    if actual.size == 0:
        raise ValueError(
            "actual must not be empty."
        )

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual and predicted must have "
            "the same shape."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "actual contains non-finite values."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "predicted contains non-finite values."
        )


def calculate_model_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """Calculate MAE and RMSE for model predictions."""

    validate_prediction_arrays(
        actual=actual,
        predicted=predicted,
    )

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        root_mean_squared_error(
            actual,
            predicted,
        )
    )

    return (
        round(mae, 4),
        round(rmse, 4),
    )


def prepare_model_training_data(
    project_data: ProjectData,
) -> PreprocessedDataset:
    """Create and preprocess model training data."""

    if not isinstance(
        project_data,
        ProjectData,
    ):
        raise TypeError(
            "project_data must be ProjectData."
        )

    training_dataset = build_training_dataset(
        sales=project_data.sales,
        stores=project_data.stores,
        products=project_data.products,
    )

    return preprocess_training_dataset(
        dataset=training_dataset
    )


def train_random_forest_artifact(
    processed_data: PreprocessedDataset,
    dataset_fingerprint: str,
) -> ForecastModelArtifact:
    """Train Random Forest and create its artifact."""

    normalized_fingerprint = (
        validate_dataset_fingerprint(
            dataset_fingerprint
        )
    )

    training_result = train_random_forest(
        processed_data=processed_data
    )

    validation_mae, validation_rmse = (
        calculate_model_metrics(
            actual=processed_data.y_validation,
            predicted=(
                training_result
                .validation_predictions
            ),
        )
    )

    test_mae, test_rmse = (
        calculate_model_metrics(
            actual=processed_data.y_test,
            predicted=(
                training_result.test_predictions
            ),
        )
    )

    return ForecastModelArtifact(
        method=RANDOM_FOREST_METHOD,
        dataset_fingerprint=(
            normalized_fingerprint
        ),
        model=training_result.model,
        preprocessor=(
            processed_data.preprocessor
        ),
        feature_names=(
            processed_data.feature_names
        ),
        feature_importance=(
            training_result.feature_importance
        ),
        validation_mae=validation_mae,
        validation_rmse=validation_rmse,
        test_mae=test_mae,
        test_rmse=test_rmse,
        trained_at=(
            datetime.now(timezone.utc)
            .isoformat()
        ),
    )


def train_adaboost_artifact(
    processed_data: PreprocessedDataset,
    dataset_fingerprint: str,
) -> ForecastModelArtifact:
    """Train AdaBoost and create its artifact."""

    normalized_fingerprint = (
        validate_dataset_fingerprint(
            dataset_fingerprint
        )
    )

    training_result = train_adaboost(
        processed_data=processed_data
    )

    validation_mae, validation_rmse = (
        calculate_model_metrics(
            actual=processed_data.y_validation,
            predicted=(
                training_result
                .validation_predictions
            ),
        )
    )

    test_mae, test_rmse = (
        calculate_model_metrics(
            actual=processed_data.y_test,
            predicted=(
                training_result.test_predictions
            ),
        )
    )

    return ForecastModelArtifact(
        method=ADABOOST_METHOD,
        dataset_fingerprint=(
            normalized_fingerprint
        ),
        model=training_result.model,
        preprocessor=(
            processed_data.preprocessor
        ),
        feature_names=(
            processed_data.feature_names
        ),
        feature_importance=(
            training_result.feature_importance
        ),
        validation_mae=validation_mae,
        validation_rmse=validation_rmse,
        test_mae=test_mae,
        test_rmse=test_rmse,
        trained_at=(
            datetime.now(timezone.utc)
            .isoformat()
        ),
    )


def train_model_artifact(
    processed_data: PreprocessedDataset,
    method: str,
    dataset_fingerprint: str,
) -> ForecastModelArtifact:
    """Train one selected forecasting model."""

    normalized_method = (
        validate_model_method(method)
    )

    if normalized_method == RANDOM_FOREST_METHOD:
        return train_random_forest_artifact(
            processed_data=processed_data,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
        )

    if normalized_method == ADABOOST_METHOD:
        return train_adaboost_artifact(
            processed_data=processed_data,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
        )

    raise RuntimeError(
        "The selected model has no "
        "training implementation."
    )


def validate_training_methods(
    methods: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and normalize requested model methods."""

    if not isinstance(methods, tuple):
        raise TypeError(
            "methods must be a tuple."
        )

    if not methods:
        raise ValueError(
            "methods must not be empty."
        )

    normalized_methods = tuple(
        validate_model_method(method)
        for method in methods
    )

    if (
        len(set(normalized_methods))
        != len(normalized_methods)
    ):
        raise ValueError(
            "methods must not contain duplicates."
        )

    return normalized_methods


def train_forecast_models(
    project_data: ProjectData,
    dataset_fingerprint: str,
    methods: tuple[str, ...] = (
        SUPPORTED_MACHINE_LEARNING_METHODS
    ),
    save_artifacts: bool = True,
) -> dict[str, ForecastModelArtifact]:
    """Train selected models and optionally save artifacts."""

    if not isinstance(save_artifacts, bool):
        raise TypeError(
            "save_artifacts must be a boolean."
        )

    normalized_fingerprint = (
        validate_dataset_fingerprint(
            dataset_fingerprint
        )
    )

    normalized_methods = (
        validate_training_methods(methods)
    )

    processed_data = (
        prepare_model_training_data(
            project_data=project_data
        )
    )

    artifacts: dict[
        str,
        ForecastModelArtifact,
    ] = {}

    for method in normalized_methods:
        artifact = train_model_artifact(
            processed_data=processed_data,
            method=method,
            dataset_fingerprint=(
                normalized_fingerprint
            ),
        )

        if save_artifacts:
            save_model_artifact(
                artifact
            )

        artifacts[method] = artifact

    return artifacts