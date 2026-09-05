"""Store and retrieve trained forecasting model artifacts."""

from dataclasses import dataclass
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)


RANDOM_FOREST_METHOD = "random_forest"
ADABOOST_METHOD = "adaboost"

SUPPORTED_MACHINE_LEARNING_METHODS = (
    RANDOM_FOREST_METHOD,
    ADABOOST_METHOD,
)

MODEL_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "models"
)

ModelType = (
    RandomForestRegressor
    | AdaBoostRegressor
)


@dataclass(frozen=True)
class ForecastModelArtifact:
    """Store a trained model and its supporting metadata."""

    method: str
    dataset_fingerprint: str
    model: ModelType
    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]
    feature_importance: pd.DataFrame
    validation_mae: float
    validation_rmse: float
    test_mae: float
    test_rmse: float
    trained_at: str


def validate_model_method(
    method: str,
) -> str:
    """Validate and normalize a machine-learning method."""

    if not isinstance(method, str):
        raise TypeError(
            "method must be a string."
        )

    normalized_method = (
        method.strip().lower()
    )

    if (
        normalized_method
        not in SUPPORTED_MACHINE_LEARNING_METHODS
    ):
        raise ValueError(
            f"Unsupported machine-learning method: "
            f"{method}"
        )

    return normalized_method


def validate_dataset_fingerprint(
    dataset_fingerprint: str,
) -> str:
    """Validate a safe dataset fingerprint."""

    if not isinstance(
        dataset_fingerprint,
        str,
    ):
        raise TypeError(
            "dataset_fingerprint must be a string."
        )

    normalized_fingerprint = (
        dataset_fingerprint.strip()
    )

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}",
        normalized_fingerprint,
    ):
        raise ValueError(
            "dataset_fingerprint contains "
            "unsupported characters."
        )

    return normalized_fingerprint


def validate_metric(
    value: float,
    metric_name: str,
) -> None:
    """Validate a nonnegative model metric."""

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise TypeError(
            f"{metric_name} must be numeric."
        )

    if (
        not np.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(
            f"{metric_name} must be finite "
            "and nonnegative."
        )


def validate_model_artifact(
    artifact: ForecastModelArtifact,
) -> None:
    """Validate an artifact before saving or using it."""

    if not isinstance(
        artifact,
        ForecastModelArtifact,
    ):
        raise TypeError(
            "artifact must be a "
            "ForecastModelArtifact."
        )

    validate_model_method(
        artifact.method
    )

    validate_dataset_fingerprint(
        artifact.dataset_fingerprint
    )

    if not isinstance(
        artifact.model,
        (
            RandomForestRegressor,
            AdaBoostRegressor,
        ),
    ):
        raise TypeError(
            "artifact.model has an "
            "unsupported model type."
        )

    if not isinstance(
        artifact.preprocessor,
        ColumnTransformer,
    ):
        raise TypeError(
            "artifact.preprocessor must be "
            "a ColumnTransformer."
        )

    if not artifact.feature_names:
        raise ValueError(
            "artifact.feature_names must "
            "not be empty."
        )

    required_importance_columns = {
        "feature",
        "importance",
    }

    missing_columns = (
        required_importance_columns
        - set(artifact.feature_importance.columns)
    )

    if missing_columns:
        raise ValueError(
            "feature_importance is missing columns: "
            f"{sorted(missing_columns)}"
        )

    for metric_name in (
        "validation_mae",
        "validation_rmse",
        "test_mae",
        "test_rmse",
    ):
        validate_metric(
            value=getattr(
                artifact,
                metric_name,
            ),
            metric_name=metric_name,
        )

    if (
        not isinstance(artifact.trained_at, str)
        or not artifact.trained_at.strip()
    ):
        raise ValueError(
            "trained_at must not be empty."
        )


def get_model_artifact_path(
    method: str,
    dataset_fingerprint: str,
) -> Path:
    """Return the artifact path for one dataset and method."""

    normalized_method = (
        validate_model_method(method)
    )

    normalized_fingerprint = (
        validate_dataset_fingerprint(
            dataset_fingerprint
        )
    )

    return (
        MODEL_ARTIFACT_ROOT
        / normalized_fingerprint
        / f"{normalized_method}.joblib"
    )


def model_artifact_exists(
    method: str,
    dataset_fingerprint: str,
) -> bool:
    """Return whether a compatible artifact exists."""

    artifact_path = get_model_artifact_path(
        method=method,
        dataset_fingerprint=(
            dataset_fingerprint
        ),
    )

    return artifact_path.is_file()


def save_model_artifact(
    artifact: ForecastModelArtifact,
) -> Path:
    """Save a validated model artifact."""

    validate_model_artifact(artifact)

    artifact_path = get_model_artifact_path(
        method=artifact.method,
        dataset_fingerprint=(
            artifact.dataset_fingerprint
        ),
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        artifact_path,
    )

    return artifact_path


def load_model_artifact(
    method: str,
    dataset_fingerprint: str,
) -> ForecastModelArtifact:
    """Load and validate one compatible model artifact."""

    artifact_path = get_model_artifact_path(
        method=method,
        dataset_fingerprint=(
            dataset_fingerprint
        ),
    )

    if not artifact_path.is_file():
        raise FileNotFoundError(
            "No compatible model artifact was found: "
            f"{artifact_path}"
        )

    artifact = joblib.load(
        artifact_path
    )

    validate_model_artifact(artifact)

    if (
        artifact.dataset_fingerprint
        != dataset_fingerprint
    ):
        raise ValueError(
            "The model artifact does not match "
            "the active dataset."
        )

    return artifact