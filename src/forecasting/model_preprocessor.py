"""Preprocess forecasting data for machine learning models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import load_all_data
from src.forecasting.training_dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TrainingDataset,
    build_training_dataset,
)


@dataclass(frozen=True)
class PreprocessedDataset:
    """Store transformed model matrices and targets."""

    preprocessor: ColumnTransformer
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def validate_feature_groups(
    categorical_columns: tuple[str, ...],
    numerical_columns: tuple[str, ...],
) -> None:
    """Validate categorical and numerical feature groups."""

    if not categorical_columns:
        raise ValueError(
            "categorical_columns must not be empty."
        )

    if not numerical_columns:
        raise ValueError(
            "numerical_columns must not be empty."
        )

    all_columns = (
        *categorical_columns,
        *numerical_columns,
    )

    if len(set(all_columns)) != len(all_columns):
        raise ValueError(
            "Feature columns must not contain "
            "duplicates."
        )


def build_model_preprocessor(
    categorical_columns: tuple[str, ...] = (
        CATEGORICAL_FEATURE_COLUMNS
    ),
    numerical_columns: tuple[str, ...] = (
        NUMERICAL_FEATURE_COLUMNS
    ),
) -> ColumnTransformer:
    """Create the preprocessing transformer."""

    validate_feature_groups(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                list(categorical_columns),
            ),
            (
                "numerical",
                "passthrough",
                list(numerical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_and_target(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate model features from the prediction target."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data must not be empty."
        )

    required_columns = {
        *feature_columns,
        target_column,
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "data is missing model columns: "
            f"{sorted(missing_columns)}"
        )

    features = data[
        list(feature_columns)
    ].copy()

    if features.isna().any().any():
        raise ValueError(
            "Model features contain missing values."
        )

    target = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "Model target contains invalid values."
        )

    if (target < 0).any():
        raise ValueError(
            "Model target must not contain "
            "negative values."
        )

    return (
        features,
        target.to_numpy(dtype=float),
    )


def validate_transformed_matrix(
    matrix: np.ndarray,
    matrix_name: str,
) -> None:
    """Validate a transformed numerical matrix."""

    if matrix.ndim != 2:
        raise ValueError(
            f"{matrix_name} must be two-dimensional."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            f"{matrix_name} must not be empty."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{matrix_name} contains non-finite values."
        )


def preprocess_training_dataset(
    dataset: TrainingDataset,
) -> PreprocessedDataset:
    """Fit preprocessing on train and transform all splits."""

    if not isinstance(dataset, TrainingDataset):
        raise TypeError(
            "dataset must be a TrainingDataset."
        )

    X_train_frame, y_train = (
        split_features_and_target(
            data=dataset.train,
            feature_columns=dataset.feature_columns,
            target_column=dataset.target_column,
        )
    )

    X_validation_frame, y_validation = (
        split_features_and_target(
            data=dataset.validation,
            feature_columns=dataset.feature_columns,
            target_column=dataset.target_column,
        )
    )

    X_test_frame, y_test = (
        split_features_and_target(
            data=dataset.test,
            feature_columns=dataset.feature_columns,
            target_column=dataset.target_column,
        )
    )

    preprocessor = build_model_preprocessor(
        categorical_columns=(
            dataset.categorical_feature_columns
        ),
        numerical_columns=(
            dataset.numerical_feature_columns
        ),
    )

    X_train = np.asarray(
        preprocessor.fit_transform(
            X_train_frame
        ),
        dtype=float,
    )

    X_validation = np.asarray(
        preprocessor.transform(
            X_validation_frame
        ),
        dtype=float,
    )

    X_test = np.asarray(
        preprocessor.transform(
            X_test_frame
        ),
        dtype=float,
    )

    validate_transformed_matrix(
        matrix=X_train,
        matrix_name="X_train",
    )

    validate_transformed_matrix(
        matrix=X_validation,
        matrix_name="X_validation",
    )

    validate_transformed_matrix(
        matrix=X_test,
        matrix_name="X_test",
    )

    feature_names = tuple(
        preprocessor.get_feature_names_out()
    )

    return PreprocessedDataset(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )


def main() -> None:
    """Preprocess the project forecasting dataset."""

    project_data = load_all_data()

    training_dataset = build_training_dataset(
        sales=project_data.sales,
        stores=project_data.stores,
        products=project_data.products,
    )

    processed_data = (
        preprocess_training_dataset(
            dataset=training_dataset,
        )
    )

    print(
        "Model preprocessing completed "
        "successfully."
    )

    print(
        f"X_train shape: "
        f"{processed_data.X_train.shape}"
    )
    print(
        f"y_train shape: "
        f"{processed_data.y_train.shape}"
    )

    print(
        f"X_validation shape: "
        f"{processed_data.X_validation.shape}"
    )
    print(
        f"y_validation shape: "
        f"{processed_data.y_validation.shape}"
    )

    print(
        f"X_test shape: "
        f"{processed_data.X_test.shape}"
    )
    print(
        f"y_test shape: "
        f"{processed_data.y_test.shape}"
    )

    print(
        "Transformed feature count: "
        f"{len(processed_data.feature_names)}"
    )

    print()
    print("First transformed features:")

    for feature_name in (
        processed_data.feature_names[:20]
    ):
        print(f"- {feature_name}")


if __name__ == "__main__":
    main()