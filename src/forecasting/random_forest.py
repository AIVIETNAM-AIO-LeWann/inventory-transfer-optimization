"""Train a Random Forest model for demand forecasting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.utils.validation import check_is_fitted

from src.config import (
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MAX_FEATURES,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_MIN_SAMPLES_SPLIT,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_SEED,
)
from src.data_loader import load_all_data
from src.forecasting.decision_tree import (
    validate_supervised_arrays,
)
from src.forecasting.model_preprocessor import (
    PreprocessedDataset,
    preprocess_training_dataset,
)
from src.forecasting.training_dataset import (
    build_training_dataset,
)


@dataclass(frozen=True)
class RandomForestResult:
    """Store a trained model and its predictions."""

    model: RandomForestRegressor
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_importance: pd.DataFrame


def validate_random_forest_settings(
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    n_jobs: int,
) -> None:
    """Validate Random Forest hyperparameters."""

    positive_integer_settings = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

    for setting_name, setting_value in (
        positive_integer_settings.items()
    ):
        if (
            isinstance(setting_value, bool)
            or not isinstance(setting_value, int)
            or setting_value <= 0
        ):
            raise ValueError(
                f"{setting_name} must be a "
                "positive integer."
            )

    if min_samples_split < 2:
        raise ValueError(
            "min_samples_split must be at least 2."
        )

    if max_features not in (
        "sqrt",
        "log2",
    ):
        raise ValueError(
            "max_features must be "
            "'sqrt' or 'log2'."
        )

    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs == 0
    ):
        raise ValueError(
            "n_jobs must be -1 or a "
            "positive integer."
        )


def build_random_forest_model(
    n_estimators: int = (
        RANDOM_FOREST_N_ESTIMATORS
    ),
    max_depth: int = RANDOM_FOREST_MAX_DEPTH,
    min_samples_split: int = (
        RANDOM_FOREST_MIN_SAMPLES_SPLIT
    ),
    min_samples_leaf: int = (
        RANDOM_FOREST_MIN_SAMPLES_LEAF
    ),
    max_features: str = (
        RANDOM_FOREST_MAX_FEATURES
    ),
    n_jobs: int = RANDOM_FOREST_N_JOBS,
    random_state: int = RANDOM_SEED,
) -> RandomForestRegressor:
    """Create a configured Random Forest regressor."""

    validate_random_forest_settings(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
    )

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
    ):
        raise ValueError(
            "random_state must be an integer."
        )

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        criterion="squared_error",
        bootstrap=True,
    )


def predict_random_forest_demand(
    model: RandomForestRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict nonnegative demand."""

    if not isinstance(
        model,
        RandomForestRegressor,
    ):
        raise TypeError(
            "model must be a "
            "RandomForestRegressor."
        )

    check_is_fitted(model)

    if not isinstance(features, np.ndarray):
        raise TypeError(
            "features must be a NumPy array."
        )

    if features.ndim != 2:
        raise ValueError(
            "features must be two-dimensional."
        )

    if features.shape[0] == 0:
        raise ValueError(
            "features must not be empty."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "features contain non-finite values."
        )

    predictions = model.predict(features)

    return np.clip(
        predictions.astype(float),
        a_min=0.0,
        a_max=None,
    )


def create_random_forest_feature_importance(
    model: RandomForestRegressor,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Create a sorted feature-importance table."""

    if not isinstance(
        model,
        RandomForestRegressor,
    ):
        raise TypeError(
            "model must be a "
            "RandomForestRegressor."
        )

    check_is_fitted(model)

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "feature_names count must match "
            "the trained model features."
        )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return importance_table.sort_values(
        by=[
            "importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        ignore_index=True,
    )


def train_random_forest(
    processed_data: PreprocessedDataset,
    n_estimators: int = (
        RANDOM_FOREST_N_ESTIMATORS
    ),
    max_depth: int = RANDOM_FOREST_MAX_DEPTH,
    min_samples_split: int = (
        RANDOM_FOREST_MIN_SAMPLES_SPLIT
    ),
    min_samples_leaf: int = (
        RANDOM_FOREST_MIN_SAMPLES_LEAF
    ),
    max_features: str = (
        RANDOM_FOREST_MAX_FEATURES
    ),
    n_jobs: int = RANDOM_FOREST_N_JOBS,
    random_state: int = RANDOM_SEED,
) -> RandomForestResult:
    """Train a Random Forest demand model."""

    if not isinstance(
        processed_data,
        PreprocessedDataset,
    ):
        raise TypeError(
            "processed_data must be a "
            "PreprocessedDataset."
        )

    validate_supervised_arrays(
        features=processed_data.X_train,
        target=processed_data.y_train,
        dataset_name="Training",
    )

    validate_supervised_arrays(
        features=processed_data.X_validation,
        target=processed_data.y_validation,
        dataset_name="Validation",
    )

    validate_supervised_arrays(
        features=processed_data.X_test,
        target=processed_data.y_test,
        dataset_name="Test",
    )

    expected_feature_count = (
        processed_data.X_train.shape[1]
    )

    if (
        processed_data.X_validation.shape[1]
        != expected_feature_count
        or processed_data.X_test.shape[1]
        != expected_feature_count
    ):
        raise ValueError(
            "Train, validation, and test must "
            "have the same feature count."
        )

    model = build_random_forest_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    model.fit(
        processed_data.X_train,
        processed_data.y_train,
    )

    validation_predictions = (
        predict_random_forest_demand(
            model=model,
            features=(
                processed_data.X_validation
            ),
        )
    )

    test_predictions = (
        predict_random_forest_demand(
            model=model,
            features=processed_data.X_test,
        )
    )

    feature_importance = (
        create_random_forest_feature_importance(
            model=model,
            feature_names=(
                processed_data.feature_names
            ),
        )
    )

    return RandomForestResult(
        model=model,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,
        feature_importance=feature_importance,
    )


def main() -> None:
    """Train Random Forest using project data."""

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

    result = train_random_forest(
        processed_data=processed_data,
    )

    validation_mae = mean_absolute_error(
        processed_data.y_validation,
        result.validation_predictions,
    )

    validation_rmse = (
        root_mean_squared_error(
            processed_data.y_validation,
            result.validation_predictions,
        )
    )

    test_mae = mean_absolute_error(
        processed_data.y_test,
        result.test_predictions,
    )

    test_rmse = root_mean_squared_error(
        processed_data.y_test,
        result.test_predictions,
    )

    print(
        "Random Forest training completed "
        "successfully."
    )
    print(
        f"Number of trees: "
        f"{len(result.model.estimators_)}"
    )

    print()
    print(
        f"Validation MAE: "
        f"{validation_mae:.4f}"
    )
    print(
        f"Validation RMSE: "
        f"{validation_rmse:.4f}"
    )
    print(
        f"Test MAE: {test_mae:.4f}"
    )
    print(
        f"Test RMSE: {test_rmse:.4f}"
    )

    print()
    print("Top 10 important features:")
    print(
        result.feature_importance.head(
            10
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()