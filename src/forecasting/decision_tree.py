"""Train a Decision Tree model for demand forecasting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from src.config import (
    DECISION_TREE_MAX_DEPTH,
    DECISION_TREE_MIN_SAMPLES_LEAF,
    DECISION_TREE_MIN_SAMPLES_SPLIT,
    RANDOM_SEED,
)
from src.data_loader import load_all_data
from src.forecasting.model_preprocessor import (
    PreprocessedDataset,
    preprocess_training_dataset,
)
from src.forecasting.training_dataset import (
    build_training_dataset,
)


@dataclass(frozen=True)
class DecisionTreeResult:
    """Store a trained model and its predictions."""

    model: DecisionTreeRegressor
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_importance: pd.DataFrame


def validate_decision_tree_settings(
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
) -> None:
    """Validate Decision Tree hyperparameters."""

    settings = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

    for setting_name, setting_value in (
        settings.items()
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


def build_decision_tree_model(
    max_depth: int = DECISION_TREE_MAX_DEPTH,
    min_samples_split: int = (
        DECISION_TREE_MIN_SAMPLES_SPLIT
    ),
    min_samples_leaf: int = (
        DECISION_TREE_MIN_SAMPLES_LEAF
    ),
    random_state: int = RANDOM_SEED,
) -> DecisionTreeRegressor:
    """Create a configured Decision Tree regressor."""

    validate_decision_tree_settings(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
    ):
        raise ValueError(
            "random_state must be an integer."
        )

    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion="squared_error",
    )


def validate_supervised_arrays(
    features: np.ndarray,
    target: np.ndarray,
    dataset_name: str,
) -> None:
    """Validate feature and target arrays."""

    if not isinstance(features, np.ndarray):
        raise TypeError(
            f"{dataset_name} features must be "
            "a NumPy array."
        )

    if not isinstance(target, np.ndarray):
        raise TypeError(
            f"{dataset_name} target must be "
            "a NumPy array."
        )

    if features.ndim != 2:
        raise ValueError(
            f"{dataset_name} features must be "
            "two-dimensional."
        )

    if target.ndim != 1:
        raise ValueError(
            f"{dataset_name} target must be "
            "one-dimensional."
        )

    if len(features) == 0:
        raise ValueError(
            f"{dataset_name} must not be empty."
        )

    if len(features) != len(target):
        raise ValueError(
            f"{dataset_name} features and target "
            "must have the same number of rows."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{dataset_name} features contain "
            "non-finite values."
        )

    if not np.isfinite(target).all():
        raise ValueError(
            f"{dataset_name} target contains "
            "non-finite values."
        )

    if (target < 0).any():
        raise ValueError(
            f"{dataset_name} target must not "
            "contain negative values."
        )


def predict_nonnegative_demand(
    model: DecisionTreeRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict demand and replace negatives with zero."""

    if not isinstance(
        model,
        DecisionTreeRegressor,
    ):
        raise TypeError(
            "model must be a DecisionTreeRegressor."
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


def create_feature_importance(
    model: DecisionTreeRegressor,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Create a sorted feature-importance table."""

    if not isinstance(
        model,
        DecisionTreeRegressor,
    ):
        raise TypeError(
            "model must be a DecisionTreeRegressor."
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


def train_decision_tree(
    processed_data: PreprocessedDataset,
    max_depth: int = DECISION_TREE_MAX_DEPTH,
    min_samples_split: int = (
        DECISION_TREE_MIN_SAMPLES_SPLIT
    ),
    min_samples_leaf: int = (
        DECISION_TREE_MIN_SAMPLES_LEAF
    ),
    random_state: int = RANDOM_SEED,
) -> DecisionTreeResult:
    """Train and evaluate a Decision Tree model."""

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

    model = build_decision_tree_model(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    model.fit(
        processed_data.X_train,
        processed_data.y_train,
    )

    validation_predictions = (
        predict_nonnegative_demand(
            model=model,
            features=(
                processed_data.X_validation
            ),
        )
    )

    test_predictions = (
        predict_nonnegative_demand(
            model=model,
            features=processed_data.X_test,
        )
    )

    feature_importance = (
        create_feature_importance(
            model=model,
            feature_names=(
                processed_data.feature_names
            ),
        )
    )

    return DecisionTreeResult(
        model=model,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,
        feature_importance=feature_importance,
    )


def main() -> None:
    """Train Decision Tree using project data."""

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

    result = train_decision_tree(
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
        "Decision Tree training completed "
        "successfully."
    )
    print(
        f"Tree depth: "
        f"{result.model.get_depth()}"
    )
    print(
        f"Leaf count: "
        f"{result.model.get_n_leaves()}"
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