"""Train an AdaBoost model for demand forecasting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from src.config import (
    ADABOOST_BASE_MAX_DEPTH,
    ADABOOST_BASE_MIN_SAMPLES_LEAF,
    ADABOOST_BASE_MIN_SAMPLES_SPLIT,
    ADABOOST_LEARNING_RATE,
    ADABOOST_LOSS,
    ADABOOST_N_ESTIMATORS,
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
class AdaBoostResult:
    """Store a trained model and its predictions."""

    model: AdaBoostRegressor
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_importance: pd.DataFrame


def validate_adaboost_settings(
    n_estimators: int,
    learning_rate: float,
    base_max_depth: int,
    base_min_samples_split: int,
    base_min_samples_leaf: int,
    loss: str,
) -> None:
    """Validate AdaBoost hyperparameters."""

    integer_settings = {
        "n_estimators": n_estimators,
        "base_max_depth": base_max_depth,
        "base_min_samples_split": (
            base_min_samples_split
        ),
        "base_min_samples_leaf": (
            base_min_samples_leaf
        ),
    }

    for setting_name, setting_value in (
        integer_settings.items()
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

    if base_min_samples_split < 2:
        raise ValueError(
            "base_min_samples_split must be "
            "at least 2."
        )

    if (
        isinstance(learning_rate, bool)
        or not isinstance(
            learning_rate,
            (int, float),
        )
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be greater "
            "than zero."
        )

    if loss not in (
        "linear",
        "square",
        "exponential",
    ):
        raise ValueError(
            "loss must be 'linear', 'square', "
            "or 'exponential'."
        )


def build_adaboost_model(
    n_estimators: int = ADABOOST_N_ESTIMATORS,
    learning_rate: float = (
        ADABOOST_LEARNING_RATE
    ),
    base_max_depth: int = (
        ADABOOST_BASE_MAX_DEPTH
    ),
    base_min_samples_split: int = (
        ADABOOST_BASE_MIN_SAMPLES_SPLIT
    ),
    base_min_samples_leaf: int = (
        ADABOOST_BASE_MIN_SAMPLES_LEAF
    ),
    loss: str = ADABOOST_LOSS,
    random_state: int = RANDOM_SEED,
) -> AdaBoostRegressor:
    """Create a configured AdaBoost regressor."""

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

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
    ):
        raise ValueError(
            "random_state must be an integer."
        )

    base_estimator = DecisionTreeRegressor(
        max_depth=base_max_depth,
        min_samples_split=(
            base_min_samples_split
        ),
        min_samples_leaf=(
            base_min_samples_leaf
        ),
        criterion="squared_error",
        random_state=random_state,
    )

    return AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=random_state,
    )


def predict_adaboost_demand(
    model: AdaBoostRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict nonnegative demand."""

    if not isinstance(
        model,
        AdaBoostRegressor,
    ):
        raise TypeError(
            "model must be an AdaBoostRegressor."
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


def create_adaboost_feature_importance(
    model: AdaBoostRegressor,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Create a sorted feature-importance table."""

    if not isinstance(
        model,
        AdaBoostRegressor,
    ):
        raise TypeError(
            "model must be an AdaBoostRegressor."
        )

    check_is_fitted(model)

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    trained_feature_count = int(
        model.n_features_in_
    )

    if (
        len(feature_names)
        != trained_feature_count
    ):
        raise ValueError(
            "feature_names count must match "
            "the trained model features."
        )

    importances = model.feature_importances_

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


def train_adaboost(
    processed_data: PreprocessedDataset,
    n_estimators: int = ADABOOST_N_ESTIMATORS,
    learning_rate: float = (
        ADABOOST_LEARNING_RATE
    ),
    base_max_depth: int = (
        ADABOOST_BASE_MAX_DEPTH
    ),
    base_min_samples_split: int = (
        ADABOOST_BASE_MIN_SAMPLES_SPLIT
    ),
    base_min_samples_leaf: int = (
        ADABOOST_BASE_MIN_SAMPLES_LEAF
    ),
    loss: str = ADABOOST_LOSS,
    random_state: int = RANDOM_SEED,
) -> AdaBoostResult:
    """Train an AdaBoost demand model."""

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

    model = build_adaboost_model(
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
        random_state=random_state,
    )

    model.fit(
        processed_data.X_train,
        processed_data.y_train,
    )

    validation_predictions = (
        predict_adaboost_demand(
            model=model,
            features=(
                processed_data.X_validation
            ),
        )
    )

    test_predictions = (
        predict_adaboost_demand(
            model=model,
            features=processed_data.X_test,
        )
    )

    feature_importance = (
        create_adaboost_feature_importance(
            model=model,
            feature_names=(
                processed_data.feature_names
            ),
        )
    )

    return AdaBoostResult(
        model=model,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,
        feature_importance=feature_importance,
    )


def main() -> None:
    """Train AdaBoost using project data."""

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

    result = train_adaboost(
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
        "AdaBoost training completed successfully."
    )
    print(
        "Trained estimators: "
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