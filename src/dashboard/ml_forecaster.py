"""Create recursive future-demand forecasts using trained models."""

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_LAG_DAYS,
    FORECAST_ROLLING_WINDOWS,
)
from src.dashboard.model_artifacts import (
    ADABOOST_METHOD,
    RANDOM_FOREST_METHOD,
    ForecastModelArtifact,
    validate_dataset_fingerprint,
    validate_model_artifact,
)
from src.data_loader import ProjectData
from src.forecasting.adaboost import (
    predict_adaboost_demand,
)
from src.forecasting.demand_forecaster import (
    get_replenishment_horizon,
)
from src.forecasting.feature_engineering import (
    validate_complete_daily_history,
)
from src.forecasting.historical_average import (
    REQUIRED_SALES_COLUMNS,
    validate_sales_for_forecasting,
)
from src.forecasting.random_forest import (
    predict_random_forest_demand,
)
from src.forecasting.training_dataset import (
    MODEL_FEATURE_COLUMNS,
)


HistoryKey = tuple[str, str]
DemandHistory = dict[HistoryKey, list[float]]


def prepare_demand_history(
    sales: pd.DataFrame,
) -> tuple[
    DemandHistory,
    pd.Timestamp,
    pd.Timestamp,
]:
    """Prepare chronological demand history for each pair."""

    validate_sales_for_forecasting(sales)

    working_sales = sales[
        list(REQUIRED_SALES_COLUMNS)
    ].copy()

    working_sales["date"] = pd.to_datetime(
        working_sales["date"],
        errors="coerce",
    ).dt.normalize()

    if working_sales["date"].isna().any():
        raise ValueError(
            "sales contains invalid dates."
        )

    working_sales["store_id"] = (
        working_sales["store_id"].astype(str)
    )

    working_sales["product_id"] = (
        working_sales["product_id"].astype(str)
    )

    working_sales = working_sales.sort_values(
        by=[
            "store_id",
            "product_id",
            "date",
        ],
        ignore_index=True,
    )

    validate_complete_daily_history(
        working_sales
    )

    required_history_days = max(
        *FORECAST_LAG_DAYS,
        *FORECAST_ROLLING_WINDOWS,
    )

    history: DemandHistory = {}

    grouped_sales = working_sales.groupby(
        [
            "store_id",
            "product_id",
        ],
        sort=True,
    )

    for pair, pair_sales in grouped_sales:
        quantities = (
            pair_sales["quantity_sold"]
            .to_numpy(dtype=float)
            .tolist()
        )

        if len(quantities) < required_history_days:
            raise ValueError(
                f"Store-product pair {pair} requires "
                f"at least {required_history_days} "
                "days of sales history."
            )

        history[
            (
                str(pair[0]),
                str(pair[1]),
            )
        ] = quantities

    first_date = pd.Timestamp(
        working_sales["date"].min()
    )

    last_date = pd.Timestamp(
        working_sales["date"].max()
    )

    return (
        history,
        first_date,
        last_date,
    )


def create_pair_reference_data(
    project_data: ProjectData,
    history: DemandHistory,
) -> pd.DataFrame:
    """Create store-product metadata required by the model."""

    if not isinstance(
        project_data,
        ProjectData,
    ):
        raise TypeError(
            "project_data must be ProjectData."
        )

    inventory_pairs = project_data.inventory[
        [
            "store_id",
            "product_id",
        ]
    ].copy()

    inventory_pairs["store_id"] = (
        inventory_pairs["store_id"].astype(str)
    )

    inventory_pairs["product_id"] = (
        inventory_pairs["product_id"].astype(str)
    )

    inventory_pairs = (
        inventory_pairs.drop_duplicates()
    )

    store_reference = project_data.stores[
        [
            "store_id",
            "city",
        ]
    ].copy()

    store_reference["store_id"] = (
        store_reference["store_id"].astype(str)
    )

    product_reference = project_data.products[
        [
            "product_id",
            "category",
            "cost",
            "price",
        ]
    ].copy()

    product_reference["product_id"] = (
        product_reference["product_id"]
        .astype(str)
    )

    pair_reference = inventory_pairs.merge(
        store_reference,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    pair_reference = pair_reference.merge(
        product_reference,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    required_columns = [
        "store_id",
        "product_id",
        "city",
        "category",
        "cost",
        "price",
    ]

    if pair_reference[
        required_columns
    ].isna().any().any():
        raise ValueError(
            "Store-product reference data contains "
            "missing model values."
        )

    missing_history_pairs = sorted(
        {
            (
                row.store_id,
                row.product_id,
            )
            for row in pair_reference.itertuples(
                index=False
            )
        }
        - set(history)
    )

    if missing_history_pairs:
        raise ValueError(
            "Inventory contains store-product pairs "
            "without sales history: "
            f"{missing_history_pairs[:10]}"
        )

    return pair_reference.sort_values(
        by=[
            "store_id",
            "product_id",
        ],
        ignore_index=True,
    )


def create_future_feature_frame(
    pair_reference: pd.DataFrame,
    history: DemandHistory,
    forecast_date: pd.Timestamp,
    first_history_date: pd.Timestamp,
) -> pd.DataFrame:
    """Create raw model features for one future date."""

    feature_frame = pair_reference.copy()

    pair_keys = [
        (
            str(row.store_id),
            str(row.product_id),
        )
        for row in pair_reference.itertuples(
            index=False
        )
    ]

    normalized_date = pd.Timestamp(
        forecast_date
    ).normalize()

    feature_frame["day_of_week"] = (
        normalized_date.dayofweek
    )

    feature_frame["day_of_month"] = (
        normalized_date.day
    )

    feature_frame["week_of_year"] = int(
        normalized_date.isocalendar().week
    )

    feature_frame["month"] = (
        normalized_date.month
    )

    feature_frame["quarter"] = (
        normalized_date.quarter
    )

    feature_frame["is_weekend"] = int(
        normalized_date.dayofweek >= 5
    )

    feature_frame["time_index"] = (
        normalized_date
        - first_history_date
    ).days

    for lag_day in FORECAST_LAG_DAYS:
        feature_frame[
            f"lag_{lag_day}"
        ] = [
            history[pair][-lag_day]
            for pair in pair_keys
        ]

    for window in FORECAST_ROLLING_WINDOWS:
        feature_frame[
            f"rolling_mean_{window}"
        ] = [
            float(
                np.mean(
                    history[pair][-window:]
                )
            )
            for pair in pair_keys
        ]

    model_features = feature_frame[
        list(MODEL_FEATURE_COLUMNS)
    ].copy()

    if model_features.isna().any().any():
        raise ValueError(
            "Future model features contain "
            "missing values."
        )

    return model_features


def predict_future_demand(
    artifact: ForecastModelArtifact,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Transform features and predict nonnegative demand."""

    validate_model_artifact(artifact)

    transformed_features = np.asarray(
        artifact.preprocessor.transform(
            feature_frame[
                list(MODEL_FEATURE_COLUMNS)
            ]
        ),
        dtype=float,
    )

    if transformed_features.ndim != 2:
        raise ValueError(
            "Transformed future features must "
            "be two-dimensional."
        )

    if transformed_features.shape[0] == 0:
        raise ValueError(
            "Transformed future features "
            "must not be empty."
        )

    if not np.isfinite(
        transformed_features
    ).all():
        raise ValueError(
            "Transformed future features contain "
            "non-finite values."
        )

    if (
        transformed_features.shape[1]
        != len(artifact.feature_names)
    ):
        raise ValueError(
            "The model artifact is incompatible "
            "with the transformed features."
        )

    if artifact.method == RANDOM_FOREST_METHOD:
        predictions = (
            predict_random_forest_demand(
                model=artifact.model,
                features=transformed_features,
            )
        )

    elif artifact.method == ADABOOST_METHOD:
        predictions = (
            predict_adaboost_demand(
                model=artifact.model,
                features=transformed_features,
            )
        )

    else:
        raise RuntimeError(
            "The artifact method has no "
            "prediction implementation."
        )

    return np.round(
        predictions,
        decimals=4,
    )


def append_predictions_to_history(
    history: DemandHistory,
    pair_reference: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """Append one forecast day to recursive history."""

    if len(pair_reference) != len(predictions):
        raise ValueError(
            "Prediction count must match "
            "store-product pair count."
        )

    for row, prediction in zip(
        pair_reference.itertuples(
            index=False
        ),
        predictions,
    ):
        pair = (
            str(row.store_id),
            str(row.product_id),
        )

        history[pair].append(
            float(prediction)
        )


def create_daily_forecast_rows(
    pair_reference: pd.DataFrame,
    predictions: np.ndarray,
    forecast_date: pd.Timestamp,
    forecast_day: int,
    method: str,
) -> pd.DataFrame:
    """Create dashboard forecast rows for one day."""

    if len(pair_reference) != len(predictions):
        raise ValueError(
            "Prediction count must match "
            "store-product pair count."
        )

    return pd.DataFrame(
        {
            "store_id": (
                pair_reference["store_id"]
                .to_numpy()
            ),
            "product_id": (
                pair_reference["product_id"]
                .to_numpy()
            ),
            "forecast_date": forecast_date,
            "forecast_day": forecast_day,
            "predicted_quantity": predictions,
            "method": method,
        }
    )


def forecast_machine_learning_demand(
    project_data: ProjectData,
    artifact: ForecastModelArtifact,
    dataset_fingerprint: str,
    requested_horizon_days: int,
) -> pd.DataFrame:
    """Forecast future demand recursively for optimization."""

    validate_model_artifact(artifact)

    normalized_fingerprint = (
        validate_dataset_fingerprint(
            dataset_fingerprint
        )
    )

    if (
        artifact.dataset_fingerprint
        != normalized_fingerprint
    ):
        raise ValueError(
            "The model artifact does not match "
            "the active dataset."
        )

    replenishment_horizon_days = (
        get_replenishment_horizon(
            requested_horizon_days
        )
    )

    (
        history,
        first_history_date,
        last_history_date,
    ) = prepare_demand_history(
        project_data.sales
    )

    pair_reference = (
        create_pair_reference_data(
            project_data=project_data,
            history=history,
        )
    )

    forecast_frames: list[pd.DataFrame] = []

    for forecast_day in range(
        1,
        replenishment_horizon_days + 1,
    ):
        forecast_date = (
            last_history_date
            + pd.Timedelta(days=forecast_day)
        )

        feature_frame = (
            create_future_feature_frame(
                pair_reference=pair_reference,
                history=history,
                forecast_date=forecast_date,
                first_history_date=(
                    first_history_date
                ),
            )
        )

        predictions = predict_future_demand(
            artifact=artifact,
            feature_frame=feature_frame,
        )

        daily_forecast = (
            create_daily_forecast_rows(
                pair_reference=pair_reference,
                predictions=predictions,
                forecast_date=forecast_date,
                forecast_day=forecast_day,
                method=artifact.method,
            )
        )

        forecast_frames.append(
            daily_forecast
        )

        append_predictions_to_history(
            history=history,
            pair_reference=pair_reference,
            predictions=predictions,
        )

    return pd.concat(
        forecast_frames,
        ignore_index=True,
    )