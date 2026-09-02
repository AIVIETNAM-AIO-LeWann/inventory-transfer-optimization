"""Evaluate demand forecasts using time-based holdout data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_EVALUATION_HORIZONS,
    MOVING_AVERAGE_WINDOW_DAYS,
    SUPPORTED_FORECAST_METHODS,
)
from src.data_loader import load_all_data
from src.forecasting.demand_forecaster import (
    forecast_demand,
    validate_forecast_method,
)
from src.forecasting.historical_average import (
    REQUIRED_SALES_COLUMNS,
    validate_forecast_horizon,
    validate_sales_for_forecasting,
)


EVALUATION_COLUMNS = (
    "store_id",
    "product_id",
    "date",
    "forecast_day",
    "method",
    "actual_quantity",
    "predicted_quantity",
    "error",
    "absolute_error",
    "squared_error",
)


@dataclass
class ForecastEvaluationResult:
    """Store all outputs from one forecast evaluation."""

    method: str
    horizon_days: int
    training_sales: pd.DataFrame
    test_sales: pd.DataFrame
    daily_forecast: pd.DataFrame
    comparison: pd.DataFrame
    metrics: dict[str, str | int | float]


def split_sales_by_time(
    sales: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sales into earlier training and later test periods."""

    validate_forecast_horizon(test_days)
    validate_sales_for_forecasting(sales)

    working_sales = sales[
        list(REQUIRED_SALES_COLUMNS)
    ].copy()

    working_sales["date"] = pd.to_datetime(
        working_sales["date"]
    ).dt.normalize()

    unique_dates = (
        working_sales["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_dates) <= test_days:
        raise ValueError(
            "Sales history must contain more dates "
            "than the requested test period."
        )

    expected_dates = pd.date_range(
        start=unique_dates.iloc[0],
        end=unique_dates.iloc[-1],
        freq="D",
    )

    if len(expected_dates) != len(unique_dates):
        raise ValueError(
            "Sales history must contain a continuous "
            "daily date sequence."
        )

    test_start_date = unique_dates.iloc[
        -test_days
    ]

    training_sales = working_sales.loc[
        working_sales["date"] < test_start_date
    ].reset_index(drop=True)

    test_sales = working_sales.loc[
        working_sales["date"] >= test_start_date
    ].reset_index(drop=True)

    if training_sales.empty:
        raise ValueError(
            "Training sales must not be empty."
        )

    if test_sales.empty:
        raise ValueError(
            "Test sales must not be empty."
        )

    return training_sales, test_sales


def align_forecast_with_actuals(
    daily_forecast: pd.DataFrame,
    test_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Align forecasted quantities with actual test sales."""

    if daily_forecast.empty:
        raise ValueError(
            "daily_forecast must not be empty."
        )

    if test_sales.empty:
        raise ValueError(
            "test_sales must not be empty."
        )

    forecast_values = daily_forecast.copy()

    forecast_values["forecast_date"] = pd.to_datetime(
        forecast_values["forecast_date"],
        errors="coerce",
    ).dt.normalize()

    if forecast_values[
        "forecast_date"
    ].isna().any():
        raise ValueError(
            "daily_forecast contains invalid dates."
        )

    forecast_values = forecast_values.rename(
        columns={
            "forecast_date": "date",
        }
    )

    actual_values = test_sales.copy()

    actual_values["date"] = pd.to_datetime(
        actual_values["date"],
        errors="coerce",
    ).dt.normalize()

    if actual_values["date"].isna().any():
        raise ValueError(
            "test_sales contains invalid dates."
        )

    actual_values = (
        actual_values.groupby(
            [
                "store_id",
                "product_id",
                "date",
            ],
            as_index=False,
        )
        .agg(
            actual_quantity=(
                "quantity_sold",
                "sum",
            )
        )
    )

    comparison = forecast_values.merge(
        actual_values,
        on=[
            "store_id",
            "product_id",
            "date",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_rows = (
        comparison["_merge"] != "both"
    ).sum()

    if unmatched_rows:
        raise ValueError(
            "Forecast and actual sales could not be "
            f"aligned for {unmatched_rows} rows."
        )

    comparison = comparison.drop(
        columns="_merge"
    )

    comparison["error"] = (
        comparison["predicted_quantity"]
        - comparison["actual_quantity"]
    )

    comparison["absolute_error"] = (
        comparison["error"].abs()
    )

    comparison["squared_error"] = (
        comparison["error"] ** 2
    )

    comparison = comparison.sort_values(
        by=[
            "store_id",
            "product_id",
            "date",
        ],
        ignore_index=True,
    )

    return comparison[
        list(EVALUATION_COLUMNS)
    ]


def calculate_forecast_metrics(
    comparison: pd.DataFrame,
) -> dict[str, str | int | float]:
    """Calculate MAE, RMSE, WAPE, and forecast bias."""

    if not isinstance(comparison, pd.DataFrame):
        raise TypeError(
            "comparison must be a pandas DataFrame."
        )

    if comparison.empty:
        raise ValueError(
            "comparison must not be empty."
        )

    missing_columns = (
        set(EVALUATION_COLUMNS)
        - set(comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "comparison is missing columns: "
            f"{sorted(missing_columns)}"
        )

    methods = comparison["method"].unique()

    if len(methods) != 1:
        raise ValueError(
            "comparison must contain exactly "
            "one forecast method."
        )

    horizon_days = int(
        comparison["forecast_day"].max()
    )

    total_actual = float(
        comparison["actual_quantity"].sum()
    )

    total_predicted = float(
        comparison["predicted_quantity"].sum()
    )

    total_absolute_error = float(
        comparison["absolute_error"].sum()
    )

    mae = float(
        comparison["absolute_error"].mean()
    )

    rmse = float(
        np.sqrt(
            comparison["squared_error"].mean()
        )
    )

    if total_actual > 0:
        wape = (
            total_absolute_error / total_actual
        )
    elif total_absolute_error == 0:
        wape = 0.0
    else:
        wape = float("nan")

    bias = float(
        comparison["error"].mean()
    )

    return {
        "method": str(methods[0]),
        "horizon_days": horizon_days,
        "observations": len(comparison),
        "total_actual": round(
            total_actual,
            2,
        ),
        "total_predicted": round(
            total_predicted,
            2,
        ),
        "mae": round(
            mae,
            4,
        ),
        "rmse": round(
            rmse,
            4,
        ),
        "wape": round(
            wape,
            4,
        ),
        "bias": round(
            bias,
            4,
        ),
    }


def evaluate_forecast_method(
    sales: pd.DataFrame,
    method: str,
    horizon_days: int,
    moving_average_window_days: int = (
        MOVING_AVERAGE_WINDOW_DAYS
    ),
) -> ForecastEvaluationResult:
    """Evaluate one forecasting method on holdout data."""

    selected_method = validate_forecast_method(
        method
    )

    training_sales, test_sales = (
        split_sales_by_time(
            sales=sales,
            test_days=horizon_days,
        )
    )

    daily_forecast = forecast_demand(
        sales=training_sales,
        method=selected_method,
        horizon_days=horizon_days,
        moving_average_window_days=(
            moving_average_window_days
        ),
    )

    comparison = align_forecast_with_actuals(
        daily_forecast=daily_forecast,
        test_sales=test_sales,
    )

    metrics = calculate_forecast_metrics(
        comparison
    )

    return ForecastEvaluationResult(
        method=selected_method,
        horizon_days=horizon_days,
        training_sales=training_sales,
        test_sales=test_sales,
        daily_forecast=daily_forecast,
        comparison=comparison,
        metrics=metrics,
    )


def compare_forecast_methods(
    sales: pd.DataFrame,
    horizon_days: int,
    methods: tuple[str, ...] = (
        SUPPORTED_FORECAST_METHODS
    ),
    moving_average_window_days: int = (
        MOVING_AVERAGE_WINDOW_DAYS
    ),
) -> pd.DataFrame:
    """Compare supported forecasting methods."""

    if not methods:
        raise ValueError(
            "methods must not be empty."
        )

    metric_records = []

    for method in methods:
        evaluation = evaluate_forecast_method(
            sales=sales,
            method=method,
            horizon_days=horizon_days,
            moving_average_window_days=(
                moving_average_window_days
            ),
        )

        metric_records.append(
            evaluation.metrics
        )

    comparison = pd.DataFrame(
        metric_records
    )

    comparison = comparison.sort_values(
        by=[
            "mae",
            "rmse",
            "wape",
            "method",
        ],
        ignore_index=True,
    )

    comparison.insert(
        0,
        "rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    return comparison


def main() -> None:
    """Evaluate all baseline forecasting methods."""

    project_data = load_all_data()

    for horizon_days in (
        FORECAST_EVALUATION_HORIZONS
    ):
        comparison = compare_forecast_methods(
            sales=project_data.sales,
            horizon_days=horizon_days,
        )

        print()
        print(
            f"Forecast horizon: {horizon_days} days"
        )
        print(
            comparison.to_string(index=False)
        )


if __name__ == "__main__":
    main()