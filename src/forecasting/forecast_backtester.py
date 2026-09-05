"""Run rolling time-based backtests for demand forecasts."""

from dataclasses import dataclass

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_MIN_TRAINING_DAYS,
    FORECAST_BACKTEST_FOLDS,
    FORECAST_EVALUATION_HORIZONS,
    MOVING_AVERAGE_WINDOW_DAYS,
    SUPPORTED_FORECAST_METHODS,
    FORECAST_BACKTEST_RESULTS_FILE,
    FORECAST_BACKTEST_SUMMARY_FILE,
)
from src.data_loader import load_all_data
from src.forecasting.demand_forecaster import (
    validate_forecast_method,
)
from src.forecasting.forecast_evaluator import (
    evaluate_forecast_method,
)
from src.forecasting.historical_average import (
    REQUIRED_SALES_COLUMNS,
    validate_forecast_horizon,
    validate_sales_for_forecasting,
)


BACKTEST_RESULT_COLUMNS = (
    "fold",
    "method",
    "horizon_days",
    "training_start_date",
    "training_end_date",
    "training_days",
    "test_start_date",
    "test_end_date",
    "observations",
    "total_actual",
    "total_predicted",
    "mae",
    "rmse",
    "wape",
    "bias",
)

BACKTEST_SUMMARY_COLUMNS = (
    "rank",
    "horizon_days",
    "method",
    "folds",
    "mean_mae",
    "std_mae",
    "mean_rmse",
    "std_rmse",
    "mean_wape",
    "std_wape",
    "mean_bias",
)


@dataclass(frozen=True)
class BacktestWindow:
    """Describe one time-based backtest window."""

    fold: int
    training_start_date: pd.Timestamp
    training_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def validate_backtest_settings(
    number_of_folds: int,
    minimum_training_days: int,
    horizon_days: int,
    total_days: int,
) -> None:
    """Validate rolling backtest settings."""

    validate_forecast_horizon(horizon_days)

    if not isinstance(number_of_folds, int):
        raise TypeError(
            "number_of_folds must be an integer."
        )

    if number_of_folds <= 0:
        raise ValueError(
            "number_of_folds must be greater "
            "than zero."
        )

    if not isinstance(minimum_training_days, int):
        raise TypeError(
            "minimum_training_days must be "
            "an integer."
        )

    if minimum_training_days <= 0:
        raise ValueError(
            "minimum_training_days must be "
            "greater than zero."
        )

    minimum_required_days = (
        minimum_training_days + horizon_days
    )

    if total_days < minimum_required_days:
        raise ValueError(
            "Sales history does not contain enough "
            "dates for the requested training and "
            "forecast periods."
        )

    available_end_positions = (
        total_days - minimum_required_days + 1
    )

    if available_end_positions < number_of_folds:
        raise ValueError(
            "Sales history does not contain enough "
            "dates to create distinct backtest folds."
        )


def create_backtest_windows(
    sales: pd.DataFrame,
    horizon_days: int,
    number_of_folds: int = FORECAST_BACKTEST_FOLDS,
    minimum_training_days: int = (
        BACKTEST_MIN_TRAINING_DAYS
    ),
) -> list[BacktestWindow]:
    """Create expanding training and fixed test windows."""

    validate_sales_for_forecasting(sales)

    sales_dates = pd.to_datetime(
        sales["date"],
        errors="coerce",
    ).dt.normalize()

    if sales_dates.isna().any():
        raise ValueError(
            "Sales data contains invalid dates."
        )

    unique_dates = (
        sales_dates
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
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

    total_days = len(unique_dates)

    validate_backtest_settings(
        number_of_folds=number_of_folds,
        minimum_training_days=(
            minimum_training_days
        ),
        horizon_days=horizon_days,
        total_days=total_days,
    )

    first_test_end_position = (
        minimum_training_days
        + horizon_days
    )

    if number_of_folds == 1:
        test_end_positions = np.array(
            [total_days],
            dtype=int,
        )
    else:
        test_end_positions = np.linspace(
            first_test_end_position,
            total_days,
            num=number_of_folds,
            dtype=int,
        )

    if (
        len(np.unique(test_end_positions))
        != number_of_folds
    ):
        raise ValueError(
            "Backtest folds must have distinct "
            "test periods."
        )

    windows = []

    for fold, test_end_position in enumerate(
        test_end_positions,
        start=1,
    ):
        test_start_position = (
            test_end_position - horizon_days
        )

        training_end_position = (
            test_start_position - 1
        )

        window = BacktestWindow(
            fold=fold,
            training_start_date=(
                unique_dates.iloc[0]
            ),
            training_end_date=(
                unique_dates.iloc[
                    training_end_position
                ]
            ),
            test_start_date=(
                unique_dates.iloc[
                    test_start_position
                ]
            ),
            test_end_date=(
                unique_dates.iloc[
                    test_end_position - 1
                ]
            ),
        )

        windows.append(window)

    return windows


def run_forecast_backtest(
    sales: pd.DataFrame,
    method: str,
    horizon_days: int,
    number_of_folds: int = FORECAST_BACKTEST_FOLDS,
    minimum_training_days: int = (
        BACKTEST_MIN_TRAINING_DAYS
    ),
    moving_average_window_days: int = (
        MOVING_AVERAGE_WINDOW_DAYS
    ),
) -> pd.DataFrame:
    """Backtest one forecast method and horizon."""

    selected_method = validate_forecast_method(
        method
    )

    windows = create_backtest_windows(
        sales=sales,
        horizon_days=horizon_days,
        number_of_folds=number_of_folds,
        minimum_training_days=(
            minimum_training_days
        ),
    )

    working_sales = sales[
        list(REQUIRED_SALES_COLUMNS)
    ].copy()

    working_sales["date"] = pd.to_datetime(
        working_sales["date"]
    ).dt.normalize()

    metric_records = []

    for window in windows:
        available_sales = working_sales.loc[
            working_sales["date"]
            <= window.test_end_date
        ].reset_index(drop=True)

        evaluation = evaluate_forecast_method(
            sales=available_sales,
            method=selected_method,
            horizon_days=horizon_days,
            moving_average_window_days=(
                moving_average_window_days
            ),
        )

        training_days = (
            window.training_end_date
            - window.training_start_date
        ).days + 1

        metrics = evaluation.metrics

        metric_records.append(
            {
                "fold": window.fold,
                "method": metrics["method"],
                "horizon_days": (
                    metrics["horizon_days"]
                ),
                "training_start_date": (
                    window.training_start_date
                ),
                "training_end_date": (
                    window.training_end_date
                ),
                "training_days": training_days,
                "test_start_date": (
                    window.test_start_date
                ),
                "test_end_date": (
                    window.test_end_date
                ),
                "observations": (
                    metrics["observations"]
                ),
                "total_actual": (
                    metrics["total_actual"]
                ),
                "total_predicted": (
                    metrics["total_predicted"]
                ),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape": metrics["wape"],
                "bias": metrics["bias"],
            }
        )

    return pd.DataFrame(
        metric_records,
        columns=BACKTEST_RESULT_COLUMNS,
    )


def summarize_backtest_results(
    backtest_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics across backtest folds."""

    if not isinstance(
        backtest_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "backtest_results must be a "
            "pandas DataFrame."
        )

    if backtest_results.empty:
        raise ValueError(
            "backtest_results must not be empty."
        )

    missing_columns = (
        set(BACKTEST_RESULT_COLUMNS)
        - set(backtest_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "backtest_results is missing columns: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        backtest_results.groupby(
            [
                "horizon_days",
                "method",
            ],
            as_index=False,
        )
        .agg(
            folds=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=(
                "mae",
                lambda values: values.std(ddof=0),
            ),
            mean_rmse=("rmse", "mean"),
            std_rmse=(
                "rmse",
                lambda values: values.std(ddof=0),
            ),
            mean_wape=("wape", "mean"),
            std_wape=(
                "wape",
                lambda values: values.std(ddof=0),
            ),
            mean_bias=("bias", "mean"),
        )
    )

    metric_columns = [
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "std_rmse",
        "mean_wape",
        "std_wape",
        "mean_bias",
    ]

    summary[metric_columns] = summary[
        metric_columns
    ].round(4)

    summary = summary.sort_values(
        by=[
            "horizon_days",
            "mean_mae",
            "mean_rmse",
            "mean_wape",
            "method",
        ],
        ignore_index=True,
    )

    summary["rank"] = (
        summary.groupby(
            "horizon_days"
        ).cumcount() + 1
    )

    return summary[
        list(BACKTEST_SUMMARY_COLUMNS)
    ]


def compare_methods_with_backtesting(
    sales: pd.DataFrame,
    horizons: tuple[int, ...] = (
        FORECAST_EVALUATION_HORIZONS
    ),
    methods: tuple[str, ...] = (
        SUPPORTED_FORECAST_METHODS
    ),
    number_of_folds: int = FORECAST_BACKTEST_FOLDS,
    minimum_training_days: int = (
        BACKTEST_MIN_TRAINING_DAYS
    ),
    moving_average_window_days: int = (
        MOVING_AVERAGE_WINDOW_DAYS
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest all requested methods and horizons."""

    if not horizons:
        raise ValueError(
            "horizons must not be empty."
        )

    if not methods:
        raise ValueError(
            "methods must not be empty."
        )

    result_frames = []

    for horizon_days in horizons:
        for method in methods:
            method_results = run_forecast_backtest(
                sales=sales,
                method=method,
                horizon_days=horizon_days,
                number_of_folds=number_of_folds,
                minimum_training_days=(
                    minimum_training_days
                ),
                moving_average_window_days=(
                    moving_average_window_days
                ),
            )

            result_frames.append(
                method_results
            )

    backtest_results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    summary = summarize_backtest_results(
        backtest_results
    )

    return backtest_results, summary

def save_backtest_outputs(
    backtest_results: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    results_output_path: str | Path = (
        FORECAST_BACKTEST_RESULTS_FILE
    ),
    summary_output_path: str | Path = (
        FORECAST_BACKTEST_SUMMARY_FILE
    ),
) -> tuple[Path, Path]:
    """Validate and save detailed and summarized backtests."""

    if not isinstance(
        backtest_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "backtest_results must be a "
            "pandas DataFrame."
        )

    if not isinstance(
        backtest_summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "backtest_summary must be a "
            "pandas DataFrame."
        )

    if backtest_results.empty:
        raise ValueError(
            "backtest_results must not be empty."
        )

    if backtest_summary.empty:
        raise ValueError(
            "backtest_summary must not be empty."
        )

    missing_result_columns = (
        set(BACKTEST_RESULT_COLUMNS)
        - set(backtest_results.columns)
    )

    if missing_result_columns:
        raise ValueError(
            "backtest_results is missing columns: "
            f"{sorted(missing_result_columns)}"
        )

    missing_summary_columns = (
        set(BACKTEST_SUMMARY_COLUMNS)
        - set(backtest_summary.columns)
    )

    if missing_summary_columns:
        raise ValueError(
            "backtest_summary is missing columns: "
            f"{sorted(missing_summary_columns)}"
        )

    results_destination = Path(
        results_output_path
    )

    summary_destination = Path(
        summary_output_path
    )

    if (
        results_destination.resolve()
        == summary_destination.resolve()
    ):
        raise ValueError(
            "Backtest results and summary must use "
            "different output paths."
        )

    results_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest_results.to_csv(
        results_destination,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    backtest_summary.to_csv(
        summary_destination,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        results_destination.resolve(),
        summary_destination.resolve(),
    )

def main() -> None:
    """Run backtests for all baseline forecast methods."""

    project_data = load_all_data()

    (
        backtest_results,
        backtest_summary,
    ) = compare_methods_with_backtesting(
        sales=project_data.sales
    )

    (
        results_output_path,
        summary_output_path,
    ) = save_backtest_outputs(
        backtest_results=backtest_results,
        backtest_summary=backtest_summary,
    )

    print(
        "Forecast backtesting completed "
        "successfully."
    )

    print(
        f"Backtest rows: {len(backtest_results)}"
    )

    print(
        "Detailed results: "
        f"{results_output_path}"
    )

    print(
        "Summary results: "
        f"{summary_output_path}"
    )

    print()

    print(
        backtest_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()