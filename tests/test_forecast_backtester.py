"""Tests for rolling forecast backtesting."""

import pandas as pd
import pytest
from pathlib import Path

from src.config import (
    HISTORICAL_AVERAGE_METHOD,
    MOVING_AVERAGE_METHOD,
)
from src.forecasting.forecast_backtester import (
    BACKTEST_RESULT_COLUMNS,
    BacktestWindow,
    compare_methods_with_backtesting,
    create_backtest_windows,
    run_forecast_backtest,
    save_backtest_outputs,
    summarize_backtest_results,
    validate_backtest_settings,
)


@pytest.fixture
def constant_sales() -> pd.DataFrame:
    """Create twenty days of constant sales."""

    dates = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "store_id": "S001",
            "product_id": "P001",
            "quantity_sold": 10,
        }
    )


def create_summary_input() -> pd.DataFrame:
    """Create backtest results for summary tests."""

    rows = [
        {
            "fold": 1,
            "method": HISTORICAL_AVERAGE_METHOD,
            "horizon_days": 3,
            "training_start_date": pd.Timestamp(
                "2026-01-01"
            ),
            "training_end_date": pd.Timestamp(
                "2026-01-10"
            ),
            "training_days": 10,
            "test_start_date": pd.Timestamp(
                "2026-01-11"
            ),
            "test_end_date": pd.Timestamp(
                "2026-01-13"
            ),
            "observations": 3,
            "total_actual": 30,
            "total_predicted": 24,
            "mae": 4,
            "rmse": 5,
            "wape": 0.4,
            "bias": -2,
        },
        {
            "fold": 2,
            "method": HISTORICAL_AVERAGE_METHOD,
            "horizon_days": 3,
            "training_start_date": pd.Timestamp(
                "2026-01-01"
            ),
            "training_end_date": pd.Timestamp(
                "2026-01-17"
            ),
            "training_days": 17,
            "test_start_date": pd.Timestamp(
                "2026-01-18"
            ),
            "test_end_date": pd.Timestamp(
                "2026-01-20"
            ),
            "observations": 3,
            "total_actual": 30,
            "total_predicted": 28,
            "mae": 2,
            "rmse": 3,
            "wape": 0.2,
            "bias": 2,
        },
        {
            "fold": 1,
            "method": MOVING_AVERAGE_METHOD,
            "horizon_days": 3,
            "training_start_date": pd.Timestamp(
                "2026-01-01"
            ),
            "training_end_date": pd.Timestamp(
                "2026-01-10"
            ),
            "training_days": 10,
            "test_start_date": pd.Timestamp(
                "2026-01-11"
            ),
            "test_end_date": pd.Timestamp(
                "2026-01-13"
            ),
            "observations": 3,
            "total_actual": 30,
            "total_predicted": 30,
            "mae": 1,
            "rmse": 2,
            "wape": 0.1,
            "bias": 0,
        },
        {
            "fold": 2,
            "method": MOVING_AVERAGE_METHOD,
            "horizon_days": 3,
            "training_start_date": pd.Timestamp(
                "2026-01-01"
            ),
            "training_end_date": pd.Timestamp(
                "2026-01-17"
            ),
            "training_days": 17,
            "test_start_date": pd.Timestamp(
                "2026-01-18"
            ),
            "test_end_date": pd.Timestamp(
                "2026-01-20"
            ),
            "observations": 3,
            "total_actual": 30,
            "total_predicted": 30,
            "mae": 1,
            "rmse": 2,
            "wape": 0.1,
            "bias": 0,
        },
    ]

    return pd.DataFrame(
        rows,
        columns=BACKTEST_RESULT_COLUMNS,
    )


def test_validate_backtest_settings_accepts_valid_values() -> None:
    """Valid backtest settings should be accepted."""

    validate_backtest_settings(
        number_of_folds=3,
        minimum_training_days=10,
        horizon_days=3,
        total_days=20,
    )


@pytest.mark.parametrize(
    (
        "number_of_folds",
        "minimum_training_days",
        "horizon_days",
        "total_days",
    ),
    [
        (0, 10, 3, 20),
        (3, 0, 3, 20),
        (3, 10, 3, 12),
        (9, 10, 3, 20),
    ],
)
def test_validate_backtest_settings_rejects_invalid_values(
    number_of_folds: int,
    minimum_training_days: int,
    horizon_days: int,
    total_days: int,
) -> None:
    """Invalid backtest settings should be rejected."""

    with pytest.raises(ValueError):
        validate_backtest_settings(
            number_of_folds=number_of_folds,
            minimum_training_days=(
                minimum_training_days
            ),
            horizon_days=horizon_days,
            total_days=total_days,
        )


def test_create_backtest_windows(
    constant_sales: pd.DataFrame,
) -> None:
    """Backtest windows should use expanding training data."""

    windows = create_backtest_windows(
        sales=constant_sales,
        horizon_days=3,
        number_of_folds=3,
        minimum_training_days=10,
    )

    assert len(windows) == 3
    assert all(
        isinstance(window, BacktestWindow)
        for window in windows
    )

    assert windows[0].training_start_date == (
        pd.Timestamp("2026-01-01")
    )
    assert windows[0].training_end_date == (
        pd.Timestamp("2026-01-10")
    )
    assert windows[0].test_start_date == (
        pd.Timestamp("2026-01-11")
    )
    assert windows[0].test_end_date == (
        pd.Timestamp("2026-01-13")
    )

    assert windows[1].training_end_date == (
        pd.Timestamp("2026-01-13")
    )
    assert windows[1].test_start_date == (
        pd.Timestamp("2026-01-14")
    )
    assert windows[1].test_end_date == (
        pd.Timestamp("2026-01-16")
    )

    assert windows[2].training_end_date == (
        pd.Timestamp("2026-01-17")
    )
    assert windows[2].test_start_date == (
        pd.Timestamp("2026-01-18")
    )
    assert windows[2].test_end_date == (
        pd.Timestamp("2026-01-20")
    )


def test_single_backtest_window_uses_latest_period(
    constant_sales: pd.DataFrame,
) -> None:
    """One fold should use the latest test period."""

    windows = create_backtest_windows(
        sales=constant_sales,
        horizon_days=3,
        number_of_folds=1,
        minimum_training_days=10,
    )

    assert len(windows) == 1
    assert windows[0].training_end_date == (
        pd.Timestamp("2026-01-17")
    )
    assert windows[0].test_start_date == (
        pd.Timestamp("2026-01-18")
    )
    assert windows[0].test_end_date == (
        pd.Timestamp("2026-01-20")
    )


def test_create_windows_rejects_date_gaps(
    constant_sales: pd.DataFrame,
) -> None:
    """Backtest history should contain continuous dates."""

    sales_with_gap = constant_sales.loc[
        constant_sales["date"]
        != pd.Timestamp("2026-01-05")
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="continuous",
    ):
        create_backtest_windows(
            sales=sales_with_gap,
            horizon_days=3,
            number_of_folds=2,
            minimum_training_days=10,
        )


def test_run_forecast_backtest_with_constant_sales(
    constant_sales: pd.DataFrame,
) -> None:
    """Constant demand should produce zero baseline error."""

    results = run_forecast_backtest(
        sales=constant_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=3,
        number_of_folds=3,
        minimum_training_days=10,
    )

    assert tuple(results.columns) == (
        BACKTEST_RESULT_COLUMNS
    )

    assert len(results) == 3
    assert results["fold"].tolist() == [1, 2, 3]
    assert results["training_days"].tolist() == [
        10,
        13,
        17,
    ]

    assert results["observations"].eq(3).all()
    assert results["total_actual"].eq(30).all()
    assert results["total_predicted"].eq(30).all()
    assert results["mae"].eq(0).all()
    assert results["rmse"].eq(0).all()
    assert results["wape"].eq(0).all()
    assert results["bias"].eq(0).all()


def test_summarize_backtest_results() -> None:
    """Summary should average metrics and rank methods."""

    results = create_summary_input()

    summary = summarize_backtest_results(
        results
    )

    moving_row = summary.loc[
        summary["method"] == MOVING_AVERAGE_METHOD
    ].iloc[0]

    historical_row = summary.loc[
        summary["method"]
        == HISTORICAL_AVERAGE_METHOD
    ].iloc[0]

    assert len(summary) == 2

    assert moving_row["rank"] == 1
    assert moving_row["folds"] == 2
    assert moving_row["mean_mae"] == 1
    assert moving_row["std_mae"] == 0
    assert moving_row["mean_rmse"] == 2
    assert moving_row["mean_wape"] == 0.1

    assert historical_row["rank"] == 2
    assert historical_row["folds"] == 2
    assert historical_row["mean_mae"] == 3
    assert historical_row["std_mae"] == 1
    assert historical_row["mean_rmse"] == 4
    assert historical_row["mean_wape"] == 0.3
    assert historical_row["mean_bias"] == 0


def test_compare_methods_with_backtesting(
    constant_sales: pd.DataFrame,
) -> None:
    """All method-horizon combinations should be evaluated."""

    details, summary = (
        compare_methods_with_backtesting(
            sales=constant_sales,
            horizons=(1, 3),
            methods=(
                HISTORICAL_AVERAGE_METHOD,
                MOVING_AVERAGE_METHOD,
            ),
            number_of_folds=2,
            minimum_training_days=10,
            moving_average_window_days=3,
        )
    )

    assert len(details) == 8
    assert len(summary) == 4

    assert set(details["horizon_days"]) == {
        1,
        3,
    }

    assert set(details["method"]) == {
        HISTORICAL_AVERAGE_METHOD,
        MOVING_AVERAGE_METHOD,
    }

    fold_counts = details.groupby(
        [
            "horizon_days",
            "method",
        ]
    )["fold"].nunique()

    assert fold_counts.eq(2).all()


def test_compare_rejects_empty_horizons(
    constant_sales: pd.DataFrame,
) -> None:
    """At least one horizon should be provided."""

    with pytest.raises(
        ValueError,
        match="horizons must not be empty",
    ):
        compare_methods_with_backtesting(
            sales=constant_sales,
            horizons=(),
        )


def test_compare_rejects_empty_methods(
    constant_sales: pd.DataFrame,
) -> None:
    """At least one method should be provided."""

    with pytest.raises(
        ValueError,
        match="methods must not be empty",
    ):
        compare_methods_with_backtesting(
            sales=constant_sales,
            horizons=(3,),
            methods=(),
        )


def test_summary_rejects_empty_results() -> None:
    """Empty backtest results should be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        summarize_backtest_results(
            pd.DataFrame()
        )


def test_save_backtest_outputs(
    tmp_path: Path,
) -> None:
    """Detailed and summarized backtests should be saved."""

    backtest_results = create_summary_input()

    backtest_summary = summarize_backtest_results(
        backtest_results
    )

    results_file = (
        tmp_path
        / "backtests"
        / "details.csv"
    )

    summary_file = (
        tmp_path
        / "backtests"
        / "summary.csv"
    )

    (
        saved_results_path,
        saved_summary_path,
    ) = save_backtest_outputs(
        backtest_results=backtest_results,
        backtest_summary=backtest_summary,
        results_output_path=results_file,
        summary_output_path=summary_file,
    )

    assert saved_results_path == (
        results_file.resolve()
    )
    assert saved_summary_path == (
        summary_file.resolve()
    )

    assert saved_results_path.exists()
    assert saved_summary_path.exists()

    loaded_results = pd.read_csv(
        saved_results_path
    )

    loaded_summary = pd.read_csv(
        saved_summary_path
    )

    assert tuple(loaded_results.columns) == (
        BACKTEST_RESULT_COLUMNS
    )

    assert len(loaded_results) == (
        len(backtest_results)
    )

    assert len(loaded_summary) == (
        len(backtest_summary)
    )