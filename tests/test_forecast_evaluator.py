"""Tests for forecast evaluation."""

import pandas as pd
import pytest

from src.config import (
    HISTORICAL_AVERAGE_METHOD,
    MOVING_AVERAGE_METHOD,
)
from src.forecasting.demand_forecaster import (
    forecast_demand,
)
from src.forecasting.forecast_evaluator import (
    EVALUATION_COLUMNS,
    ForecastEvaluationResult,
    align_forecast_with_actuals,
    calculate_forecast_metrics,
    compare_forecast_methods,
    evaluate_forecast_method,
    split_sales_by_time,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create ten days of sales for two products."""

    dates = pd.date_range(
        "2026-01-01",
        periods=10,
        freq="D",
    )

    rows = []

    for index, date in enumerate(
        dates,
        start=1,
    ):
        rows.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": index,
            }
        )

        rows.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 2,
            }
        )

    return pd.DataFrame(rows)


def create_metric_comparison() -> pd.DataFrame:
    """Create simple actual and predicted values."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "date": pd.Timestamp("2026-01-08"),
                "forecast_day": 1,
                "method": HISTORICAL_AVERAGE_METHOD,
                "actual_quantity": 10,
                "predicted_quantity": 12,
                "error": 2,
                "absolute_error": 2,
                "squared_error": 4,
            },
            {
                "store_id": "S001",
                "product_id": "P001",
                "date": pd.Timestamp("2026-01-09"),
                "forecast_day": 2,
                "method": HISTORICAL_AVERAGE_METHOD,
                "actual_quantity": 20,
                "predicted_quantity": 16,
                "error": -4,
                "absolute_error": 4,
                "squared_error": 16,
            },
        ],
        columns=EVALUATION_COLUMNS,
    )


def test_split_sales_by_time(
    sample_sales: pd.DataFrame,
) -> None:
    """The latest dates should form the test period."""

    training_sales, test_sales = split_sales_by_time(
        sales=sample_sales,
        test_days=3,
    )

    assert len(training_sales) == 14
    assert len(test_sales) == 6

    assert training_sales["date"].min() == (
        pd.Timestamp("2026-01-01")
    )
    assert training_sales["date"].max() == (
        pd.Timestamp("2026-01-07")
    )

    assert test_sales["date"].min() == (
        pd.Timestamp("2026-01-08")
    )
    assert test_sales["date"].max() == (
        pd.Timestamp("2026-01-10")
    )


def test_split_rejects_insufficient_history(
    sample_sales: pd.DataFrame,
) -> None:
    """Training data must remain after the split."""

    with pytest.raises(
        ValueError,
        match="more dates",
    ):
        split_sales_by_time(
            sales=sample_sales,
            test_days=10,
        )


def test_split_rejects_date_gaps(
    sample_sales: pd.DataFrame,
) -> None:
    """Sales history should contain continuous dates."""

    sales_with_gap = sample_sales.loc[
        sample_sales["date"]
        != pd.Timestamp("2026-01-05")
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="continuous",
    ):
        split_sales_by_time(
            sales=sales_with_gap,
            test_days=3,
        )


def test_align_forecast_with_actuals(
    sample_sales: pd.DataFrame,
) -> None:
    """Forecasts should align with the holdout period."""

    training_sales, test_sales = split_sales_by_time(
        sales=sample_sales,
        test_days=3,
    )

    daily_forecast = forecast_demand(
        sales=training_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=3,
    )

    comparison = align_forecast_with_actuals(
        daily_forecast=daily_forecast,
        test_sales=test_sales,
    )

    product_comparison = comparison.loc[
        comparison["product_id"] == "P001"
    ]

    assert len(comparison) == 6

    assert product_comparison[
        "actual_quantity"
    ].tolist() == [8, 9, 10]

    assert product_comparison[
        "predicted_quantity"
    ].tolist() == [
        pytest.approx(4),
        pytest.approx(4),
        pytest.approx(4),
    ]

    assert product_comparison[
        "error"
    ].tolist() == [
        pytest.approx(-4),
        pytest.approx(-5),
        pytest.approx(-6),
    ]


def test_align_rejects_unmatched_rows(
    sample_sales: pd.DataFrame,
) -> None:
    """Every forecast row should have an actual row."""

    training_sales, test_sales = split_sales_by_time(
        sales=sample_sales,
        test_days=3,
    )

    daily_forecast = forecast_demand(
        sales=training_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=3,
    )

    incomplete_test_sales = test_sales.drop(
        index=test_sales.index[0]
    )

    with pytest.raises(
        ValueError,
        match="could not be aligned",
    ):
        align_forecast_with_actuals(
            daily_forecast=daily_forecast,
            test_sales=incomplete_test_sales,
        )


def test_calculate_forecast_metrics() -> None:
    """Forecast metrics should match manual calculations."""

    comparison = create_metric_comparison()

    metrics = calculate_forecast_metrics(
        comparison
    )

    assert metrics["method"] == (
        HISTORICAL_AVERAGE_METHOD
    )
    assert metrics["horizon_days"] == 2
    assert metrics["observations"] == 2
    assert metrics["total_actual"] == 30
    assert metrics["total_predicted"] == 28
    assert metrics["mae"] == pytest.approx(3)
    assert metrics["rmse"] == pytest.approx(3.1623)
    assert metrics["wape"] == pytest.approx(0.2)
    assert metrics["bias"] == pytest.approx(-1)


def test_metrics_reject_multiple_methods() -> None:
    """One comparison should contain only one method."""

    comparison = create_metric_comparison()

    comparison.loc[
        1,
        "method",
    ] = MOVING_AVERAGE_METHOD

    with pytest.raises(
        ValueError,
        match="exactly one forecast method",
    ):
        calculate_forecast_metrics(
            comparison
        )


def test_evaluate_historical_average(
    sample_sales: pd.DataFrame,
) -> None:
    """Historical average should be evaluated end to end."""

    result = evaluate_forecast_method(
        sales=sample_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=3,
    )

    assert isinstance(
        result,
        ForecastEvaluationResult,
    )

    assert result.method == (
        HISTORICAL_AVERAGE_METHOD
    )
    assert result.horizon_days == 3
    assert len(result.training_sales) == 14
    assert len(result.test_sales) == 6
    assert len(result.daily_forecast) == 6
    assert len(result.comparison) == 6

    assert result.metrics["total_actual"] == 33
    assert result.metrics["total_predicted"] == 18
    assert result.metrics["mae"] == pytest.approx(2.5)
    assert result.metrics["rmse"] == pytest.approx(3.5824)
    assert result.metrics["wape"] == pytest.approx(0.4545)
    assert result.metrics["bias"] == pytest.approx(-2.5)


def test_compare_forecast_methods_ranks_best_method(
    sample_sales: pd.DataFrame,
) -> None:
    """The method with lower error should rank first."""

    comparison = compare_forecast_methods(
        sales=sample_sales,
        horizon_days=3,
        methods=(
            HISTORICAL_AVERAGE_METHOD,
            MOVING_AVERAGE_METHOD,
        ),
        moving_average_window_days=3,
    )

    assert len(comparison) == 2
    assert comparison.iloc[0]["rank"] == 1
    assert comparison.iloc[0]["method"] == (
        MOVING_AVERAGE_METHOD
    )
    assert comparison.iloc[1]["method"] == (
        HISTORICAL_AVERAGE_METHOD
    )

    assert comparison.iloc[0]["mae"] < (
        comparison.iloc[1]["mae"]
    )


def test_compare_forecast_methods_rejects_empty_methods(
    sample_sales: pd.DataFrame,
) -> None:
    """At least one forecast method is required."""

    with pytest.raises(
        ValueError,
        match="methods must not be empty",
    ):
        compare_forecast_methods(
            sales=sample_sales,
            horizon_days=3,
            methods=(),
        )