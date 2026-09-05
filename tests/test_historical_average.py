"""Tests for historical-average demand forecasting."""

import pandas as pd
import pytest

from src.config import (
    HISTORICAL_AVERAGE_METHOD,
    MAX_FORECAST_HORIZON_DAYS,
)
from src.forecasting.historical_average import (
    calculate_historical_average,
    create_horizon_summary,
    forecast_historical_average,
    save_daily_forecast,
    validate_forecast_horizon,
    validate_sales_for_forecasting,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create three days of sales for two products."""

    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 10,
            },
            {
                "date": "2026-01-02",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 20,
            },
            {
                "date": "2026-01-03",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 0,
            },
            {
                "date": "2026-01-01",
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 5,
            },
            {
                "date": "2026-01-02",
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 5,
            },
            {
                "date": "2026-01-03",
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 5,
            },
        ]
    )


def test_calculate_historical_average(
    sample_sales: pd.DataFrame,
) -> None:
    """Historical averages should use the full date range."""

    result = calculate_historical_average(
        sales=sample_sales
    )

    result_by_product = result.set_index(
        "product_id"
    )

    assert (
        result_by_product.loc[
            "P001",
            "history_days",
        ]
        == 3
    )

    assert (
        result_by_product.loc[
            "P001",
            "total_quantity_sold",
        ]
        == 30
    )

    assert (
        result_by_product.loc[
            "P001",
            "average_daily_demand",
        ]
        == 10.0
    )

    assert (
        result_by_product.loc[
            "P002",
            "average_daily_demand",
        ]
        == 5.0
    )


def test_forecast_historical_average(
    sample_sales: pd.DataFrame,
) -> None:
    """Forecasts should contain one row per pair and day."""

    forecast = forecast_historical_average(
        sales=sample_sales,
        horizon_days=3,
    )

    assert len(forecast) == 6

    assert forecast["forecast_day"].unique().tolist() == [
        1,
        2,
        3,
    ]

    assert forecast["method"].unique().tolist() == [
        HISTORICAL_AVERAGE_METHOD
    ]

    product_one = forecast.loc[
        forecast["product_id"] == "P001"
    ]

    assert product_one[
        "predicted_quantity"
    ].tolist() == [
        10.0,
        10.0,
        10.0,
    ]

    assert product_one[
        "forecast_date"
    ].tolist() == [
        pd.Timestamp("2026-01-04"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
    ]


def test_forecast_supports_custom_horizon(
    sample_sales: pd.DataFrame,
) -> None:
    """Forecasting should support any allowed horizon."""

    forecast = forecast_historical_average(
        sales=sample_sales,
        horizon_days=9,
    )

    assert len(forecast) == 18
    assert forecast["forecast_day"].min() == 1
    assert forecast["forecast_day"].max() == 9

    product_one_total = forecast.loc[
        forecast["product_id"] == "P001",
        "predicted_quantity",
    ].sum()

    assert product_one_total == 90.0


def test_create_horizon_summary(
    sample_sales: pd.DataFrame,
) -> None:
    """Daily predictions should be aggregated by pair."""

    forecast = forecast_historical_average(
        sales=sample_sales,
        horizon_days=3,
    )

    summary = create_horizon_summary(
        daily_forecast=forecast
    )

    summary_by_product = summary.set_index(
        "product_id"
    )

    assert (
        summary_by_product.loc[
            "P001",
            "horizon_days",
        ]
        == 3
    )

    assert (
        summary_by_product.loc[
            "P001",
            "predicted_horizon_demand",
        ]
        == 30.0
    )

    assert (
        summary_by_product.loc[
            "P001",
            "predicted_daily_average",
        ]
        == 10.0
    )

    assert (
        summary_by_product.loc[
            "P002",
            "predicted_horizon_demand",
        ]
        == 15.0
    )


@pytest.mark.parametrize(
    "horizon_days",
    [
        0,
        MAX_FORECAST_HORIZON_DAYS + 1,
    ],
)
def test_validate_forecast_horizon_rejects_invalid_range(
    horizon_days: int,
) -> None:
    """Forecast horizon must remain inside its limits."""

    with pytest.raises(
        ValueError,
        match="horizon_days must be between",
    ):
        validate_forecast_horizon(
            horizon_days=horizon_days
        )


def test_validate_forecast_horizon_rejects_non_integer() -> None:
    """Forecast horizon must be an integer."""

    with pytest.raises(
        TypeError,
        match="horizon_days must be an integer",
    ):
        validate_forecast_horizon(
            horizon_days=7.5
        )


def test_sales_validation_rejects_missing_column(
    sample_sales: pd.DataFrame,
) -> None:
    """Required sales columns must exist."""

    invalid_sales = sample_sales.drop(
        columns="quantity_sold"
    )

    with pytest.raises(
        ValueError,
        match="sales is missing columns",
    ):
        validate_sales_for_forecasting(
            sales=invalid_sales
        )


def test_sales_validation_rejects_duplicates(
    sample_sales: pd.DataFrame,
) -> None:
    """Duplicate date-store-product rows are invalid."""

    duplicated_sales = pd.concat(
        [
            sample_sales,
            sample_sales.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="sales contains duplicate",
    ):
        validate_sales_for_forecasting(
            sales=duplicated_sales
        )


def test_sales_validation_rejects_negative_quantity(
    sample_sales: pd.DataFrame,
) -> None:
    """Sales quantities must not be negative."""

    invalid_sales = sample_sales.copy()

    invalid_sales.loc[
        0,
        "quantity_sold",
    ] = -1

    with pytest.raises(
        ValueError,
        match="must not contain negative values",
    ):
        validate_sales_for_forecasting(
            sales=invalid_sales
        )


def test_save_daily_forecast(
    sample_sales: pd.DataFrame,
    tmp_path,
) -> None:
    """Daily forecasts should be saved as CSV."""

    forecast = forecast_historical_average(
        sales=sample_sales,
        horizon_days=3,
    )

    output_path = (
        tmp_path / "daily_demand_forecast.csv"
    )

    saved_path = save_daily_forecast(
        daily_forecast=forecast,
        output_path=output_path,
    )

    assert saved_path.exists()

    loaded_forecast = pd.read_csv(
        saved_path
    )

    assert len(loaded_forecast) == 6

    assert loaded_forecast[
        "method"
    ].unique().tolist() == [
        HISTORICAL_AVERAGE_METHOD
    ]