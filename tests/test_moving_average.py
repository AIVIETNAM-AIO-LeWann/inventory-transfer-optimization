"""Tests for moving-average demand forecasting."""

import pandas as pd
import pytest

from src.config import MOVING_AVERAGE_METHOD
from src.forecasting.historical_average import (
    calculate_historical_average,
)
from src.forecasting.moving_average import (
    calculate_moving_average,
    forecast_moving_average,
    validate_moving_average_window,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create ten days of sales for two products."""

    product_one_quantities = [
        1,
        1,
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    records = []

    for day_index, quantity in enumerate(
        product_one_quantities
    ):
        date = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=day_index)
        )

        records.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": quantity,
            }
        )

        records.append(
            {
                "date": date,
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 2,
            }
        )

    return pd.DataFrame(records)


def test_calculate_moving_average_uses_recent_window(
    sample_sales: pd.DataFrame,
) -> None:
    """Moving average should use only the latest days."""

    result = calculate_moving_average(
        sales=sample_sales,
        window_days=7,
    )

    result_by_product = result.set_index(
        "product_id"
    )

    product_one = result_by_product.loc["P001"]

    assert product_one["window_days"] == 7

    assert product_one["window_start_date"] == (
        pd.Timestamp("2026-01-04")
    )

    assert product_one["window_end_date"] == (
        pd.Timestamp("2026-01-10")
    )

    assert product_one["total_window_quantity"] == 49

    assert (
        product_one[
            "moving_average_daily_demand"
        ]
        == 7.0
    )

    assert (
        result_by_product.loc[
            "P002",
            "moving_average_daily_demand",
        ]
        == 2.0
    )


def test_forecast_moving_average(
    sample_sales: pd.DataFrame,
) -> None:
    """Moving average should create future predictions."""

    forecast = forecast_moving_average(
        sales=sample_sales,
        horizon_days=3,
        window_days=7,
    )

    assert len(forecast) == 6

    assert forecast["method"].unique().tolist() == [
        MOVING_AVERAGE_METHOD
    ]

    product_one = forecast.loc[
        forecast["product_id"] == "P001"
    ]

    assert product_one[
        "predicted_quantity"
    ].tolist() == [
        7.0,
        7.0,
        7.0,
    ]

    assert product_one[
        "forecast_date"
    ].tolist() == [
        pd.Timestamp("2026-01-11"),
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-13"),
    ]


def test_moving_average_supports_custom_window(
    sample_sales: pd.DataFrame,
) -> None:
    """A custom window should change the recent average."""

    result = calculate_moving_average(
        sales=sample_sales,
        window_days=3,
    )

    product_one = result.loc[
        result["product_id"] == "P001"
    ].iloc[0]

    assert product_one["window_start_date"] == (
        pd.Timestamp("2026-01-08")
    )

    assert product_one["total_window_quantity"] == 27

    assert (
        product_one[
            "moving_average_daily_demand"
        ]
        == 9.0
    )


def test_pair_without_recent_sales_receives_zero_average() -> None:
    """A pair with no recent rows should receive zero demand."""

    sales = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 10,
            },
            {
                "date": "2026-01-10",
                "store_id": "S001",
                "product_id": "P002",
                "quantity_sold": 6,
            },
        ]
    )

    result = calculate_moving_average(
        sales=sales,
        window_days=3,
    )

    result_by_product = result.set_index(
        "product_id"
    )

    assert (
        result_by_product.loc[
            "P001",
            "total_window_quantity",
        ]
        == 0
    )

    assert (
        result_by_product.loc[
            "P001",
            "moving_average_daily_demand",
        ]
        == 0.0
    )

    assert (
        result_by_product.loc[
            "P002",
            "moving_average_daily_demand",
        ]
        == 2.0
    )


def test_moving_average_differs_from_historical_average(
    sample_sales: pd.DataFrame,
) -> None:
    """Recent growth should make moving average higher."""

    historical = calculate_historical_average(
        sales=sample_sales
    ).set_index("product_id")

    moving = calculate_moving_average(
        sales=sample_sales,
        window_days=7,
    ).set_index("product_id")

    historical_demand = historical.loc[
        "P001",
        "average_daily_demand",
    ]

    moving_demand = moving.loc[
        "P001",
        "moving_average_daily_demand",
    ]

    assert historical_demand == 5.2
    assert moving_demand == 7.0
    assert moving_demand > historical_demand


@pytest.mark.parametrize(
    (
        "window_days",
        "history_days",
        "expected_error",
        "expected_message",
    ),
    [
        (
            0,
            10,
            ValueError,
            "greater than zero",
        ),
        (
            11,
            10,
            ValueError,
            "available sales history",
        ),
        (
            3.5,
            10,
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_invalid_moving_average_window(
    window_days,
    history_days: int,
    expected_error,
    expected_message: str,
) -> None:
    """Invalid moving-average windows should be rejected."""

    with pytest.raises(
        expected_error,
        match=expected_message,
    ):
        validate_moving_average_window(
            window_days=window_days,
            history_days=history_days,
        )
        