"""Tests for the demand forecasting coordinator."""

import pandas as pd
import pytest

from src.config import (
    HISTORICAL_AVERAGE_METHOD,
    LONG_TERM_REPLENISHMENT_DAYS,
    MOVING_AVERAGE_METHOD,
    SHORT_TERM_REPLENISHMENT_DAYS,
)

from src.forecasting.demand_forecaster import (
    forecast_demand,
    forecast_demand_for_optimization,
    get_replenishment_horizon,
    validate_forecast_method,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create simple sales data for forecasting tests."""

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []

    for index, date in enumerate(dates, start=1):
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


@pytest.mark.parametrize(
    "method",
    [
        HISTORICAL_AVERAGE_METHOD,
        MOVING_AVERAGE_METHOD,
    ],
)
def test_validate_forecast_method_accepts_supported_methods(
    method: str,
) -> None:
    """Supported methods should be accepted."""

    result = validate_forecast_method(method)

    assert result == method


def test_validate_forecast_method_normalizes_input() -> None:
    """Method names should be stripped and converted to lowercase."""

    result = validate_forecast_method("  MOVING_AVERAGE  ")

    assert result == MOVING_AVERAGE_METHOD


def test_validate_forecast_method_rejects_unsupported_method() -> None:
    """Unsupported methods should raise ValueError."""

    with pytest.raises(ValueError, match="Unsupported forecast method"):
        validate_forecast_method("unknown_model")


def test_validate_forecast_method_rejects_non_string() -> None:
    """Non-string method values should raise TypeError."""

    with pytest.raises(TypeError, match="must be a string"):
        validate_forecast_method(123)


def test_forecast_demand_uses_historical_average(
    sample_sales: pd.DataFrame,
) -> None:
    """The coordinator should call historical average correctly."""

    forecast = forecast_demand(
        sales=sample_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=3,
    )

    product_forecast = forecast[
        forecast["product_id"] == "P001"
    ]

    assert len(forecast) == 6
    assert product_forecast["predicted_quantity"].tolist() == [
        pytest.approx(5.5),
        pytest.approx(5.5),
        pytest.approx(5.5),
    ]
    assert set(forecast["method"]) == {
        HISTORICAL_AVERAGE_METHOD
    }


def test_forecast_demand_uses_moving_average(
    sample_sales: pd.DataFrame,
) -> None:
    """The coordinator should call moving average correctly."""

    forecast = forecast_demand(
        sales=sample_sales,
        method=MOVING_AVERAGE_METHOD,
        horizon_days=3,
        moving_average_window_days=3,
    )

    product_forecast = forecast[
        forecast["product_id"] == "P001"
    ]

    assert len(forecast) == 6
    assert product_forecast["predicted_quantity"].tolist() == [
        pytest.approx(9.0),
        pytest.approx(9.0),
        pytest.approx(9.0),
    ]
    assert set(forecast["method"]) == {
        MOVING_AVERAGE_METHOD
    }


def test_forecast_demand_supports_custom_horizon(
    sample_sales: pd.DataFrame,
) -> None:
    """A custom horizon such as nine days should be supported."""

    forecast = forecast_demand(
        sales=sample_sales,
        method=HISTORICAL_AVERAGE_METHOD,
        horizon_days=9,
    )

    assert len(forecast) == 18
    assert forecast["forecast_day"].min() == 1
    assert forecast["forecast_day"].max() == 9
    assert forecast["forecast_date"].nunique() == 9


@pytest.mark.parametrize(
    (
        "requested_horizon_days",
        "expected_replenishment_days",
    ),
    [
        (1, 7),
        (5, 7),
        (7, 7),
        (8, 14),
        (9, 14),
        (14, 14),
    ],
)
def test_get_replenishment_horizon(
    requested_horizon_days: int,
    expected_replenishment_days: int,
) -> None:
    """Requested horizons should map to replenishment tiers."""

    result = get_replenishment_horizon(
        requested_horizon_days
    )

    assert result == expected_replenishment_days


@pytest.mark.parametrize(
    "requested_horizon_days",
    [
        0,
        15,
    ],
)
def test_get_replenishment_horizon_rejects_invalid_values(
    requested_horizon_days: int,
) -> None:
    """Horizons outside the supported range should be rejected."""

    with pytest.raises(ValueError):
        get_replenishment_horizon(
            requested_horizon_days
        )


@pytest.mark.parametrize(
    (
        "requested_horizon_days",
        "expected_replenishment_days",
    ),
    [
        (
            5,
            SHORT_TERM_REPLENISHMENT_DAYS,
        ),
        (
            9,
            LONG_TERM_REPLENISHMENT_DAYS,
        ),
    ],
)
def test_optimization_forecast_uses_replenishment_horizon(
    sample_sales: pd.DataFrame,
    requested_horizon_days: int,
    expected_replenishment_days: int,
) -> None:
    """Optimization should forecast the mapped replenishment horizon."""

    forecast = forecast_demand_for_optimization(
        sales=sample_sales,
        requested_horizon_days=(
            requested_horizon_days
        ),
        method=MOVING_AVERAGE_METHOD,
        moving_average_window_days=3,
    )

    number_of_pairs = 2

    assert len(forecast) == (
        number_of_pairs
        * expected_replenishment_days
    )

    assert forecast["forecast_day"].min() == 1

    assert forecast["forecast_day"].max() == (
        expected_replenishment_days
    )


def test_forecast_demand_rejects_invalid_horizon(
    sample_sales: pd.DataFrame,
) -> None:
    """Invalid forecast horizons should be rejected."""

    with pytest.raises(ValueError):
        forecast_demand(
            sales=sample_sales,
            method=HISTORICAL_AVERAGE_METHOD,
            horizon_days=0,
        )