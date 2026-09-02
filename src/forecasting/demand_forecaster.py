"""Coordinate demand forecasting methods."""

import pandas as pd

from src.config import (
    DAILY_DEMAND_FORECAST_FILE,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_FORECAST_METHOD,
    HISTORICAL_AVERAGE_METHOD,
    MOVING_AVERAGE_METHOD,
    MOVING_AVERAGE_WINDOW_DAYS,
    SHORT_TERM_REPLENISHMENT_DAYS,
    LONG_TERM_REPLENISHMENT_DAYS,
    SALES_FILE,
    SUPPORTED_FORECAST_METHODS,
)
from src.forecasting.historical_average import (
    create_horizon_summary,
    forecast_historical_average,
    save_daily_forecast,
    validate_forecast_horizon,
)
from src.forecasting.moving_average import forecast_moving_average


def validate_forecast_method(method: str) -> str:
    """Validate and normalize the selected forecast method."""

    if not isinstance(method, str):
        raise TypeError("Forecast method must be a string.")

    normalized_method = method.strip().lower()

    if normalized_method not in SUPPORTED_FORECAST_METHODS:
        supported_methods = ", ".join(SUPPORTED_FORECAST_METHODS)

        raise ValueError(
            f"Unsupported forecast method: {method}. "
            f"Supported methods: {supported_methods}."
        )

    return normalized_method

def get_replenishment_horizon(
    requested_horizon_days: int,
) -> int:
    """Convert a requested horizon into a replenishment horizon."""

    validate_forecast_horizon(
        requested_horizon_days
    )

    if (
        requested_horizon_days
        <= SHORT_TERM_REPLENISHMENT_DAYS
    ):
        return SHORT_TERM_REPLENISHMENT_DAYS

    return LONG_TERM_REPLENISHMENT_DAYS

def forecast_demand(
    sales: pd.DataFrame,
    method: str = DEFAULT_FORECAST_METHOD,
    horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    moving_average_window_days: int = MOVING_AVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Generate daily demand forecasts using the selected method."""

    selected_method = validate_forecast_method(method)

    if selected_method == HISTORICAL_AVERAGE_METHOD:
        return forecast_historical_average(
            sales=sales,
            horizon_days=horizon_days,
        )

    if selected_method == MOVING_AVERAGE_METHOD:
        return forecast_moving_average(
            sales=sales,
            horizon_days=horizon_days,
            window_days=moving_average_window_days,
        )

    raise RuntimeError(
        f"Forecast method has no implementation: {selected_method}"
    )


def forecast_demand_for_optimization(
    sales: pd.DataFrame,
    requested_horizon_days: int,
    method: str = DEFAULT_FORECAST_METHOD,
    moving_average_window_days: int = MOVING_AVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Forecast demand for the appropriate replenishment horizon."""

    replenishment_horizon_days = (
        get_replenishment_horizon(
            requested_horizon_days
        )
    )

    return forecast_demand(
        sales=sales,
        method=method,
        horizon_days=replenishment_horizon_days,
        moving_average_window_days=(
            moving_average_window_days
        ),
    )


def main() -> None:
    """Run a demand forecasting example from the command line."""

    sales = pd.read_csv(SALES_FILE)

    daily_forecast = forecast_demand(
        sales=sales,
        method=DEFAULT_FORECAST_METHOD,
        horizon_days=DEFAULT_FORECAST_HORIZON_DAYS,
        moving_average_window_days=MOVING_AVERAGE_WINDOW_DAYS,
    )

    horizon_summary = create_horizon_summary(daily_forecast)

    save_daily_forecast(
        daily_forecast,
        DAILY_DEMAND_FORECAST_FILE,
    )

    print("Demand forecasting completed.")
    print(f"Method: {DEFAULT_FORECAST_METHOD}")
    print(f"Horizon: {DEFAULT_FORECAST_HORIZON_DAYS} days")
    print(f"Daily forecast rows: {len(daily_forecast):,}")
    print(f"Output file: {DAILY_DEMAND_FORECAST_FILE}")

    print("\nForecast summary:")
    print(horizon_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()