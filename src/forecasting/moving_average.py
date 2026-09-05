"""Forecast demand using a moving average."""

import pandas as pd

from src.config import (
    MAX_FORECAST_HORIZON_DAYS,
    MOVING_AVERAGE_METHOD,
    MOVING_AVERAGE_WINDOW_DAYS,
)
from src.data_loader import load_all_data
from src.forecasting.historical_average import (
    FORECAST_COLUMNS,
    create_horizon_summary,
    save_daily_forecast,
    validate_forecast_horizon,
    validate_sales_for_forecasting,
)


def validate_moving_average_window(
    window_days: int,
    history_days: int,
) -> None:
    """Validate the moving-average window."""

    if not isinstance(window_days, int):
        raise TypeError(
            "window_days must be an integer."
        )

    if window_days <= 0:
        raise ValueError(
            "window_days must be greater than zero."
        )

    if window_days > history_days:
        raise ValueError(
            "window_days must not be greater than "
            "the available sales history."
        )


def calculate_moving_average(
    sales: pd.DataFrame,
    window_days: int = MOVING_AVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Calculate recent average demand for each pair."""

    validate_sales_for_forecasting(sales)

    working_sales = sales[
        [
            "date",
            "store_id",
            "product_id",
            "quantity_sold",
        ]
    ].copy()

    working_sales["date"] = pd.to_datetime(
        working_sales["date"]
    )

    working_sales["store_id"] = (
        working_sales["store_id"].astype(str)
    )

    working_sales["product_id"] = (
        working_sales["product_id"].astype(str)
    )

    history_start_date = (
        working_sales["date"].min().normalize()
    )

    history_end_date = (
        working_sales["date"].max().normalize()
    )

    history_days = (
        history_end_date - history_start_date
    ).days + 1

    validate_moving_average_window(
        window_days=window_days,
        history_days=history_days,
    )

    window_start_date = (
        history_end_date
        - pd.Timedelta(
            days=window_days - 1
        )
    )

    window_sales = working_sales.loc[
        working_sales["date"]
        >= window_start_date
    ].copy()

    store_product_pairs = (
        working_sales[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    window_totals = (
        window_sales.groupby(
            [
                "store_id",
                "product_id",
            ],
            as_index=False,
        )
        .agg(
            total_window_quantity=(
                "quantity_sold",
                "sum",
            )
        )
    )

    moving_average = store_product_pairs.merge(
        window_totals,
        on=[
            "store_id",
            "product_id",
        ],
        how="left",
        validate="one_to_one",
    )

    moving_average[
        "total_window_quantity"
    ] = moving_average[
        "total_window_quantity"
    ].fillna(0)

    moving_average["window_start_date"] = (
        window_start_date
    )

    moving_average["window_end_date"] = (
        history_end_date
    )

    moving_average["window_days"] = window_days

    moving_average[
        "moving_average_daily_demand"
    ] = (
        moving_average["total_window_quantity"]
        / window_days
    ).round(4)

    return moving_average[
        [
            "store_id",
            "product_id",
            "window_start_date",
            "window_end_date",
            "window_days",
            "total_window_quantity",
            "moving_average_daily_demand",
        ]
    ]


def forecast_moving_average(
    sales: pd.DataFrame,
    horizon_days: int,
    window_days: int = MOVING_AVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Forecast daily demand using a moving average."""

    validate_forecast_horizon(horizon_days)

    moving_average = calculate_moving_average(
        sales=sales,
        window_days=window_days,
    )

    last_sales_date = pd.to_datetime(
        sales["date"]
    ).max().normalize()

    future_days = pd.DataFrame(
        {
            "forecast_day": range(
                1,
                horizon_days + 1,
            )
        }
    )

    forecast = moving_average.merge(
        future_days,
        how="cross",
    )

    forecast["forecast_date"] = (
        last_sales_date
        + pd.to_timedelta(
            forecast["forecast_day"],
            unit="D",
        )
    )

    forecast["predicted_quantity"] = (
        forecast[
            "moving_average_daily_demand"
        ]
    )

    forecast["method"] = MOVING_AVERAGE_METHOD

    forecast = forecast.sort_values(
        by=[
            "store_id",
            "product_id",
            "forecast_day",
        ],
        ignore_index=True,
    )

    return forecast[list(FORECAST_COLUMNS)]


def main() -> None:
    """Run moving-average demand forecasting."""

    project_data = load_all_data()

    daily_forecast = forecast_moving_average(
        sales=project_data.sales,
        horizon_days=MAX_FORECAST_HORIZON_DAYS,
        window_days=MOVING_AVERAGE_WINDOW_DAYS,
    )

    summary = create_horizon_summary(
        daily_forecast=daily_forecast
    )

    output_path = save_daily_forecast(
        daily_forecast=daily_forecast
    )

    print(
        "Moving-average forecasting "
        "completed successfully."
    )
    print(f"Forecast rows: {len(daily_forecast)}")
    print(
        "Moving-average window: "
        f"{MOVING_AVERAGE_WINDOW_DAYS} days"
    )
    print(f"Saved to: {output_path}")
    print()
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()