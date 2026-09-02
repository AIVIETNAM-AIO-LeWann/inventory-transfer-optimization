"""Forecast demand using historical average sales."""

from pathlib import Path

import pandas as pd

from src.config import (
    DAILY_DEMAND_FORECAST_FILE,
    HISTORICAL_AVERAGE_METHOD,
    MAX_FORECAST_HORIZON_DAYS,
    MIN_FORECAST_HORIZON_DAYS,
)
from src.data_loader import load_all_data


REQUIRED_SALES_COLUMNS = (
    "date",
    "store_id",
    "product_id",
    "quantity_sold",
)

FORECAST_COLUMNS = (
    "store_id",
    "product_id",
    "forecast_date",
    "forecast_day",
    "predicted_quantity",
    "method",
)


def validate_forecast_horizon(
    horizon_days: int,
) -> None:
    """Validate the number of future days to forecast."""

    if not isinstance(horizon_days, int):
        raise TypeError(
            "horizon_days must be an integer."
        )

    if not (
        MIN_FORECAST_HORIZON_DAYS
        <= horizon_days
        <= MAX_FORECAST_HORIZON_DAYS
    ):
        raise ValueError(
            "horizon_days must be between "
            f"{MIN_FORECAST_HORIZON_DAYS} and "
            f"{MAX_FORECAST_HORIZON_DAYS}."
        )


def validate_sales_for_forecasting(
    sales: pd.DataFrame,
) -> None:
    """Validate sales data required for forecasting."""

    if not isinstance(sales, pd.DataFrame):
        raise TypeError(
            "sales must be a pandas DataFrame."
        )

    if sales.empty:
        raise ValueError(
            "sales must not be empty."
        )

    missing_columns = (
        set(REQUIRED_SALES_COLUMNS)
        - set(sales.columns)
    )

    if missing_columns:
        raise ValueError(
            "sales is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        sales[list(REQUIRED_SALES_COLUMNS)]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "sales contains missing values."
        )

    converted_dates = pd.to_datetime(
        sales["date"],
        errors="coerce",
    )

    if converted_dates.isna().any():
        raise ValueError(
            "sales.date contains invalid values."
        )

    if not pd.api.types.is_numeric_dtype(
        sales["quantity_sold"]
    ):
        raise TypeError(
            "sales.quantity_sold must be numeric."
        )

    if (sales["quantity_sold"] < 0).any():
        raise ValueError(
            "sales.quantity_sold must not "
            "contain negative values."
        )

    duplicated_rows = sales.duplicated(
        subset=[
            "date",
            "store_id",
            "product_id",
        ],
        keep=False,
    )

    if duplicated_rows.any():
        raise ValueError(
            "sales contains duplicate "
            "date-store-product rows."
        )


def calculate_historical_average(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average daily demand for each pair."""

    validate_sales_for_forecasting(sales)

    working_sales = sales[
        list(REQUIRED_SALES_COLUMNS)
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

    if history_days <= 0:
        raise ValueError(
            "Historical sales period must be "
            "greater than zero."
        )

    historical_average = (
        working_sales.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )
        .agg(
            total_quantity_sold=(
                "quantity_sold",
                "sum",
            )
        )
    )

    historical_average["history_start_date"] = (
        history_start_date
    )

    historical_average["history_end_date"] = (
        history_end_date
    )

    historical_average["history_days"] = (
        history_days
    )

    historical_average["average_daily_demand"] = (
        historical_average["total_quantity_sold"]
        / history_days
    )

    historical_average[
        "average_daily_demand"
    ] = historical_average[
        "average_daily_demand"
    ].round(4)

    return historical_average[
        [
            "store_id",
            "product_id",
            "history_start_date",
            "history_end_date",
            "history_days",
            "total_quantity_sold",
            "average_daily_demand",
        ]
    ]


def forecast_historical_average(
    sales: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Forecast daily demand using historical averages."""

    validate_forecast_horizon(horizon_days)

    historical_average = (
        calculate_historical_average(sales)
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

    forecast = historical_average.merge(
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
        forecast["average_daily_demand"]
    )

    forecast["method"] = (
        HISTORICAL_AVERAGE_METHOD
    )

    forecast = forecast.sort_values(
        by=[
            "store_id",
            "product_id",
            "forecast_day",
        ],
        ignore_index=True,
    )

    return forecast[list(FORECAST_COLUMNS)]


def create_horizon_summary(
    daily_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize total demand over the forecast horizon."""

    if daily_forecast.empty:
        raise ValueError(
            "daily_forecast must not be empty."
        )

    missing_columns = (
        set(FORECAST_COLUMNS)
        - set(daily_forecast.columns)
    )

    if missing_columns:
        raise ValueError(
            "daily_forecast is missing columns: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        daily_forecast.groupby(
            [
                "store_id",
                "product_id",
                "method",
            ],
            as_index=False,
        )
        .agg(
            forecast_start_date=(
                "forecast_date",
                "min",
            ),
            forecast_end_date=(
                "forecast_date",
                "max",
            ),
            horizon_days=(
                "forecast_day",
                "max",
            ),
            predicted_horizon_demand=(
                "predicted_quantity",
                "sum",
            ),
            predicted_daily_average=(
                "predicted_quantity",
                "mean",
            ),
        )
    )

    summary[
        [
            "predicted_horizon_demand",
            "predicted_daily_average",
        ]
    ] = summary[
        [
            "predicted_horizon_demand",
            "predicted_daily_average",
        ]
    ].round(2)

    return summary


def save_daily_forecast(
    daily_forecast: pd.DataFrame,
    output_path: str | Path = (
        DAILY_DEMAND_FORECAST_FILE
    ),
) -> Path:
    """Save daily demand forecasts to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_forecast.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Run historical-average demand forecasting."""

    project_data = load_all_data()

    daily_forecast = forecast_historical_average(
        sales=project_data.sales,
        horizon_days=MAX_FORECAST_HORIZON_DAYS,
    )

    summary = create_horizon_summary(
        daily_forecast=daily_forecast
    )

    output_path = save_daily_forecast(
        daily_forecast=daily_forecast
    )

    print(
        "Historical-average forecasting "
        "completed successfully."
    )
    print(f"Forecast rows: {len(daily_forecast)}")
    print(f"Saved to: {output_path}")
    print()
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()