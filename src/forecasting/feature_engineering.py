"""Create time-series features for demand forecasting models."""

import pandas as pd

from src.config import (
    FORECAST_LAG_DAYS,
    FORECAST_ROLLING_WINDOWS,
)
from src.data_loader import load_all_data
from src.forecasting.historical_average import (
    REQUIRED_SALES_COLUMNS,
    validate_sales_for_forecasting,
)


CALENDAR_FEATURE_COLUMNS = (
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "time_index",
)


def validate_feature_settings(
    lag_days: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> None:
    """Validate lag days and rolling window settings."""

    if not lag_days:
        raise ValueError(
            "lag_days must not be empty."
        )

    if not rolling_windows:
        raise ValueError(
            "rolling_windows must not be empty."
        )

    invalid_lag_days = [
        lag_day
        for lag_day in lag_days
        if (
            not isinstance(lag_day, int)
            or lag_day <= 0
        )
    ]

    if invalid_lag_days:
        raise ValueError(
            "lag_days must contain positive "
            f"integers: {invalid_lag_days}"
        )

    invalid_rolling_windows = [
        window
        for window in rolling_windows
        if (
            not isinstance(window, int)
            or window <= 0
        )
    ]

    if invalid_rolling_windows:
        raise ValueError(
            "rolling_windows must contain positive "
            f"integers: {invalid_rolling_windows}"
        )

    if len(set(lag_days)) != len(lag_days):
        raise ValueError(
            "lag_days must not contain duplicates."
        )

    if (
        len(set(rolling_windows))
        != len(rolling_windows)
    ):
        raise ValueError(
            "rolling_windows must not contain "
            "duplicates."
        )


def validate_complete_daily_history(
    sales: pd.DataFrame,
) -> None:
    """Validate that every pair has one row for every date."""

    sales_dates = pd.to_datetime(
        sales["date"],
        errors="coerce",
    ).dt.normalize()

    if sales_dates.isna().any():
        raise ValueError(
            "Sales data contains invalid dates."
        )

    first_date = sales_dates.min()
    last_date = sales_dates.max()

    expected_days = (
        last_date - first_date
    ).days + 1

    unique_date_count = sales_dates.nunique()

    if unique_date_count != expected_days:
        raise ValueError(
            "Sales history must contain a continuous "
            "daily date sequence."
        )

    history = sales[
        [
            "store_id",
            "product_id",
        ]
    ].copy()

    history["_date"] = sales_dates

    pair_statistics = (
        history.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )
        .agg(
            record_count=("_date", "size"),
            first_date=("_date", "min"),
            last_date=("_date", "max"),
        )
    )

    complete_pairs = (
        (
            pair_statistics["record_count"]
            == expected_days
        )
        & (
            pair_statistics["first_date"]
            == first_date
        )
        & (
            pair_statistics["last_date"]
            == last_date
        )
    )

    if not complete_pairs.all():
        incomplete_pair_count = int(
            (~complete_pairs).sum()
        )

        raise ValueError(
            "Every store-product pair must contain "
            "a complete daily history. "
            f"Incomplete pairs: {incomplete_pair_count}."
        )


def add_calendar_features(
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add calendar-based features from the sales date."""

    result = feature_data.copy()

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_month"] = (
        result["date"].dt.day
    )

    result["week_of_year"] = (
        result["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    result["month"] = (
        result["date"].dt.month
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    first_date = result["date"].min()

    result["time_index"] = (
        result["date"] - first_date
    ).dt.days.astype(int)

    return result


def add_lag_features(
    feature_data: pd.DataFrame,
    lag_days: tuple[int, ...],
) -> pd.DataFrame:
    """Add previous demand values for each pair."""

    result = feature_data.copy()

    grouped_sales = result.groupby(
        ["store_id", "product_id"],
        sort=False,
    )["quantity_sold"]

    for lag_day in lag_days:
        result[f"lag_{lag_day}"] = (
            grouped_sales.shift(lag_day)
        )

    return result


def add_rolling_features(
    feature_data: pd.DataFrame,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    """Add rolling means using only previous demand."""

    result = feature_data.copy()

    for window in rolling_windows:
        result[f"rolling_mean_{window}"] = (
            result.groupby(
                ["store_id", "product_id"],
                sort=False,
            )["quantity_sold"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

    return result


def get_feature_data_columns(
    lag_days: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> tuple[str, ...]:
    """Return the ordered columns of feature data."""

    lag_columns = tuple(
        f"lag_{lag_day}"
        for lag_day in lag_days
    )

    rolling_columns = tuple(
        f"rolling_mean_{window}"
        for window in rolling_windows
    )

    return (
        "date",
        "store_id",
        "product_id",
        "quantity_sold",
        *CALENDAR_FEATURE_COLUMNS,
        *lag_columns,
        *rolling_columns,
    )


def create_time_series_features(
    sales: pd.DataFrame,
    lag_days: tuple[int, ...] = (
        FORECAST_LAG_DAYS
    ),
    rolling_windows: tuple[int, ...] = (
        FORECAST_ROLLING_WINDOWS
    ),
    drop_incomplete_rows: bool = False,
) -> pd.DataFrame:
    """Create calendar, lag, and rolling demand features."""

    validate_sales_for_forecasting(sales)

    validate_feature_settings(
        lag_days=lag_days,
        rolling_windows=rolling_windows,
    )

    if not isinstance(
        drop_incomplete_rows,
        bool,
    ):
        raise TypeError(
            "drop_incomplete_rows must be a boolean."
        )

    working_sales = sales[
        list(REQUIRED_SALES_COLUMNS)
    ].copy()

    working_sales["date"] = pd.to_datetime(
        working_sales["date"]
    ).dt.normalize()

    working_sales["store_id"] = (
        working_sales["store_id"].astype(str)
    )

    working_sales["product_id"] = (
        working_sales["product_id"].astype(str)
    )

    working_sales = working_sales.sort_values(
        by=[
            "store_id",
            "product_id",
            "date",
        ],
        ignore_index=True,
    )

    validate_complete_daily_history(
        working_sales
    )

    feature_data = add_calendar_features(
        working_sales
    )

    feature_data = add_lag_features(
        feature_data=feature_data,
        lag_days=lag_days,
    )

    feature_data = add_rolling_features(
        feature_data=feature_data,
        rolling_windows=rolling_windows,
    )

    feature_columns = get_feature_data_columns(
        lag_days=lag_days,
        rolling_windows=rolling_windows,
    )

    if drop_incomplete_rows:
        generated_feature_columns = [
            column
            for column in feature_columns
            if (
                column.startswith("lag_")
                or column.startswith(
                    "rolling_mean_"
                )
            )
        ]

        feature_data = feature_data.dropna(
            subset=generated_feature_columns
        ).reset_index(drop=True)

    return feature_data[
        list(feature_columns)
    ]


def main() -> None:
    """Create time-series features from project sales."""

    project_data = load_all_data()

    feature_data = create_time_series_features(
        sales=project_data.sales,
        drop_incomplete_rows=False,
    )

    complete_feature_data = (
        create_time_series_features(
            sales=project_data.sales,
            drop_incomplete_rows=True,
        )
    )

    incomplete_rows = (
        len(feature_data)
        - len(complete_feature_data)
    )

    print(
        "Time-series features created "
        "successfully."
    )
    print(
        f"Original sales rows: "
        f"{len(project_data.sales):,}"
    )
    print(
        f"Feature rows: {len(feature_data):,}"
    )
    print(
        "Complete training rows: "
        f"{len(complete_feature_data):,}"
    )
    print(
        f"Rows requiring history: "
        f"{incomplete_rows:,}"
    )
    print()
    print(
        complete_feature_data.tail(
            10
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()