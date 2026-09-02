"""Tests for forecast-based inventory analysis."""

import pandas as pd
import pytest

from src.config import MOVING_AVERAGE_METHOD
from src.inventory_analyzer import (
    BALANCED_STATUS,
    EXCESS_STATUS,
    FORECAST_ANALYSIS_COLUMNS,
    SHORTAGE_STATUS,
    analyze_inventory_with_forecast,
    summarize_inventory_forecast,
)


def create_daily_forecast(
    horizon_days: int,
    predicted_quantity: float = 10.0,
) -> pd.DataFrame:
    """Create daily forecasts for three stores."""

    rows = []
    stores = ["S001", "S002", "S003"]
    forecast_start_date = pd.Timestamp("2026-01-11")

    for store_id in stores:
        for forecast_day in range(
            1,
            horizon_days + 1,
        ):
            rows.append(
                {
                    "store_id": store_id,
                    "product_id": "P001",
                    "forecast_date": (
                        forecast_start_date
                        + pd.Timedelta(
                            days=forecast_day - 1
                        )
                    ),
                    "forecast_day": forecast_day,
                    "predicted_quantity": (
                        predicted_quantity
                    ),
                    "method": MOVING_AVERAGE_METHOD,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_inventory() -> pd.DataFrame:
    """Create shortage, balanced, and excess inventory."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "current_stock": 40,
                "last_updated": "2026-01-10",
            },
            {
                "store_id": "S002",
                "product_id": "P001",
                "current_stock": 150,
                "last_updated": "2026-01-10",
            },
            {
                "store_id": "S003",
                "product_id": "P001",
                "current_stock": 230,
                "last_updated": "2026-01-10",
            },
        ]
    )


def test_summarize_inventory_forecast() -> None:
    """Daily forecasts should be summarized by store-product."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    summary = summarize_inventory_forecast(
        daily_forecast
    )

    first_row = summary.loc[
        summary["store_id"] == "S001"
    ].iloc[0]

    assert len(summary) == 3
    assert first_row["forecast_method"] == (
        MOVING_AVERAGE_METHOD
    )
    assert first_row["replenishment_horizon_days"] == 7
    assert first_row["predicted_short_term_demand"] == 70
    assert first_row["predicted_horizon_demand"] == 70
    assert first_row["predicted_daily_average"] == 10


def test_analyze_short_term_inventory(
    sample_inventory: pd.DataFrame,
) -> None:
    """A short-term request should target seven days."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    result = analyze_inventory_with_forecast(
        inventory=sample_inventory,
        daily_forecast=daily_forecast,
    )

    shortage_row = result.loc[
        result["store_id"] == "S001"
    ].iloc[0]

    balanced_row = result.loc[
        result["store_id"] == "S002"
    ].iloc[0]

    excess_row = result.loc[
        result["store_id"] == "S003"
    ].iloc[0]

    assert tuple(result.columns) == (
        FORECAST_ANALYSIS_COLUMNS
    )

    assert shortage_row["status"] == SHORTAGE_STATUS
    assert shortage_row["minimum_stock"] == 70
    assert shortage_row["target_stock"] == 70
    assert shortage_row["shortage_quantity"] == 30
    assert shortage_row["excess_quantity"] == 0

    assert balanced_row["status"] == BALANCED_STATUS
    assert balanced_row["shortage_quantity"] == 0
    assert balanced_row["excess_quantity"] == 0

    assert excess_row["status"] == EXCESS_STATUS
    assert excess_row["donor_reserve_stock"] == 140
    assert excess_row["maximum_stock"] == 210
    assert excess_row["shortage_quantity"] == 0
    assert excess_row["excess_quantity"] == 90


def test_analyze_long_term_inventory(
    sample_inventory: pd.DataFrame,
) -> None:
    """A long-term request should target fourteen days."""

    daily_forecast = create_daily_forecast(
        horizon_days=14
    )

    result = analyze_inventory_with_forecast(
        inventory=sample_inventory,
        daily_forecast=daily_forecast,
    )

    shortage_row = result.loc[
        result["store_id"] == "S001"
    ].iloc[0]

    assert shortage_row[
        "replenishment_horizon_days"
    ] == 14

    assert shortage_row[
        "predicted_short_term_demand"
    ] == 70

    assert shortage_row[
        "predicted_horizon_demand"
    ] == 140

    assert shortage_row["minimum_stock"] == 70
    assert shortage_row["target_stock"] == 140
    assert shortage_row["shortage_quantity"] == 100


def test_zero_demand_with_stock_is_excess() -> None:
    """Stock without predicted demand should be excess."""

    inventory = pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "current_stock": 20,
                "last_updated": "2026-01-10",
            }
        ]
    )

    daily_forecast = create_daily_forecast(
        horizon_days=7,
        predicted_quantity=0,
    )

    daily_forecast = daily_forecast.loc[
        daily_forecast["store_id"] == "S001"
    ].reset_index(drop=True)

    result = analyze_inventory_with_forecast(
        inventory=inventory,
        daily_forecast=daily_forecast,
    )

    row = result.iloc[0]

    assert row["predicted_daily_average"] == 0
    assert row["target_stock"] == 0
    assert row["status"] == EXCESS_STATUS
    assert row["excess_quantity"] == 20


def test_missing_forecast_for_inventory_is_rejected(
    sample_inventory: pd.DataFrame,
) -> None:
    """Every inventory row should have a forecast."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    additional_inventory = pd.DataFrame(
        [
            {
                "store_id": "S004",
                "product_id": "P001",
                "current_stock": 10,
                "last_updated": "2026-01-10",
            }
        ]
    )

    inventory = pd.concat(
        [
            sample_inventory,
            additional_inventory,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="does not cover",
    ):
        analyze_inventory_with_forecast(
            inventory=inventory,
            daily_forecast=daily_forecast,
        )


def test_unsupported_forecast_horizon_is_rejected() -> None:
    """Only seven-day and fourteen-day forecasts are supported."""

    daily_forecast = create_daily_forecast(
        horizon_days=9
    )

    with pytest.raises(
        ValueError,
        match="supported replenishment horizon",
    ):
        summarize_inventory_forecast(
            daily_forecast
        )


def test_duplicate_forecast_day_is_rejected() -> None:
    """Duplicate store-product-day forecasts should be rejected."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    duplicated_forecast = pd.concat(
        [
            daily_forecast,
            daily_forecast.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        summarize_inventory_forecast(
            duplicated_forecast
        )


def test_negative_forecast_is_rejected() -> None:
    """Negative predicted quantities should be rejected."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    daily_forecast.loc[
        0,
        "predicted_quantity",
    ] = -1

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        summarize_inventory_forecast(
            daily_forecast
        )


def test_missing_inventory_column_is_rejected(
    sample_inventory: pd.DataFrame,
) -> None:
    """Missing inventory columns should be rejected."""

    invalid_inventory = sample_inventory.drop(
        columns="last_updated"
    )

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        analyze_inventory_with_forecast(
            inventory=invalid_inventory,
            daily_forecast=daily_forecast,
        )


def test_non_dataframe_inventory_is_rejected() -> None:
    """Inventory must be a pandas DataFrame."""

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        analyze_inventory_with_forecast(
            inventory="invalid",
            daily_forecast=daily_forecast,
        )


def test_missing_inventory_value_is_rejected(
    sample_inventory: pd.DataFrame,
) -> None:
    """Missing inventory values should be rejected."""

    invalid_inventory = sample_inventory.copy()

    invalid_inventory.loc[
        0,
        "current_stock",
    ] = None

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        analyze_inventory_with_forecast(
            inventory=invalid_inventory,
            daily_forecast=daily_forecast,
        )


def test_negative_inventory_is_rejected(
    sample_inventory: pd.DataFrame,
) -> None:
    """Negative inventory quantities should be rejected."""

    invalid_inventory = sample_inventory.copy()

    invalid_inventory.loc[
        0,
        "current_stock",
    ] = -1

    daily_forecast = create_daily_forecast(
        horizon_days=7
    )

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        analyze_inventory_with_forecast(
            inventory=invalid_inventory,
            daily_forecast=daily_forecast,
        )