"""Analyze inventory shortage and excess levels."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INVENTORY_ANALYSIS_FILE,
    LONG_TERM_REPLENISHMENT_DAYS,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    SHORT_TERM_REPLENISHMENT_DAYS,
    TARGET_INVENTORY_DAYS,
)

from src.forecasting.historical_average import (
    FORECAST_COLUMNS,
)

from src.data_loader import load_all_data


SHORTAGE_STATUS = "shortage"
BALANCED_STATUS = "balanced"
EXCESS_STATUS = "excess"


ANALYSIS_COLUMNS = (
    "store_id",
    "product_id",
    "current_stock",
    "total_quantity_sold",
    "analysis_days",
    "average_daily_sales",
    "inventory_days",
    "minimum_stock",
    "target_stock",
    "maximum_stock",
    "status",
    "shortage_quantity",
    "excess_quantity",
    "last_updated",
)

FORECAST_ANALYSIS_COLUMNS = (
    "store_id",
    "product_id",
    "current_stock",
    "forecast_method",
    "forecast_start_date",
    "forecast_end_date",
    "replenishment_horizon_days",
    "predicted_short_term_demand",
    "predicted_horizon_demand",
    "predicted_daily_average",
    "inventory_days",
    "minimum_stock",
    "target_stock",
    "donor_reserve_stock",
    "maximum_stock",
    "status",
    "shortage_quantity",
    "excess_quantity",
    "last_updated",
)

def validate_inventory_thresholds(
    minimum_days: int,
    target_days: int,
    maximum_days: int,
) -> None:
    """Validate inventory coverage thresholds."""

    if not (
        0 < minimum_days < target_days < maximum_days
    ):
        raise ValueError(
            "Inventory thresholds must satisfy: "
            "0 < minimum_days < target_days < maximum_days."
        )


def calculate_average_daily_sales(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average daily sales for each store-product pair."""

    sales_dates = pd.to_datetime(
        sales["date"],
        errors="coerce",
    )

    if sales_dates.isna().any():
        raise ValueError(
            "Sales data contains invalid date values."
        )

    first_date = sales_dates.min()
    last_date = sales_dates.max()

    analysis_days = (
        last_date - first_date
    ).days + 1

    if analysis_days <= 0:
        raise ValueError(
            "Sales analysis period must be greater than zero."
        )

    sales_totals = (
        sales.groupby(
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

    sales_totals["analysis_days"] = analysis_days

    sales_totals["average_daily_sales"] = (
        sales_totals["total_quantity_sold"]
        / analysis_days
    )

    return sales_totals

def summarize_inventory_forecast(
    daily_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize daily forecasts for inventory analysis."""

    if not isinstance(daily_forecast, pd.DataFrame):
        raise TypeError(
            "daily_forecast must be a pandas DataFrame."
        )

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

    if (
        daily_forecast[list(FORECAST_COLUMNS)]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "daily_forecast contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        daily_forecast["predicted_quantity"]
    ):
        raise TypeError(
            "predicted_quantity must be numeric."
        )

    if (
        daily_forecast["predicted_quantity"] < 0
    ).any():
        raise ValueError(
            "predicted_quantity must not be negative."
        )

    if not pd.api.types.is_integer_dtype(
        daily_forecast["forecast_day"]
    ):
        raise TypeError(
            "forecast_day must contain integers."
        )

    duplicated_rows = daily_forecast.duplicated(
        subset=[
            "store_id",
            "product_id",
            "forecast_day",
        ],
        keep=False,
    )

    if duplicated_rows.any():
        raise ValueError(
            "daily_forecast contains duplicate "
            "store-product-day rows."
        )

    working_forecast = daily_forecast[
        list(FORECAST_COLUMNS)
    ].copy()

    working_forecast["store_id"] = (
        working_forecast["store_id"].astype(str)
    )

    working_forecast["product_id"] = (
        working_forecast["product_id"].astype(str)
    )

    working_forecast["forecast_date"] = pd.to_datetime(
        working_forecast["forecast_date"],
        errors="coerce",
    )

    if working_forecast["forecast_date"].isna().any():
        raise ValueError(
            "forecast_date contains invalid values."
        )

    methods = working_forecast[
        "method"
    ].unique()

    if len(methods) != 1:
        raise ValueError(
            "daily_forecast must contain exactly "
            "one forecast method."
        )

    replenishment_horizon_days = int(
        working_forecast["forecast_day"].max()
    )

    supported_horizons = {
        SHORT_TERM_REPLENISHMENT_DAYS,
        LONG_TERM_REPLENISHMENT_DAYS,
    }

    if (
        replenishment_horizon_days
        not in supported_horizons
    ):
        raise ValueError(
            "Forecast horizon must be a supported "
            f"replenishment horizon: {supported_horizons}."
        )

    pair_day_statistics = (
        working_forecast.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )
        .agg(
            row_count=("forecast_day", "size"),
            minimum_day=("forecast_day", "min"),
            maximum_day=("forecast_day", "max"),
        )
    )

    complete_pairs = (
        (
            pair_day_statistics["row_count"]
            == replenishment_horizon_days
        )
        & (
            pair_day_statistics["minimum_day"]
            == 1
        )
        & (
            pair_day_statistics["maximum_day"]
            == replenishment_horizon_days
        )
    )

    if not complete_pairs.all():
        raise ValueError(
            "Every store-product pair must contain "
            "a complete forecast horizon."
        )

    horizon_summary = (
        working_forecast.groupby(
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
            replenishment_horizon_days=(
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
        .rename(
            columns={
                "method": "forecast_method",
            }
        )
    )

    short_term_summary = (
        working_forecast.loc[
            working_forecast["forecast_day"]
            <= SHORT_TERM_REPLENISHMENT_DAYS
        ]
        .groupby(
            ["store_id", "product_id"],
            as_index=False,
        )
        .agg(
            predicted_short_term_demand=(
                "predicted_quantity",
                "sum",
            )
        )
    )

    return horizon_summary.merge(
        short_term_summary,
        on=["store_id", "product_id"],
        how="left",
        validate="one_to_one",
    )

def analyze_inventory(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    minimum_days: int = MIN_INVENTORY_DAYS,
    target_days: int = TARGET_INVENTORY_DAYS,
    maximum_days: int = MAX_INVENTORY_DAYS,
) -> pd.DataFrame:
    """Analyze inventory status for every store-product pair."""

    validate_inventory_thresholds(
        minimum_days=minimum_days,
        target_days=target_days,
        maximum_days=maximum_days,
    )

    average_sales = calculate_average_daily_sales(
        sales
    )

    analysis = inventory.merge(
        average_sales,
        on=["store_id", "product_id"],
        how="left",
        validate="one_to_one",
    )

    analysis[
        [
            "total_quantity_sold",
            "average_daily_sales",
        ]
    ] = analysis[
        [
            "total_quantity_sold",
            "average_daily_sales",
        ]
    ].fillna(0)

    analysis["analysis_days"] = (
        analysis["analysis_days"]
        .fillna(0)
        .astype(int)
    )

    average_daily_sales = analysis[
        "average_daily_sales"
    ].to_numpy(dtype=float)

    current_stock = analysis[
        "current_stock"
    ].to_numpy(dtype=float)

    inventory_days = np.divide(
        current_stock,
        average_daily_sales,
        out=np.full(
            len(analysis),
            np.nan,
            dtype=float,
        ),
        where=average_daily_sales > 0,
    )

    analysis["inventory_days"] = inventory_days

    analysis["minimum_stock"] = np.ceil(
        analysis["average_daily_sales"]
        * minimum_days
    ).astype(int)

    analysis["target_stock"] = np.ceil(
        analysis["average_daily_sales"]
        * target_days
    ).astype(int)

    analysis["maximum_stock"] = np.ceil(
        analysis["average_daily_sales"]
        * maximum_days
    ).astype(int)

    shortage_condition = (
        (analysis["average_daily_sales"] > 0)
        & (analysis["inventory_days"] < minimum_days)
    )

    excess_condition = (
        (
            (analysis["average_daily_sales"] > 0)
            & (
                analysis["inventory_days"]
                > maximum_days
            )
        )
        | (
            (analysis["average_daily_sales"] == 0)
            & (analysis["current_stock"] > 0)
        )
    )

    analysis["status"] = np.select(
        condlist=[
            shortage_condition,
            excess_condition,
        ],
        choicelist=[
            SHORTAGE_STATUS,
            EXCESS_STATUS,
        ],
        default=BALANCED_STATUS,
    )

    analysis["shortage_quantity"] = np.where(
        analysis["status"] == SHORTAGE_STATUS,
        np.maximum(
            analysis["target_stock"]
            - analysis["current_stock"],
            0,
        ),
        0,
    ).astype(int)

    analysis["excess_quantity"] = np.where(
        analysis["status"] == EXCESS_STATUS,
        np.maximum(
            analysis["current_stock"]
            - analysis["target_stock"],
            0,
        ),
        0,
    ).astype(int)

    analysis["average_daily_sales"] = analysis[
        "average_daily_sales"
    ].round(2)

    analysis["inventory_days"] = analysis[
        "inventory_days"
    ].round(2)

    return analysis[list(ANALYSIS_COLUMNS)]


def analyze_inventory_with_forecast(
    inventory: pd.DataFrame,
    daily_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze inventory using forecasted demand."""

    if not isinstance(inventory, pd.DataFrame):
        raise TypeError(
            "inventory must be a pandas DataFrame."
        )

    if inventory.empty:
        raise ValueError(
            "inventory must not be empty."
        )

    required_inventory_columns = {
        "store_id",
        "product_id",
        "current_stock",
        "last_updated",
    }

    missing_columns = (
        required_inventory_columns
        - set(inventory.columns)
    )

    if missing_columns:
        raise ValueError(
            "inventory is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        inventory[list(required_inventory_columns)]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "inventory contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        inventory["current_stock"]
    ):
        raise TypeError(
            "inventory.current_stock must be numeric."
        )

    if (inventory["current_stock"] < 0).any():
        raise ValueError(
            "inventory.current_stock must not "
            "be negative."
        )

    duplicated_inventory = inventory.duplicated(
        subset=["store_id", "product_id"],
        keep=False,
    )

    if duplicated_inventory.any():
        raise ValueError(
            "inventory contains duplicate "
            "store-product rows."
        )

    forecast_summary = summarize_inventory_forecast(
        daily_forecast
    )

    working_inventory = inventory.copy()

    working_inventory["store_id"] = (
        working_inventory["store_id"].astype(str)
    )

    working_inventory["product_id"] = (
        working_inventory["product_id"].astype(str)
    )

    analysis = working_inventory.merge(
        forecast_summary,
        on=["store_id", "product_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_forecasts = (
        analysis["_merge"] == "left_only"
    ).sum()

    if missing_forecasts:
        raise ValueError(
            "daily_forecast does not cover "
            f"{missing_forecasts} inventory rows."
        )

    analysis = analysis.drop(columns="_merge")

    predicted_daily_average = analysis[
        "predicted_daily_average"
    ].to_numpy(dtype=float)

    current_stock = analysis[
        "current_stock"
    ].to_numpy(dtype=float)

    analysis["inventory_days"] = np.divide(
        current_stock,
        predicted_daily_average,
        out=np.full(
            len(analysis),
            np.nan,
            dtype=float,
        ),
        where=predicted_daily_average > 0,
    )

    analysis["minimum_stock"] = np.ceil(
        analysis["predicted_short_term_demand"]
    ).astype(int)

    analysis["target_stock"] = np.ceil(
        analysis["predicted_horizon_demand"]
    ).astype(int)

    analysis["donor_reserve_stock"] = np.ceil(
        analysis["predicted_daily_average"]
        * LONG_TERM_REPLENISHMENT_DAYS
    ).astype(int)

    analysis["maximum_stock"] = np.ceil(
        analysis["predicted_daily_average"]
        * MAX_INVENTORY_DAYS
    ).astype(int)

    shortage_condition = (
        (analysis["predicted_daily_average"] > 0)
        & (
            analysis["current_stock"]
            < analysis["target_stock"]
        )
    )

    excess_condition = (
        (
            (analysis["predicted_daily_average"] > 0)
            & (
                analysis["current_stock"]
                > analysis["maximum_stock"]
            )
        )
        | (
            (analysis["predicted_daily_average"] == 0)
            & (analysis["current_stock"] > 0)
        )
    )

    analysis["status"] = np.select(
        condlist=[
            shortage_condition,
            excess_condition,
        ],
        choicelist=[
            SHORTAGE_STATUS,
            EXCESS_STATUS,
        ],
        default=BALANCED_STATUS,
    )

    analysis["shortage_quantity"] = np.where(
        analysis["status"] == SHORTAGE_STATUS,
        np.maximum(
            analysis["target_stock"]
            - analysis["current_stock"],
            0,
        ),
        0,
    ).astype(int)

    analysis["excess_quantity"] = np.where(
        analysis["status"] == EXCESS_STATUS,
        np.maximum(
            analysis["current_stock"]
            - analysis["donor_reserve_stock"],
            0,
        ),
        0,
    ).astype(int)

    columns_to_round = [
        "predicted_short_term_demand",
        "predicted_horizon_demand",
        "predicted_daily_average",
        "inventory_days",
    ]

    analysis[columns_to_round] = analysis[
        columns_to_round
    ].round(2)

    return analysis[
        list(FORECAST_ANALYSIS_COLUMNS)
    ]


def create_inventory_summary(
    inventory_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize inventory results by status."""

    summary = (
        inventory_analysis.groupby(
            "status",
            as_index=False,
        )
        .agg(
            store_product_count=(
                "product_id",
                "size",
            ),
            total_current_stock=(
                "current_stock",
                "sum",
            ),
            total_shortage_quantity=(
                "shortage_quantity",
                "sum",
            ),
            total_excess_quantity=(
                "excess_quantity",
                "sum",
            ),
        )
    )

    return summary


def save_inventory_analysis(
    inventory_analysis: pd.DataFrame,
    output_path: str | Path = INVENTORY_ANALYSIS_FILE,
) -> Path:
    """Save inventory analysis results to a CSV file."""

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_analysis.to_csv(
        destination,
        index=False,
        encoding="utf-8-sig",
    )

    return destination.resolve()


def main() -> None:
    """Run inventory analysis using project data."""

    project_data = load_all_data()

    inventory_analysis = analyze_inventory(
        sales=project_data.sales,
        inventory=project_data.inventory,
    )

    summary = create_inventory_summary(
        inventory_analysis
    )

    output_path = save_inventory_analysis(
        inventory_analysis
    )

    print("Inventory analysis completed successfully.")
    print(f"Analyzed rows: {len(inventory_analysis)}")
    print(f"Saved to: {output_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()