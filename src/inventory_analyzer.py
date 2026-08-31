"""Analyze inventory shortage and excess levels."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    INVENTORY_ANALYSIS_FILE,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    TARGET_INVENTORY_DAYS,
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