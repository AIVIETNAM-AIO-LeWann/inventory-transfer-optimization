"""Tests for inventory analysis functions."""

import numpy as np
import pandas as pd
import pytest

from src.inventory_analyzer import (
    BALANCED_STATUS,
    EXCESS_STATUS,
    SHORTAGE_STATUS,
    analyze_inventory,
    create_inventory_summary,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create simple sales data for inventory tests."""

    records = []

    for product_id in ("P001", "P002", "P003"):
        for date in pd.date_range(
            start="2026-01-01",
            periods=2,
            freq="D",
        ):
            records.append(
                {
                    "date": date,
                    "store_id": "S001",
                    "product_id": product_id,
                    "quantity_sold": 10,
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def sample_inventory() -> pd.DataFrame:
    """Create shortage, balanced, and excess inventory rows."""

    return pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "current_stock": 20,
                "last_updated": pd.Timestamp("2026-01-02"),
            },
            {
                "store_id": "S001",
                "product_id": "P002",
                "current_stock": 140,
                "last_updated": pd.Timestamp("2026-01-02"),
            },
            {
                "store_id": "S001",
                "product_id": "P003",
                "current_stock": 250,
                "last_updated": pd.Timestamp("2026-01-02"),
            },
        ]
    )


def test_analyze_inventory_classifies_inventory_statuses(
    sample_sales: pd.DataFrame,
    sample_inventory: pd.DataFrame,
) -> None:
    """Inventory rows should be classified correctly."""

    result = analyze_inventory(
        sales=sample_sales,
        inventory=sample_inventory,
        minimum_days=7,
        target_days=14,
        maximum_days=21,
    )

    result_by_product = result.set_index("product_id")

    assert (
        result_by_product.loc["P001", "status"]
        == SHORTAGE_STATUS
    )

    assert (
        result_by_product.loc["P002", "status"]
        == BALANCED_STATUS
    )

    assert (
        result_by_product.loc["P003", "status"]
        == EXCESS_STATUS
    )


def test_analyze_inventory_calculates_quantities(
    sample_sales: pd.DataFrame,
    sample_inventory: pd.DataFrame,
) -> None:
    """Inventory quantities should follow the target stock level."""

    result = analyze_inventory(
        sales=sample_sales,
        inventory=sample_inventory,
        minimum_days=7,
        target_days=14,
        maximum_days=21,
    )

    result_by_product = result.set_index("product_id")

    assert (
        result_by_product.loc[
            "P001",
            "average_daily_sales",
        ]
        == 10.0
    )

    assert (
        result_by_product.loc[
            "P001",
            "inventory_days",
        ]
        == 2.0
    )

    assert (
        result_by_product.loc[
            "P001",
            "target_stock",
        ]
        == 140
    )

    assert (
        result_by_product.loc[
            "P001",
            "shortage_quantity",
        ]
        == 120
    )

    assert (
        result_by_product.loc[
            "P003",
            "excess_quantity",
        ]
        == 110
    )


def test_zero_sales_with_stock_is_excess() -> None:
    """Stock with no demand should be classified as excess."""

    sales = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 0,
            },
            {
                "date": "2026-01-02",
                "store_id": "S001",
                "product_id": "P001",
                "quantity_sold": 0,
            },
        ]
    )

    inventory = pd.DataFrame(
        [
            {
                "store_id": "S001",
                "product_id": "P001",
                "current_stock": 20,
                "last_updated": "2026-01-02",
            }
        ]
    )

    result = analyze_inventory(
        sales=sales,
        inventory=inventory,
    )

    row = result.iloc[0]

    assert row["average_daily_sales"] == 0
    assert np.isnan(row["inventory_days"])
    assert row["status"] == EXCESS_STATUS
    assert row["excess_quantity"] == 20


def test_create_inventory_summary(
    sample_sales: pd.DataFrame,
    sample_inventory: pd.DataFrame,
) -> None:
    """Summary should aggregate inventory results by status."""

    analysis = analyze_inventory(
        sales=sample_sales,
        inventory=sample_inventory,
        minimum_days=7,
        target_days=14,
        maximum_days=21,
    )

    summary = create_inventory_summary(
        inventory_analysis=analysis
    )

    summary_by_status = summary.set_index("status")

    assert (
        summary_by_status.loc[
            SHORTAGE_STATUS,
            "store_product_count",
        ]
        == 1
    )

    assert (
        summary_by_status.loc[
            SHORTAGE_STATUS,
            "total_shortage_quantity",
        ]
        == 120
    )

    assert (
        summary_by_status.loc[
            EXCESS_STATUS,
            "total_excess_quantity",
        ]
        == 110
    )


def test_analyze_inventory_rejects_invalid_thresholds(
    sample_sales: pd.DataFrame,
    sample_inventory: pd.DataFrame,
) -> None:
    """Inventory thresholds must be in ascending order."""

    with pytest.raises(
        ValueError,
        match="Inventory thresholds must satisfy",
    ):
        analyze_inventory(
            sales=sample_sales,
            inventory=sample_inventory,
            minimum_days=14,
            target_days=7,
            maximum_days=21,
        )