"""Tests for time-series feature engineering."""

import pandas as pd
import pytest

from src.forecasting.feature_engineering import (
    CALENDAR_FEATURE_COLUMNS,
    create_time_series_features,
    get_feature_data_columns,
    validate_complete_daily_history,
    validate_feature_settings,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Create thirty-five days of sales for two products."""

    dates = pd.date_range(
        "2026-01-01",
        periods=35,
        freq="D",
    )

    rows = []

    for index, date in enumerate(
        dates,
        start=1,
    ):
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
                "quantity_sold": 100 + index,
            }
        )

    return pd.DataFrame(rows)


def test_validate_feature_settings_accepts_valid_values() -> None:
    """Valid lag and rolling settings should be accepted."""

    validate_feature_settings(
        lag_days=(1, 7, 14),
        rolling_windows=(7, 14, 28),
    )


@pytest.mark.parametrize(
    (
        "lag_days",
        "rolling_windows",
    ),
    [
        ((), (7,)),
        ((0,), (7,)),
        ((1.5,), (7,)),
        ((1, 1), (7,)),
        ((1,), ()),
        ((1,), (7, 7)),
    ],
)
def test_validate_feature_settings_rejects_invalid_values(
    lag_days: tuple,
    rolling_windows: tuple,
) -> None:
    """Invalid feature settings should be rejected."""

    with pytest.raises(ValueError):
        validate_feature_settings(
            lag_days=lag_days,
            rolling_windows=rolling_windows,
        )


def test_validate_complete_daily_history(
    sample_sales: pd.DataFrame,
) -> None:
    """Complete store-product histories should be accepted."""

    validate_complete_daily_history(
        sample_sales
    )


def test_incomplete_store_product_history_is_rejected(
    sample_sales: pd.DataFrame,
) -> None:
    """Every pair should contain all expected dates."""

    incomplete_sales = sample_sales.loc[
        ~(
            (
                sample_sales["product_id"]
                == "P001"
            )
            & (
                sample_sales["date"]
                == pd.Timestamp("2026-01-10")
            )
        )
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="Incomplete pairs",
    ):
        validate_complete_daily_history(
            incomplete_sales
        )


def test_calendar_features(
    sample_sales: pd.DataFrame,
) -> None:
    """Calendar features should match the sales date."""

    features = create_time_series_features(
        sales=sample_sales,
        lag_days=(1,),
        rolling_windows=(2,),
    )

    first_row = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-01")
        )
    ].iloc[0]

    saturday_row = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-03")
        )
    ].iloc[0]

    assert first_row["day_of_week"] == 3
    assert first_row["day_of_month"] == 1
    assert first_row["week_of_year"] == 1
    assert first_row["month"] == 1
    assert first_row["quarter"] == 1
    assert first_row["is_weekend"] == 0
    assert first_row["time_index"] == 0

    assert saturday_row["day_of_week"] == 5
    assert saturday_row["is_weekend"] == 1
    assert saturday_row["time_index"] == 2


def test_lag_features_are_calculated_per_product(
    sample_sales: pd.DataFrame,
) -> None:
    """Lag values should not cross product boundaries."""

    features = create_time_series_features(
        sales=sample_sales,
        lag_days=(1, 7),
        rolling_windows=(7,),
    )

    product_one_day_eight = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-08")
        )
    ].iloc[0]

    product_two_day_eight = features.loc[
        (
            features["product_id"] == "P002"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-08")
        )
    ].iloc[0]

    assert product_one_day_eight["lag_1"] == 7
    assert product_one_day_eight["lag_7"] == 1

    assert product_two_day_eight["lag_1"] == 107
    assert product_two_day_eight["lag_7"] == 101


def test_rolling_mean_uses_only_previous_days(
    sample_sales: pd.DataFrame,
) -> None:
    """Rolling features must not include the current target."""

    features = create_time_series_features(
        sales=sample_sales,
        lag_days=(1,),
        rolling_windows=(7,),
    )

    day_eight = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-08")
        )
    ].iloc[0]

    expected_rolling_mean = (
        1 + 2 + 3 + 4 + 5 + 6 + 7
    ) / 7

    assert day_eight[
        "rolling_mean_7"
    ] == pytest.approx(
        expected_rolling_mean
    )

    assert day_eight[
        "rolling_mean_7"
    ] != pytest.approx(5)


def test_initial_feature_rows_contain_missing_history(
    sample_sales: pd.DataFrame,
) -> None:
    """Early rows should be missing unavailable history."""

    features = create_time_series_features(
        sales=sample_sales,
        lag_days=(1, 7),
        rolling_windows=(7,),
    )

    day_one = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-01")
        )
    ].iloc[0]

    day_seven = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-07")
        )
    ].iloc[0]

    day_eight = features.loc[
        (
            features["product_id"] == "P001"
        )
        & (
            features["date"]
            == pd.Timestamp("2026-01-08")
        )
    ].iloc[0]

    assert pd.isna(day_one["lag_1"])
    assert pd.isna(day_one["lag_7"])
    assert pd.isna(day_one["rolling_mean_7"])
    assert pd.isna(day_seven["rolling_mean_7"])

    assert day_eight["lag_1"] == 7
    assert day_eight["rolling_mean_7"] == 4


def test_drop_incomplete_feature_rows(
    sample_sales: pd.DataFrame,
) -> None:
    """Rows without complete feature history should be removed."""

    features = create_time_series_features(
        sales=sample_sales,
        drop_incomplete_rows=True,
    )

    generated_columns = [
        column
        for column in features.columns
        if (
            column.startswith("lag_")
            or column.startswith(
                "rolling_mean_"
            )
        )
    ]

    expected_rows_per_product = 35 - 28
    expected_rows = (
        2 * expected_rows_per_product
    )

    assert len(features) == expected_rows

    assert features["date"].min() == (
        pd.Timestamp("2026-01-29")
    )

    assert not features[
        generated_columns
    ].isna().any().any()


def test_feature_output_is_sorted(
    sample_sales: pd.DataFrame,
) -> None:
    """Feature data should be sorted by pair and date."""

    shuffled_sales = sample_sales.sample(
        frac=1,
        random_state=2026,
    ).reset_index(drop=True)

    features = create_time_series_features(
        sales=shuffled_sales,
        lag_days=(1,),
        rolling_windows=(2,),
    )

    actual_order = list(
        zip(
            features["store_id"],
            features["product_id"],
            features["date"],
        )
    )

    assert actual_order == sorted(actual_order)


def test_drop_incomplete_rows_must_be_boolean(
    sample_sales: pd.DataFrame,
) -> None:
    """The drop option should only accept booleans."""

    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        create_time_series_features(
            sales=sample_sales,
            drop_incomplete_rows="yes",
        )


def test_get_feature_data_columns() -> None:
    """Feature column names should match their settings."""

    columns = get_feature_data_columns(
        lag_days=(1, 2),
        rolling_windows=(3,),
    )

    assert columns[:4] == (
        "date",
        "store_id",
        "product_id",
        "quantity_sold",
    )

    assert all(
        column in columns
        for column in CALENDAR_FEATURE_COLUMNS
    )

    assert columns[-3:] == (
        "lag_1",
        "lag_2",
        "rolling_mean_3",
    )

    assert "revenue" not in columns
    assert "cost_of_goods_sold" not in columns
