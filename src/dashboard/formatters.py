"""Formatting helpers for the Streamlit dashboard."""

from numbers import Real

import numpy as np
import pandas as pd


def validate_finite_number(
    value: Real,
    value_name: str = "value",
) -> float:
    """Validate and return a finite numerical value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            f"{value_name} must be a real number."
        )

    numerical_value = float(value)

    if not np.isfinite(numerical_value):
        raise ValueError(
            f"{value_name} must be finite."
        )

    return numerical_value


def validate_decimal_places(
    decimal_places: int,
) -> None:
    """Validate a decimal-place setting."""

    if (
        isinstance(decimal_places, bool)
        or not isinstance(decimal_places, int)
        or decimal_places < 0
    ):
        raise ValueError(
            "decimal_places must be a "
            "nonnegative integer."
        )


def format_integer(
    value: Real,
) -> str:
    """Format a numerical value as a whole number."""

    numerical_value = validate_finite_number(
        value
    )

    return f"{numerical_value:,.0f}"


def format_decimal(
    value: Real,
    decimal_places: int = 2,
) -> str:
    """Format a number with a fixed precision."""

    validate_decimal_places(
        decimal_places
    )

    numerical_value = validate_finite_number(
        value
    )

    return (
        f"{numerical_value:,.{decimal_places}f}"
    )


def format_percentage(
    value: Real,
    decimal_places: int = 2,
) -> str:
    """Format a decimal rate as a percentage."""

    validate_decimal_places(
        decimal_places
    )

    numerical_value = validate_finite_number(
        value
    )

    return (
        f"{numerical_value:.{decimal_places}%}"
    )


def format_currency(
    value: Real,
    currency: str = "VND",
    decimal_places: int = 0,
) -> str:
    """Format a numerical value as currency."""

    validate_decimal_places(
        decimal_places
    )

    if not isinstance(currency, str):
        raise TypeError(
            "currency must be a string."
        )

    normalized_currency = (
        currency.strip().upper()
    )

    if not normalized_currency:
        raise ValueError(
            "currency must not be empty."
        )

    numerical_value = validate_finite_number(
        value
    )

    formatted_value = (
        f"{numerical_value:,.{decimal_places}f}"
    )

    return (
        f"{formatted_value} "
        f"{normalized_currency}"
    )


def format_duration_minutes(
    value: Real,
) -> str:
    """Format a duration using hours and minutes."""

    numerical_value = validate_finite_number(
        value,
        value_name="duration",
    )

    if numerical_value < 0:
        raise ValueError(
            "duration must not be negative."
        )

    total_minutes = int(
        round(numerical_value)
    )

    hours, minutes = divmod(
        total_minutes,
        60,
    )

    if hours == 0:
        return f"{minutes} min"

    if minutes == 0:
        hour_label = (
            "hr"
            if hours == 1
            else "hrs"
        )

        return f"{hours} {hour_label}"

    hour_label = (
        "hr"
        if hours == 1
        else "hrs"
    )

    return (
        f"{hours} {hour_label} "
        f"{minutes} min"
    )


def format_identifier(
    identifier: str,
) -> str:
    """Convert an internal identifier into a label."""

    if not isinstance(identifier, str):
        raise TypeError(
            "identifier must be a string."
        )

    normalized_identifier = (
        identifier.strip()
    )

    if not normalized_identifier:
        raise ValueError(
            "identifier must not be empty."
        )

    return (
        normalized_identifier
        .replace("_", " ")
        .title()
    )


def dataframe_to_csv_bytes(
    data: pd.DataFrame,
) -> bytes:
    """Convert a DataFrame into UTF-8 CSV bytes."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    return data.to_csv(
        index=False,
    ).encode("utf-8-sig")