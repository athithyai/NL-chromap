"""Construct monthly modelling targets from validated hourly observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregationRules:
    """Completeness and indicator rules used for station aggregation."""

    minimum_valid_hours_per_day: int = 18
    minimum_valid_days_fraction_per_month: float = 0.75
    minimum_valid_hours_per_8h_window: int = 6
    somo35_threshold_ugm3: float = 70.0

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_valid_hours_per_day <= 24:
            raise ValueError("minimum_valid_hours_per_day must be in [1, 24]")
        if not 0 < self.minimum_valid_days_fraction_per_month <= 1:
            raise ValueError(
                "minimum_valid_days_fraction_per_month must be in (0, 1]"
            )
        if not 1 <= self.minimum_valid_hours_per_8h_window <= 8:
            raise ValueError("minimum_valid_hours_per_8h_window must be in [1, 8]")


def _prepare_hourly(values: pd.Series) -> pd.Series:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("values must use a DatetimeIndex")
    if values.index.tz is None:
        raise ValueError("hourly timestamps must be timezone-aware")

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.sort_index().tz_convert("UTC")
    if numeric.index.has_duplicates:
        numeric = numeric.groupby(level=0).mean()
    return numeric


def _monthly_frame(daily_values: pd.Series, rules: AggregationRules) -> pd.DataFrame:
    monthly_period = daily_values.index.tz_localize(None).to_period("M")
    value = daily_values.groupby(monthly_period).mean()
    valid_days = daily_values.notna().groupby(monthly_period).sum().astype(int)
    expected_days = pd.Series(
        value.index.days_in_month,
        index=value.index,
        dtype=int,
    )
    coverage = valid_days / expected_days
    is_valid = coverage >= rules.minimum_valid_days_fraction_per_month
    value = value.where(is_valid)

    result = pd.DataFrame(
        {
            "value": value,
            "valid_days": valid_days,
            "expected_days": expected_days,
            "coverage": coverage,
            "is_valid": is_valid,
        }
    )
    result.index.name = "month"
    return result


def monthly_mean(
    hourly_values: pd.Series,
    rules: AggregationRules = AggregationRules(),
) -> pd.DataFrame:
    """Calculate monthly concentration means after daily completeness filtering."""

    hourly = _prepare_hourly(hourly_values)
    daily_count = hourly.resample("D").count()
    daily_mean = hourly.resample("D").mean()
    daily_mean = daily_mean.where(
        daily_count >= rules.minimum_valid_hours_per_day
    )
    return _monthly_frame(daily_mean, rules)


def monthly_mean_from_daily(
    daily_values: pd.Series,
    rules: AggregationRules = AggregationRules(),
) -> pd.DataFrame:
    """Calculate monthly means from source-provided validated daily means."""

    if not isinstance(daily_values.index, pd.DatetimeIndex):
        raise TypeError("daily_values must use a DatetimeIndex")
    values = pd.to_numeric(daily_values, errors="coerce").astype(float).sort_index()
    if values.index.has_duplicates:
        values = values.groupby(level=0).mean()
    if values.index.tz is not None:
        values.index = values.index.tz_localize(None)
    return _monthly_frame(values, rules)


def monthly_somo35(
    hourly_o3_ugm3: pd.Series,
    rules: AggregationRules = AggregationRules(),
) -> pd.DataFrame:
    """Calculate monthly SOMO35 contributions from hourly O3 in micrograms/m3.

    Daily contributions are max(daily maximum running 8-hour mean - 70, 0).
    Monthly values are unscaled sums of valid daily contributions.
    """

    hourly = _prepare_hourly(hourly_o3_ugm3)
    hourly_count = hourly.resample("D").count()
    rolling_8h = hourly.rolling(
        window=8,
        min_periods=rules.minimum_valid_hours_per_8h_window,
    ).mean()
    daily_max_8h = rolling_8h.resample("D").max()
    daily_max_8h = daily_max_8h.where(
        hourly_count >= rules.minimum_valid_hours_per_day
    )
    daily_contribution = (daily_max_8h - rules.somo35_threshold_ugm3).clip(
        lower=0
    )

    monthly_period = daily_contribution.index.tz_localize(None).to_period("M")
    value = daily_contribution.groupby(monthly_period).sum(min_count=1)
    valid_days = daily_contribution.notna().groupby(monthly_period).sum().astype(int)
    expected_days = pd.Series(
        value.index.days_in_month,
        index=value.index,
        dtype=int,
    )
    coverage = valid_days / expected_days
    is_valid = coverage >= rules.minimum_valid_days_fraction_per_month
    value = value.where(is_valid)

    result = pd.DataFrame(
        {
            "value": value,
            "valid_days": valid_days,
            "expected_days": expected_days,
            "coverage": coverage,
            "is_valid": is_valid,
        }
    )
    result.index.name = "month"
    return result
