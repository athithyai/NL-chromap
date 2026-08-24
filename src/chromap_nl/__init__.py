"""Monthly CHROMAP-style air-quality mapping for the Netherlands."""

from .targets import (
    AggregationRules,
    monthly_mean,
    monthly_mean_from_daily,
    monthly_somo35,
)

__all__ = [
    "AggregationRules",
    "monthly_mean",
    "monthly_mean_from_daily",
    "monthly_somo35",
]
