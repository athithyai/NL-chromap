import unittest

import numpy as np
import pandas as pd

from chromap_nl.targets import (
    AggregationRules,
    monthly_mean,
    monthly_mean_from_daily,
    monthly_somo35,
)


class MonthlyTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = AggregationRules()

    def test_constant_monthly_mean(self) -> None:
        index = pd.date_range(
            "2021-01-01", "2021-02-01", freq="h", inclusive="left", tz="UTC"
        )
        result = monthly_mean(pd.Series(42.0, index=index), self.rules)
        self.assertAlmostEqual(result.loc[pd.Period("2021-01"), "value"], 42.0)
        self.assertTrue(result.loc[pd.Period("2021-01"), "is_valid"])

    def test_constant_o3_somo35(self) -> None:
        index = pd.date_range(
            "2021-01-01", "2021-02-01", freq="h", inclusive="left", tz="UTC"
        )
        result = monthly_somo35(pd.Series(80.0, index=index), self.rules)
        self.assertAlmostEqual(result.loc[pd.Period("2021-01"), "value"], 310.0)
        self.assertEqual(result.loc[pd.Period("2021-01"), "valid_days"], 31)

    def test_incomplete_month_is_rejected(self) -> None:
        index = pd.date_range(
            "2021-01-01", "2021-01-11", freq="h", inclusive="left", tz="UTC"
        )
        result = monthly_mean(pd.Series(20.0, index=index), self.rules)
        self.assertFalse(result.loc[pd.Period("2021-01"), "is_valid"])
        self.assertTrue(np.isnan(result.loc[pd.Period("2021-01"), "value"]))

    def test_naive_timestamps_are_rejected(self) -> None:
        index = pd.date_range("2021-01-01", periods=24, freq="h")
        with self.assertRaises(ValueError):
            monthly_mean(pd.Series(10.0, index=index), self.rules)

    def test_monthly_mean_from_daily_values(self) -> None:
        index = pd.date_range("2021-01-01", "2021-01-31", freq="D")
        result = monthly_mean_from_daily(pd.Series(12.0, index=index), self.rules)
        self.assertAlmostEqual(result.loc[pd.Period("2021-01"), "value"], 12.0)
        self.assertEqual(result.loc[pd.Period("2021-01"), "valid_days"], 31)


if __name__ == "__main__":
    unittest.main()
