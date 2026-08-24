import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chromap_nl.rivm import read_measurements


class RivmReaderTests(unittest.TestCase):
    def test_reads_hourly_air_values_and_preserves_negatives(self) -> None:
        content = """#export;test
meetreeks_id;meetlocatie_id;component;matrix;meetduur;eenheid;begindatumtijd;waarde
NL1_O3_lucht;NL1;O3;lucht;uur;µg/m³;2021-01-01T00:00:00+01:00;-1.5
NL1_O3_lucht;NL1;O3;lucht;dag;µg/m³;2021-01-02T00:00:00+01:00;20
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "o3.csv"
            path.write_text(content, encoding="utf-8")
            result = read_measurements(path)
        self.assertEqual(len(result), 2)
        hourly = result.loc[result["meetduur"] == "uur"].iloc[0]
        self.assertEqual(hourly["waarde"], -1.5)
        self.assertEqual(str(hourly["timestamp"]), "2020-12-31 23:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
