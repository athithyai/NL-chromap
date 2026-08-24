"""Harmonize Dutch RIVM and cross-border EEA targets for the 2021 pilot."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pandas as pd
from pyproj import Transformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rivm = pd.read_csv("data/processed/observations/rivm_station_month_targets_2013_2025.csv.gz")
    rivm = rivm[rivm.month.astype(str).str.startswith("2021-")].copy()
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    rivm["x"], rivm["y"] = transformer.transform(rivm.longitude.values, rivm.latitude.values)
    nl = pd.DataFrame({
        "source": "RIVM", "country": "NL", "station_id": rivm.meetlocatie_id,
        "sampling_id": rivm.meetreeks_id, "indicator": rivm.indicator.str.lower(),
        "month": pd.to_datetime(rivm.month), "value": rivm.value,
        "valid_days": rivm.valid_days, "expected_days": rivm.expected_days,
        "coverage": rivm.coverage, "is_valid": rivm.is_valid,
        "source_resolution": rivm.source_resolution, "station_type": rivm.stationstype,
        "area_type": rivm.gebiedstype, "station_name": rivm.meetlocatie_naam,
        "longitude": rivm.longitude, "latitude": rivm.latitude,
        "x": rivm.x, "y": rivm.y, "distance_to_nl_km": 0.0,
    })

    eea = pd.read_csv("data/processed/observations/eea_crossborder_station_month_targets_2021.csv.gz")
    common = {
        "source": "EEA-E1a", "country": eea.country, "station_id": eea.station_eoi_code,
        "sampling_id": eea.Samplingpoint, "month": pd.to_datetime(eea.month),
        "expected_days": eea.expected_days, "station_type": eea.station_class,
        "area_type": np.nan, "station_name": eea.station_name,
        "longitude": eea.longitude, "latitude": eea.latitude, "x": eea.x, "y": eea.y,
        "distance_to_nl_km": eea.distance_to_nl_km,
    }
    means = pd.DataFrame({
        **common, "indicator": eea.pollutant_name.str.lower().replace({"pm2.5": "pm2p5"}), "value": eea.value,
        "valid_days": eea.valid_days, "coverage": eea.coverage,
        "is_valid": eea.is_valid, "source_resolution": eea.source_aggregation,
    })
    ozone = eea[eea.pollutant_name == "O3"].copy()
    somo = pd.DataFrame({
        "source": "EEA-E1a", "country": ozone.country,
        "station_id": ozone.station_eoi_code, "sampling_id": ozone.Samplingpoint,
        "indicator": "somo35", "month": pd.to_datetime(ozone.month), "value": ozone.somo35,
        "valid_days": np.rint(ozone.somo35_coverage * ozone.expected_days),
        "expected_days": ozone.expected_days, "coverage": ozone.somo35_coverage,
        "is_valid": ozone.somo35_is_valid.fillna(False), "source_resolution": "hour",
        "station_type": ozone.station_class, "area_type": np.nan,
        "station_name": ozone.station_name, "longitude": ozone.longitude,
        "latitude": ozone.latitude, "x": ozone.x, "y": ozone.y,
        "distance_to_nl_km": ozone.distance_to_nl_km,
    })
    result = pd.concat([nl, means, somo], ignore_index=True)
    # Ratified annual RIVM archives can overlap at a year boundary. Prefer a
    # valid/non-missing row and then the greatest reported coverage.
    result["_has_value"] = result.value.notna()
    result = result.sort_values(
        ["source", "sampling_id", "indicator", "month", "is_valid", "_has_value", "coverage"],
        ascending=[True, True, True, True, False, False, False],
    ).drop_duplicates(["source", "sampling_id", "indicator", "month"], keep="first")
    result = result.drop(columns="_has_value")
    result = result.sort_values(["indicator", "month", "country", "station_id", "sampling_id"])
    output = Path("data/processed/observations/combined_station_month_targets_2021.csv.gz")
    result.to_csv(output, index=False, compression="gzip")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(result),
        "valid_rows": int(result.value.notna().sum()),
        "stations_by_source": result.groupby("source").station_id.nunique().to_dict(),
        "rows_by_indicator": result.groupby("indicator").size().to_dict(),
        "output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
