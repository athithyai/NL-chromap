"""Build 2021 cross-border station-month targets from verified EEA Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


COUNTRIES = ("BE", "DE", "FR", "LU")
POLLUTANTS = {5: "PM10", 7: "O3", 8: "NO2", 6001: "PM2.5"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(root: Path, stations: gpd.GeoDataFrame) -> tuple[list[str], pd.DataFrame]:
    records: list[dict[str, object]] = []
    for country in COUNTRIES:
        subset = stations[stations.CountryCode == country]
        code_rows = [(str(row.AirQualityStationEoICode), row) for _, row in subset.iterrows()]
        for path in (root / country).glob("*.parquet"):
            hits = [(code, row) for code, row in code_rows if code and code in path.stem]
            if not hits:
                continue
            code, row = max(hits, key=lambda item: len(item[0]))
            records.append({
                "filename_key": path.resolve().as_posix(), "station_eoi_code": code,
                "country": country, "station_name": row.AQStationName,
                "station_class": row.stationClass, "longitude": row.longitude,
                "latitude": row.latitude, "x": row.x, "y": row.y,
                "distance_to_nl_km": row.distance_to_nl_km,
            })
    mapping = pd.DataFrame(records).drop_duplicates("filename_key")
    return mapping.filename_key.tolist(), mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--buffer-km", type=float, default=150)
    parser.add_argument("--root", type=Path, default=Path("data/raw/eea_air_quality"))
    parser.add_argument("--grid", type=Path, default=Path("data/processed/grid/chromap_nl_500m_grid.nc"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/observations/eea_crossborder_station_month_targets_2021.csv.gz"))
    args = parser.parse_args()

    stations_ll = gpd.read_file(args.root / "station_locations.geojson")
    stations = stations_ll.to_crs(3035)
    stations["x"] = stations.geometry.x
    stations["y"] = stations.geometry.y
    stations["longitude"] = stations_ll.geometry.x.values
    stations["latitude"] = stations_ll.geometry.y.values
    grid = xr.open_dataset(args.grid)
    iy, ix = np.where(grid.land_mask.values == 1)
    tree = cKDTree(np.column_stack((grid.x.values[ix], grid.y.values[iy])))
    distance, _ = tree.query(stations[["x", "y"]].values)
    stations["distance_to_nl_km"] = distance / 1000
    stations = stations[
        stations.CountryCode.isin(COUNTRIES)
        & (stations.distance_to_nl_km <= args.buffer_km)
    ].copy()
    paths, mapping = selected_files(args.root, stations)
    if not paths:
        raise RuntimeError("No EEA Parquet files matched buffered stations")

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='4GB'")
    relation = connection.read_parquet(paths, filename=True, union_by_name=True)
    relation.create_view("raw_eea")
    connection.execute(f"""
        CREATE TEMP VIEW obs AS
        SELECT filename, Samplingpoint, CAST(Pollutant AS INTEGER) AS pollutant,
               Start AS start_time, CAST(Value AS DOUBLE) AS value, Unit, AggType
        FROM raw_eea
        WHERE Start >= TIMESTAMP '{args.year - 1}-12-31 17:00:00'
          AND Start < TIMESTAMP '{args.year + 1}-01-01 00:00:00'
          AND Validity = 1 AND Verification = 1 AND Value IS NOT NULL
          AND CAST(Pollutant AS INTEGER) IN (5, 7, 8, 6001)
    """)

    means = connection.execute(f"""
        WITH daily_hour AS (
            SELECT filename, Samplingpoint, pollutant, CAST(date_trunc('day', start_time) AS DATE) AS day,
                   avg(value) AS daily_value, count(*) AS valid_hours
            FROM obs WHERE AggType = 'hour'
            GROUP BY ALL
        ), hourly_month AS (
            SELECT filename, Samplingpoint, pollutant, date_trunc('month', day) AS month,
                   avg(daily_value) AS value, count(*) AS valid_days,
                   day(last_day(month)) AS expected_days, 'hour' AS source_aggregation
            FROM daily_hour WHERE valid_hours >= 18 AND year(day) = {args.year}
            GROUP BY filename, Samplingpoint, pollutant, month
        ), daily_source AS (
            SELECT filename, Samplingpoint, pollutant, CAST(date_trunc('day', start_time) AS DATE) AS day,
                   avg(value) AS daily_value
            FROM obs WHERE AggType = 'day' AND year(start_time) = {args.year}
            GROUP BY ALL
        ), daily_month AS (
            SELECT filename, Samplingpoint, pollutant, date_trunc('month', day) AS month,
                   avg(daily_value) AS value, count(*) AS valid_days,
                   day(last_day(month)) AS expected_days, 'day' AS source_aggregation
            FROM daily_source GROUP BY filename, Samplingpoint, pollutant, month
        ), combined AS (
            SELECT * FROM hourly_month UNION ALL SELECT * FROM daily_month
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY filename, Samplingpoint, pollutant, month
                ORDER BY CASE source_aggregation WHEN 'hour' THEN 1 ELSE 2 END
            ) AS priority
            FROM combined
        )
        SELECT *, valid_days::DOUBLE / expected_days AS coverage,
               (valid_days::DOUBLE / expected_days) >= 0.75 AS is_valid
        FROM ranked WHERE priority = 1
    """).fetchdf()

    somo = connection.execute(f"""
        WITH ozone AS (
            SELECT *,
                avg(value) OVER w AS mean_8h,
                count(value) OVER w AS count_8h
            FROM obs WHERE pollutant = 7 AND AggType = 'hour'
            WINDOW w AS (
                PARTITION BY filename, Samplingpoint ORDER BY start_time
                RANGE BETWEEN INTERVAL 7 HOURS PRECEDING AND CURRENT ROW
            )
        ), daily AS (
            SELECT filename, Samplingpoint, CAST(date_trunc('day', start_time) AS DATE) AS day,
                   max(mean_8h) FILTER (WHERE count_8h >= 6) AS daily_mda8,
                   count(value) AS valid_hours
            FROM ozone GROUP BY ALL
        ), monthly AS (
            SELECT filename, Samplingpoint, date_trunc('month', day) AS month,
                   sum(greatest(daily_mda8 - 70.0, 0.0)) AS somo35,
                   count(daily_mda8) FILTER (WHERE valid_hours >= 18) AS valid_days,
                   day(last_day(month)) AS expected_days
            FROM daily WHERE year(day) = {args.year} AND valid_hours >= 18
            GROUP BY filename, Samplingpoint, month
        )
        SELECT *, valid_days::DOUBLE / expected_days AS somo35_coverage,
               (valid_days::DOUBLE / expected_days) >= 0.75 AS somo35_is_valid
        FROM monthly
    """).fetchdf()
    connection.close()

    means["filename_key"] = means.filename.map(lambda value: Path(value).resolve().as_posix())
    means["pollutant_name"] = means.pollutant.map(POLLUTANTS)
    means.loc[~means.is_valid, "value"] = np.nan
    somo["filename_key"] = somo.filename.map(lambda value: Path(value).resolve().as_posix())
    somo.loc[~somo.somo35_is_valid, "somo35"] = np.nan
    result = means.merge(
        somo[["filename_key", "Samplingpoint", "month", "somo35", "somo35_coverage", "somo35_is_valid"]],
        on=["filename_key", "Samplingpoint", "month"], how="left",
    ).merge(mapping, on="filename_key", how="left")
    result = result.drop(columns=["filename", "filename_key", "priority"])
    result = result.sort_values(["country", "station_eoi_code", "pollutant_name", "month"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "year": args.year,
        "buffer_km": args.buffer_km, "input_file_count": len(paths),
        "station_count": int(result.station_eoi_code.nunique()), "rows": len(result),
        "valid_monthly_means": int(result.value.notna().sum()),
        "valid_somo35": int(result.somo35.notna().sum()),
        "output": str(args.output), "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
