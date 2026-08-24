"""Build January 2021 SO2 CAMS baselines and station predictor table."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

from build_eea_pilot_targets import selected_files
from build_station_predictors_2021 import (
    elevation,
    interpolate,
    land_cover,
    population,
    road_density,
)
from chromap_nl.rivm import build_monthly_targets, read_measurements, read_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_cams() -> tuple[Path, Path]:
    archive_path = Path("data/raw/cams/cams_ensemble_validated_reanalysis_2021_01_so2.zip")
    cropped_path = Path("data/processed/cams/cams_monthly_predictors_2021_01_so2.nc")
    baseline_path = Path("data/processed/baseline/cams_baseline_nl_500m_2021_01_so2.nc")
    cropped_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if ".so2." in name]
            if len(members) != 1:
                raise ValueError(f"expected one SO2 member, found {members}")
            source_path = Path(archive.extract(members[0], temporary))
        with xr.open_dataset(source_path) as source:
            subset = source.so2.sel(lon=slice(0.5, 10.5), lat=slice(49.0, 55.5))
            mean = subset.mean("time", skipna=True).load().expand_dims(
                time=[np.datetime64("2021-01-01")]
            )
    mean.name = "so2"
    mean.attrs.update({
        "long_name": "monthly mean mass concentration of sulphur dioxide in air",
        "units": "ug m-3", "aggregation": "calendar-month mean of hourly CAMS fields",
    })
    cropped = mean.to_dataset()
    cropped.attrs.update({
        "source_dataset": "cams-europe-air-quality-reanalyses",
        "source_archive": archive_path.name, "source_archive_sha256": sha256(archive_path),
        "reanalysis_type": "validated_reanalysis",
    })
    cropped.to_netcdf(cropped_path, engine="netcdf4",
                      encoding={"so2": {"zlib": True, "complevel": 4, "dtype": "float32"}})

    with xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc") as grid:
        field = mean.interp(lon=grid.longitude, lat=grid.latitude, method="linear")
        field = field.where(grid.land_mask == 1).astype("float32")
        field.attrs.update({"grid_mapping": "spatial_ref",
                            "processing_level": "CAMS 0.1-degree baseline interpolated to target grid"})
        baseline = xr.Dataset(
            {"so2": field, "land_mask": grid.land_mask.load(),
             "longitude": grid.longitude.load(), "latitude": grid.latitude.load(),
             "spatial_ref": grid.spatial_ref.load()},
            coords={"time": mean.time, "y": grid.y.load(), "x": grid.x.load()},
            attrs={"title": "CHROMAP-NL CAMS SO2 baseline on the 500 m target grid",
                   "target_crs": "EPSG:3035", "resolution_m": 500},
        )
    baseline.to_netcdf(baseline_path, engine="netcdf4", encoding={
        "so2": {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": -9999.0},
        "land_mask": {"zlib": True, "complevel": 5, "dtype": "uint8"},
        "longitude": {"zlib": True, "complevel": 5},
        "latitude": {"zlib": True, "complevel": 5},
    })
    return cropped_path, baseline_path


def rivm_targets() -> pd.DataFrame:
    metadata = read_metadata(
        "data/raw/rivm/luchtmeetnet_meetlocaties.csv",
        "data/raw/rivm/luchtmeetnet_meetreeksen.csv",
    )
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile("data/raw/rivm/2021.zip") as archive:
            path = Path(archive.extract("2021_SO2.csv", temporary))
        target = build_monthly_targets(read_measurements(path), metadata)
    target = target[(target.month == "2021-01") & target.is_valid & target.value.notna()].copy()
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    target["x"], target["y"] = transformer.transform(target.longitude, target.latitude)
    return pd.DataFrame({
        "source": "RIVM", "country": "NL", "station_id": target.meetlocatie_id,
        "sampling_id": target.meetreeks_id, "indicator": "so2",
        "month": pd.to_datetime(target.month), "value": target.value,
        "valid_days": target.valid_days, "expected_days": target.expected_days,
        "coverage": target.coverage, "is_valid": target.is_valid,
        "source_resolution": target.source_resolution, "station_type": target.stationstype,
        "area_type": target.gebiedstype, "station_name": target.meetlocatie_naam,
        "longitude": target.longitude, "latitude": target.latitude,
        "x": target.x, "y": target.y, "distance_to_nl_km": 0.0,
    })


def eea_targets() -> pd.DataFrame:
    root = Path("data/raw/eea_air_quality_so2")
    stations_ll = gpd.read_file("data/raw/eea_air_quality/station_locations.geojson")
    stations = stations_ll.to_crs(3035)
    stations["x"], stations["y"] = stations.geometry.x, stations.geometry.y
    stations["longitude"], stations["latitude"] = stations_ll.geometry.x.values, stations_ll.geometry.y.values
    with xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc") as grid:
        iy, ix = np.where(grid.land_mask.values == 1)
        distance, _ = cKDTree(np.column_stack((grid.x.values[ix], grid.y.values[iy]))).query(
            stations[["x", "y"]].values
        )
    stations["distance_to_nl_km"] = distance / 1000
    stations = stations[stations.CountryCode.isin(("BE", "DE", "FR", "LU"))
                        & (stations.distance_to_nl_km <= 150)].copy()
    paths, mapping = selected_files(root, stations)
    if not paths:
        return pd.DataFrame()
    connection = duckdb.connect()
    connection.read_parquet(paths, filename=True, union_by_name=True).create_view("raw_eea")
    means = connection.execute("""
        WITH obs AS (
          SELECT filename, Samplingpoint, Start AS start_time, CAST(Value AS DOUBLE) AS value,
                 AggType FROM raw_eea
          WHERE Start >= TIMESTAMP '2021-01-01' AND Start < TIMESTAMP '2021-02-01'
            AND CAST(Pollutant AS INTEGER) = 1 AND Validity = 1 AND Verification = 1
            AND Value IS NOT NULL
        ), daily_hour AS (
          SELECT filename, Samplingpoint, CAST(date_trunc('day', start_time) AS DATE) AS day,
                 avg(value) AS daily_value, count(*) AS valid_hours
          FROM obs WHERE AggType = 'hour' GROUP BY ALL
        ), hourly_month AS (
          SELECT filename, Samplingpoint, avg(daily_value) AS value, count(*) AS valid_days,
                 'hour' AS source_resolution FROM daily_hour WHERE valid_hours >= 18 GROUP BY ALL
        ), daily_source AS (
          SELECT filename, Samplingpoint, CAST(date_trunc('day', start_time) AS DATE) AS day,
                 avg(value) AS daily_value FROM obs WHERE AggType = 'day' GROUP BY ALL
        ), daily_month AS (
          SELECT filename, Samplingpoint, avg(daily_value) AS value, count(*) AS valid_days,
                 'day' AS source_resolution FROM daily_source GROUP BY ALL
        ), combined AS (SELECT * FROM hourly_month UNION ALL SELECT * FROM daily_month),
        ranked AS (
          SELECT *, row_number() OVER (PARTITION BY filename, Samplingpoint
            ORDER BY CASE source_resolution WHEN 'hour' THEN 1 ELSE 2 END) AS priority
          FROM combined
        )
        SELECT * FROM ranked WHERE priority = 1 AND valid_days::DOUBLE / 31 >= 0.75
    """).fetchdf()
    connection.close()
    if means.empty:
        return pd.DataFrame()
    means["filename_key"] = means.filename.map(lambda value: Path(value).resolve().as_posix())
    result = means.merge(mapping, on="filename_key", how="left")
    return pd.DataFrame({
        "source": "EEA-E1a", "country": result.country,
        "station_id": result.station_eoi_code, "sampling_id": result.Samplingpoint,
        "indicator": "so2", "month": pd.Timestamp("2021-01-01"), "value": result.value,
        "valid_days": result.valid_days, "expected_days": 31,
        "coverage": result.valid_days / 31, "is_valid": True,
        "source_resolution": result.source_resolution, "station_type": result.station_class,
        "area_type": np.nan, "station_name": result.station_name,
        "longitude": result.longitude, "latitude": result.latitude,
        "x": result.x, "y": result.y, "distance_to_nl_km": result.distance_to_nl_km,
    })


def add_predictors(targets: pd.DataFrame, cams_path: Path) -> pd.DataFrame:
    keys = ["source", "country", "station_id"]
    stations = targets.drop_duplicates(keys).reset_index(drop=True)
    stations["population_density_km2"] = population(
        stations, Path("data/raw/eurostat/Eurostat_Census-GRID_2021_V1-0.zip"))
    stations["log1p_population_density"] = np.log1p(stations.population_density_km2)
    stations["elevation"] = elevation(stations, Path("data/raw/copernicus_dem/glo30"))
    for name, values in land_cover(stations, Path("data/raw/esa_worldcover/2021/v200")
                                    if Path("data/raw/esa_worldcover/2021/v200").exists()
                                    else Path("data/raw/esa_worldcover/2021_v200")).items():
        stations[name] = values
    for name, values in road_density(stations, Path("data/raw/osm_geofabrik")).items():
        stations[name] = values
    result = targets.merge(stations, on=keys, suffixes=("", "_predictor"))
    result["month_number"], result["month_sin"], result["month_cos"] = 1, 0.5, np.cos(2 * np.pi / 12)
    with xr.open_dataset(cams_path) as cams:
        result["cams_so2"] = interpolate(cams.so2.isel(time=0), result.latitude.values,
                                          result.longitude.values)
    era_files = sorted(Path("data/raw/era5/era5_monthly_nl_buffer_2021").glob("*.nc"))
    era_parts = [xr.open_dataset(path) for path in era_files]
    try:
        for index, dataset in enumerate(era_parts):
            era_parts[index] = dataset.assign_coords(
                valid_time=dataset.valid_time.values.astype("datetime64[M]").astype("datetime64[ns]"))
        era = xr.merge(era_parts, join="exact", compat="override")
        for name in ("u10", "v10", "t2m", "blh", "sp", "tcc", "ssrd", "tp"):
            result[f"era5_{name}"] = interpolate(
                era[name].sel(valid_time="2021-01-01"), result.latitude.values,
                result.longitude.values)
    finally:
        for dataset in era_parts:
            dataset.close()
    result["era5_wind_speed"] = np.hypot(result.era5_u10, result.era5_v10)
    result["cams_value"] = result.cams_so2
    return result


def main() -> None:
    cropped, baseline = process_cams()
    targets = pd.concat([rivm_targets(), eea_targets()], ignore_index=True)
    targets = targets.sort_values(["source", "sampling_id", "coverage"], ascending=[True, True, False])
    targets = targets.drop_duplicates(["source", "sampling_id"], keep="first").reset_index(drop=True)
    target_path = Path("data/processed/observations/so2_station_targets_2021_01.csv.gz")
    targets.to_csv(target_path, index=False, compression="gzip")
    predictors = add_predictors(targets, cropped)
    predictor_path = Path("data/processed/model/so2_station_predictors_2021_01.csv.gz")
    predictors.to_csv(predictor_path, index=False, compression="gzip")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_rows": len(targets), "stations_by_source": targets.groupby("source").station_id.nunique().to_dict(),
        "cropped_cams": str(cropped), "baseline": str(baseline),
        "targets": str(target_path), "predictors": str(predictor_path),
        "sha256": {str(path): sha256(path) for path in (cropped, baseline, target_path, predictor_path)},
    }
    path = Path("data/processed/model/so2_inputs_2021_01_manifest.json")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
