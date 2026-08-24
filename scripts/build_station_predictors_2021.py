"""Extract the 2021 pilot predictor stack at RIVM and EEA stations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import rasterio.features
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import box


ROAD_CLASSES = {
    "major": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"},
    "secondary": {"secondary", "secondary_link", "tertiary", "tertiary_link"},
    "local": {"residential", "living_street", "unclassified", "service", "road"},
}
PBF = {
    "NL": ["netherlands-latest.osm.pbf"], "BE": ["belgium-latest.osm.pbf"],
    "DE": ["niedersachsen-latest.osm.pbf", "nordrhein-westfalen-latest.osm.pbf", "rheinland-pfalz-latest.osm.pbf"],
    "FR": ["nord-pas-de-calais-latest.osm.pbf"], "LU": ["luxembourg-latest.osm.pbf"],
}
WORLD_COVER = {
    "tree": (10,), "shrub": (20,), "grass": (30,), "cropland": (40,),
    "built_up": (50,), "bare_sparse": (60,), "permanent_water": (80,), "wetland": (90, 95),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(field: xr.DataArray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat = field.lat.values if "lat" in field.coords else field.latitude.values
    lon = field.lon.values if "lon" in field.coords else field.longitude.values
    data = np.asarray(field.values).squeeze()
    if lat[0] > lat[-1]:
        lat, data = lat[::-1], data[::-1, :]
    return RegularGridInterpolator((lat, lon), data, bounds_error=False, fill_value=np.nan)(
        np.column_stack((latitudes, longitudes))
    )


def population(stations: pd.DataFrame, archive: Path) -> np.ndarray:
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if name.lower().endswith(".tiff"))
    uri = f"zip://{archive.resolve()}!{member}"
    with rasterio.open(uri) as source:
        values = np.array([value[0] for value in source.sample(stations[["x", "y"]].values)])
    return np.where(values >= 0, values, 0).astype("float32")


def elevation(stations: pd.DataFrame, directory: Path) -> np.ndarray:
    paths: dict[tuple[int, int], Path] = {}
    pattern = re.compile(r"_N(\d+)_00_E(\d+)_00_DEM")
    for path in directory.glob("*.tif"):
        match = pattern.search(path.stem)
        if match:
            paths[(int(match.group(1)), int(match.group(2)))] = path
    result = np.full(len(stations), np.nan, dtype="float32")
    for key, indices in stations.groupby([stations.latitude.astype(int), stations.longitude.astype(int)]).groups.items():
        path = paths.get(tuple(key))
        if path is None:
            continue
        with rasterio.open(path) as source:
            points = stations.loc[indices, ["longitude", "latitude"]].values
            result[indices] = [value[0] for value in source.sample(points)]
    return result


def land_cover(stations: pd.DataFrame, directory: Path) -> dict[str, np.ndarray]:
    paths: dict[tuple[int, int], Path] = {}
    pattern = re.compile(r"_N(\d+)E(\d+)_Map")
    for path in directory.glob("*.tif"):
        match = pattern.search(path.stem)
        if match:
            paths[(int(match.group(1)), int(match.group(2)))] = path
    to_lonlat = Transformer.from_crs(3035, 4326, always_xy=True)
    codes = np.zeros((len(stations), 25), dtype="uint8")
    for position, (_, station) in enumerate(stations.iterrows()):
        left = np.floor(station.x / 500) * 500
        bottom = np.floor(station.y / 500) * 500
        xx, yy = np.meshgrid(left + np.arange(50, 500, 100), bottom + np.arange(50, 500, 100))
        lon, lat = to_lonlat.transform(xx.ravel(), yy.ravel())
        groups: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
        for sample_index, (sample_lon, sample_lat) in enumerate(zip(lon, lat)):
            key = (int(np.floor(sample_lat / 3) * 3), int(np.floor(sample_lon / 3) * 3))
            groups.setdefault(key, []).append((sample_index, sample_lon, sample_lat))
        for key, samples in groups.items():
            path = paths.get(key)
            if path is None:
                continue
            with rasterio.open(path) as source:
                values = source.sample([(item[1], item[2]) for item in samples])
                for item, value in zip(samples, values):
                    codes[position, item[0]] = value[0]
    return {f"landcover_{name}_fraction": np.isin(codes, values).mean(axis=1).astype("float32")
            for name, values in WORLD_COVER.items()}


def road_density(stations: pd.DataFrame, directory: Path) -> dict[str, np.ndarray]:
    output = {name: np.full(len(stations), np.nan, dtype="float32") for name in ROAD_CLASSES}
    all_types = sorted(set().union(*ROAD_CLASSES.values()))
    where = "highway IN (" + ",".join(f"'{value}'" for value in all_types) + ")"
    for country, station_indices in stations.groupby("country").groups.items():
        subset = stations.loc[station_indices]
        bbox = (subset.longitude.min() - 0.02, subset.latitude.min() - 0.02,
                subset.longitude.max() + 0.02, subset.latitude.max() + 0.02)
        frames = []
        for filename in PBF[country]:
            frame = pyogrio.read_dataframe(
                directory / filename, layer="lines", columns=["highway"], where=where, bbox=bbox
            )
            if len(frame):
                frames.append(frame)
        if not frames:
            continue
        roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(3035)
        spatial_index = roads.sindex
        for index in station_indices:
            station = stations.loc[index]
            left = np.floor(station.x / 500) * 500
            bottom = np.floor(station.y / 500) * 500
            cell = box(left, bottom, left + 500, bottom + 500)
            candidates = roads.iloc[spatial_index.query(cell, predicate="intersects")]
            transform = from_origin(left, bottom + 500, 50, 50)
            for name, types in ROAD_CLASSES.items():
                selected = candidates[candidates.highway.isin(types)]
                burned = rasterio.features.rasterize(
                    ((geometry, 1) for geometry in selected.geometry), out_shape=(10, 10),
                    transform=transform, fill=0, dtype="uint8", all_touched=False,
                )
                output[name][index] = burned.sum() * 50 / 1000 / 0.25
    output["motorized_total"] = sum(output.values()).astype("float32")
    return {f"road_density_{name}": values for name, values in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("data/processed/observations/combined_station_month_targets_2021.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model/station_predictors_2021.csv.gz"))
    args = parser.parse_args()
    targets = pd.read_csv(args.targets, parse_dates=["month"])
    station_keys = ["source", "country", "station_id"]
    stations = targets.drop_duplicates(station_keys).reset_index(drop=True)
    stations["population_density_km2"] = population(
        stations, Path("data/raw/eurostat/Eurostat_Census-GRID_2021_V1-0.zip")
    )
    stations["log1p_population_density"] = np.log1p(stations.population_density_km2)
    stations["elevation"] = elevation(stations, Path("data/raw/copernicus_dem/glo30"))
    for name, values in land_cover(stations, Path("data/raw/esa_worldcover/2021_v200")).items():
        stations[name] = values
    for name, values in road_density(stations, Path("data/raw/osm_geofabrik")).items():
        stations[name] = values

    result = targets.merge(stations[station_keys + [column for column in stations if column not in targets.columns]], on=station_keys)
    result["month_number"] = result.month.dt.month
    result["month_sin"] = np.sin(2 * np.pi * result.month_number / 12)
    result["month_cos"] = np.cos(2 * np.pi * result.month_number / 12)

    era_files = sorted(Path("data/raw/era5/era5_monthly_nl_buffer_2021").glob("*.nc"))
    era_parts = [xr.open_dataset(path) for path in era_files]
    for index, dataset in enumerate(era_parts):
        era_parts[index] = dataset.assign_coords(
            valid_time=dataset.valid_time.values.astype("datetime64[M]").astype("datetime64[ns]")
        )
    era = xr.merge(era_parts, join="exact", compat="override")
    for month in range(1, 13):
        rows = result.index[result.month_number == month]
        lat, lon = result.loc[rows, "latitude"].values, result.loc[rows, "longitude"].values
        cams = xr.open_dataset(f"data/processed/cams/cams_monthly_predictors_2021_{month:02d}.nc")
        for name in ("no2", "o3", "pm10", "pm2p5", "somo35"):
            result.loc[rows, f"cams_{name}"] = interpolate(cams[name].isel(time=0), lat, lon)
        for name in ("u10", "v10", "t2m", "blh", "sp", "tcc", "ssrd", "tp"):
            result.loc[rows, f"era5_{name}"] = interpolate(
                era[name].sel(valid_time=f"2021-{month:02d}-01"), lat, lon
            )
        cams.close()
    result["era5_wind_speed"] = np.hypot(result.era5_u10, result.era5_v10)
    result["cams_value"] = result.apply(lambda row: row.get(f"cams_{row.indicator}"), axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "rows": len(result),
        "stations": len(stations), "columns": result.columns.tolist(),
        "missing_by_predictor": {column: int(result[column].isna().sum()) for column in result.columns if column.startswith(("cams_", "era5_", "landcover_", "road_"))},
        "output": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
