"""Assemble the paper-defined CHROMAP NO2 feature stack for January 2021."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio.features
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import box


FEATURES = [
    "CHEM_CTM_no2", "CHEM_Sat_no2", "MET_Precip", "MET_Rad", "MET_Pbl",
    "MET_Winds", "MET_Temp", "MET_SurfP", "MET_CloudC", "GEO_Crop",
    "GEO_Tree", "GEO_Grass", "GEO_Builtup", "GEO_Shrub", "GEO_Bare",
    "GEO_Lat", "GEO_Lon", "GEO_Alt", "GEO_DistCoast", "GEO_RoadNet",
    "GEO_PopDens", "GEO_UrbDeg",
]
STATIC = [name for name in FEATURES if name.startswith("GEO_") and name != "GEO_RoadNet"]
ROAD_TYPES = {"motorway", "trunk", "primary", "secondary"}
PBF = {
    "NL": ["netherlands-latest.osm.pbf"], "BE": ["belgium-latest.osm.pbf"],
    "DE": ["niedersachsen-latest.osm.pbf", "nordrhein-westfalen-latest.osm.pbf", "rheinland-pfalz-latest.osm.pbf"],
    "FR": ["nord-pas-de-calais-latest.osm.pbf"], "LU": ["luxembourg-latest.osm.pbf"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(field: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    latitude = field.latitude.values if "latitude" in field.coords else field.lat.values
    longitude = field.longitude.values if "longitude" in field.coords else field.lon.values
    values = np.asarray(field.values).squeeze()
    if latitude[0] > latitude[-1]:
        latitude, values = latitude[::-1], values[::-1]
    return RegularGridInterpolator((latitude, longitude), values, bounds_error=False,
                                   fill_value=np.nan)(np.column_stack((lat, lon)))


def sample_static(static: xr.Dataset, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    points = {"x": xr.DataArray(x, dims="point"), "y": xr.DataArray(y, dims="point")}
    return {name: static[name].sel(points, method="nearest").values for name in STATIC}


def road_at_stations(stations: pd.DataFrame, directory: Path) -> np.ndarray:
    """Approximate total selected centreline length inside each 500 m cell."""
    output = np.zeros(len(stations), dtype="float32")
    where = "highway IN (" + ",".join(f"'{item}'" for item in sorted(ROAD_TYPES)) + ")"
    for country, indices in stations.groupby("country").groups.items():
        subset = stations.loc[indices]
        bbox = (subset.longitude.min() - .02, subset.latitude.min() - .02,
                subset.longitude.max() + .02, subset.latitude.max() + .02)
        frames = []
        for filename in PBF[country]:
            frame = pyogrio.read_dataframe(directory / filename, layer="lines",
                                            columns=["highway"], where=where, bbox=bbox)
            if len(frame):
                frames.append(frame)
        if not frames:
            continue
        roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(3035)
        spatial_index = roads.sindex
        for index in indices:
            row = stations.loc[index]
            left, bottom = np.floor(row.x / 500) * 500, np.floor(row.y / 500) * 500
            transform = from_origin(left, bottom + 500, 25, 25)
            bounds = box(left, bottom, left + 500, bottom + 500)
            candidates = roads.iloc[spatial_index.query(bounds, predicate="intersects")]
            burned = rasterio.features.rasterize(
                ((geometry, 1) for geometry in candidates.geometry), out_shape=(20, 20),
                transform=transform, fill=0, dtype="uint8", all_touched=False,
            )
            output[index] = burned.sum() * 25 / 1000  # km per 500 m cell
    return output


def road_on_grid(grid: xr.Dataset, pbf: Path) -> np.ndarray:
    bbox = (float(np.nanmin(grid.longitude)) - .05, float(np.nanmin(grid.latitude)) - .05,
            float(np.nanmax(grid.longitude)) + .05, float(np.nanmax(grid.latitude)) + .05)
    where = "highway IN (" + ",".join(f"'{item}'" for item in sorted(ROAD_TYPES)) + ")"
    roads = pyogrio.read_dataframe(pbf, layer="lines", columns=["highway"], where=where,
                                   bbox=bbox).to_crs(3035)
    factor, resolution = 20, 25
    coarse = (grid.sizes["y"], grid.sizes["x"])
    transform = from_origin(float(grid.x.min()) - 250, float(grid.y.max()) + 250,
                            resolution, resolution)
    burned = rasterio.features.rasterize(
        ((geometry, 1) for geometry in roads.geometry),
        out_shape=(coarse[0] * factor, coarse[1] * factor), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    return np.flipud(burned.reshape(coarse[0], factor, coarse[1], factor).sum(axis=(1, 3))).astype("float32") * resolution / 1000


def main() -> None:
    output_dir = Path("data/processed/model/paper_no2_2021_01")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv("data/processed/model/station_predictors_2021.csv.gz", parse_dates=["month"])
    stations = base[(base.indicator == "no2") & (base.month.dt.month == 1)
                    & base.value.notna() & base.is_valid].copy().reset_index(drop=True)
    grid = xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc")
    mask = grid.land_mask.values == 1
    gx, gy = np.meshgrid(grid.x.values, grid.y.values)
    grid_lon, grid_lat = grid.longitude.values, grid.latitude.values
    static_path = Path("data/processed/predictors/chromap_static_buffer_500m_2021.nc")
    static = xr.open_dataset(static_path)
    for name, values in sample_static(static, stations.x.values, stations.y.values).items():
        stations[name] = values
    static_grid = sample_static(static, gx.ravel(), gy.ravel())

    cams = xr.open_dataset("data/processed/cams/cams_monthly_predictors_2021_01.nc")
    satellite = xr.open_dataset("data/processed/tropomi/tropomi_total_no2_2021_01.nc")
    stations["CHEM_CTM_no2"] = regular(cams.no2.isel(time=0), stations.latitude.values, stations.longitude.values)
    stations["CHEM_Sat_no2"] = regular(satellite.CHEM_Sat_no2, stations.latitude.values, stations.longitude.values)
    dynamic_grid: dict[str, np.ndarray] = {
        "CHEM_CTM_no2": regular(cams.no2.isel(time=0), grid_lat.ravel(), grid_lon.ravel()),
        "CHEM_Sat_no2": regular(satellite.CHEM_Sat_no2, grid_lat.ravel(), grid_lon.ravel()),
    }

    era_parts = []
    for path in sorted(Path("data/raw/era5/era5_monthly_nl_buffer_2021").glob("*.nc")):
        part = xr.open_dataset(path)
        part = part.assign_coords(valid_time=part.valid_time.values.astype("datetime64[M]").astype("datetime64[ns]"))
        era_parts.append(part)
    era = xr.merge(era_parts, join="exact", compat="override")
    era_map = {"MET_Precip": "tp", "MET_Rad": "ssrd", "MET_Pbl": "blh"}
    for feature, source in era_map.items():
        field = era[source].sel(valid_time="2021-01-01", method="nearest")
        stations[feature] = regular(field, stations.latitude.values, stations.longitude.values)
        dynamic_grid[feature] = regular(field, grid_lat.ravel(), grid_lon.ravel())

    # CERRA uses a curvilinear grid. Nearest 5.5 km source cells are sufficient
    # because these meteorological predictors are much coarser than the 500 m target.
    cerra = xr.open_dataset("data/raw/cerra/cerra_chromap_2021_01.nc")
    tree = cKDTree(np.column_stack((cerra.latitude.values.ravel(), cerra.longitude.values.ravel())))
    station_index = tree.query(np.column_stack((stations.latitude, stations.longitude)))[1]
    grid_index = tree.query(np.column_stack((grid_lat.ravel(), grid_lon.ravel())))[1]
    cerra_map = {"MET_Winds": "si10", "MET_Temp": "t2m", "MET_SurfP": "sp", "MET_CloudC": "tcc"}
    for feature, source in cerra_map.items():
        mean = cerra[source].mean("valid_time").values.ravel()
        stations[feature] = mean[station_index]
        dynamic_grid[feature] = mean[grid_index]
        print(f"processed {feature}", flush=True)

    stations["GEO_RoadNet"] = road_at_stations(stations, Path("data/raw/osm_geofabrik"))
    road_grid = road_on_grid(grid, Path("data/raw/osm_geofabrik/netherlands-latest.osm.pbf"))
    station_path = output_dir / "paper_no2_station_features_2021_01.csv.gz"
    stations.to_csv(station_path, index=False, compression="gzip")

    variables = {}
    for name in FEATURES:
        if name == "GEO_RoadNet":
            values = road_grid
        elif name in dynamic_grid:
            values = dynamic_grid[name].reshape(mask.shape)
        else:
            values = static_grid[name].reshape(mask.shape)
        variables[name] = (("y", "x"), np.where(mask, values, np.nan).astype("float32"))
    output = xr.Dataset(variables, coords={"y": grid.y.values, "x": grid.x.values}, attrs={
        "title": "Paper-defined CHROMAP NO2 predictors, January 2021 monthly adaptation",
        "feature_order": json.dumps(FEATURES), "target_crs": "EPSG:3035", "resolution_m": 500,
        "deviations": "January aggregation instead of 2019-2023 annual; GLO-30 instead of GLO-90; 2026 OSM snapshot; Natural Earth coastline",
    })
    grid_path = output_dir / "paper_no2_grid_features_2021_01.nc"
    output.to_netcdf(grid_path, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0} for name in FEATURES})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "features": FEATURES,
        "station_rows": len(stations), "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (station_path, grid_path, static_path)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
