"""Align 2021 pilot raster predictors to the frozen CHROMAP-NL 500 m grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Avoid an unrelated PostgreSQL/PostGIS PROJ installation injected into PATH.
os.environ.pop("PROJ_LIB", None)

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.interpolate import RegularGridInterpolator


TARGET_CRS = CRS.from_epsg(3035)
WORLD_COVER = {
    "tree": (10,), "shrub": (20,), "grass": (30,), "cropland": (40,),
    "built_up": (50,), "bare_sparse": (60,), "snow_ice": (70,),
    "permanent_water": (80,), "wetland": (90, 95), "moss_lichen": (100,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def population(grid: xr.Dataset, transform, shape, source_zip: Path) -> np.ndarray:
    with zipfile.ZipFile(source_zip) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".tiff"))
    uri = f"zip://{source_zip.resolve()}!{member}"
    north_up = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(uri) as source:
        reproject(
            source=rasterio.band(source, 1), destination=north_up,
            src_transform=source.transform, src_crs=TARGET_CRS, src_nodata=source.nodata,
            dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    result = np.flipud(north_up)
    # Eurostat uses -1/nodata for cells without an observed population value.
    result = np.where(np.isfinite(result) & (result >= 0), result, 0).astype("float32")
    return np.where(grid.land_mask.values == 1, result, np.nan)


def elevation(grid: xr.Dataset, transform, shape, directory: Path) -> tuple[np.ndarray, int]:
    target_bounds = rasterio.transform.array_bounds(*shape, transform)
    total = np.zeros(shape, dtype="float64")
    count = np.zeros(shape, dtype="uint16")
    used = 0
    for path in sorted(directory.glob("*.tif")):
        with rasterio.open(path) as source:
            source_bounds = transform_bounds(source.crs, TARGET_CRS, *source.bounds, densify_pts=21)
            if not intersects(source_bounds, target_bounds):
                continue
            used += 1
            warped = np.full(shape, np.nan, dtype="float32")
            reproject(
                source=rasterio.band(source, 1), destination=warped,
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            valid = np.isfinite(warped)
            total[valid] += warped[valid]
            count[valid] += 1
    result = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
    result = np.flipud(result).astype("float32")
    return np.where(grid.land_mask.values == 1, result, np.nan), used


def land_cover(grid: xr.Dataset, transform, shape, directory: Path) -> tuple[dict[str, np.ndarray], int]:
    # A 100 m categorical mosaic provides 25 systematic samples per 500 m cell.
    factor = 5
    fine_shape = (shape[0] * factor, shape[1] * factor)
    fine_transform = from_origin(transform.c, transform.f, 100, 100)
    target_bounds = rasterio.transform.array_bounds(*fine_shape, fine_transform)
    mosaic = np.zeros(fine_shape, dtype="uint8")
    used = 0
    for path in sorted(directory.glob("*.tif")):
        with rasterio.open(path) as source:
            source_bounds = transform_bounds(source.crs, TARGET_CRS, *source.bounds, densify_pts=21)
            if not intersects(source_bounds, target_bounds):
                continue
            used += 1
            warped = np.zeros(fine_shape, dtype="uint8")
            reproject(
                source=rasterio.band(source, 1), destination=warped,
                src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
                dst_transform=fine_transform, dst_crs=TARGET_CRS, dst_nodata=0,
                resampling=Resampling.nearest,
            )
            valid = warped != 0
            mosaic[valid] = warped[valid]
    blocks = mosaic.reshape(shape[0], factor, shape[1], factor)
    output: dict[str, np.ndarray] = {}
    for name, codes in WORLD_COVER.items():
        fraction = np.isin(blocks, codes).mean(axis=(1, 3)).astype("float32")
        fraction = np.flipud(fraction)
        output[f"landcover_{name}_fraction"] = np.where(grid.land_mask.values == 1, fraction, np.nan)
    coverage = (blocks != 0).mean(axis=(1, 3)).astype("float32")
    output["landcover_sample_coverage"] = np.where(
        grid.land_mask.values == 1, np.flipud(coverage), np.nan
    )
    return output, used


def era5(grid: xr.Dataset, directory: Path) -> xr.Dataset:
    files = sorted(directory.glob("*.nc"))
    datasets = [xr.open_dataset(path) for path in files]
    # CDS stamps accumulated monthly means at 06:00 and instantaneous monthly
    # means at 00:00. Both represent the same calendar month for this product.
    for index, dataset in enumerate(datasets):
        month_start = dataset.valid_time.values.astype("datetime64[M]").astype("datetime64[ns]")
        datasets[index] = dataset.assign_coords(valid_time=month_start)
    source = xr.merge(datasets, compat="override", join="exact")
    target_points = np.column_stack((grid.latitude.values.ravel(), grid.longitude.values.ravel()))
    values: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {}
    for name, variable in source.data_vars.items():
        lat = source.latitude.values
        lon = source.longitude.values
        data = variable.values
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[:, ::-1, :]
        monthly = np.empty((data.shape[0], *grid.land_mask.shape), dtype="float32")
        for month in range(data.shape[0]):
            interpolator = RegularGridInterpolator(
                (lat, lon), data[month], method="linear", bounds_error=False, fill_value=np.nan
            )
            field = interpolator(target_points).reshape(grid.land_mask.shape).astype("float32")
            monthly[month] = np.where(grid.land_mask.values == 1, field, np.nan)
        values[name] = (("time", "y", "x"), monthly, {
            "long_name": str(variable.attrs.get("long_name", name)),
            "units": str(variable.attrs.get("units", "")),
            "source_step_type": str(variable.attrs.get("GRIB_stepType", "")),
        })
    values["wind_speed_10m"] = (
        ("time", "y", "x"), np.hypot(values["u10"][1], values["v10"][1]).astype("float32"),
        {"long_name": "10 metre wind speed", "units": "m s-1"},
    )
    output = xr.Dataset(
        data_vars={name: xr.DataArray(data, dims=dims, attrs=attrs) for name, (dims, data, attrs) in values.items()},
        coords={"time": source.valid_time.values, "y": grid.y.values, "x": grid.x.values},
        attrs={"source": "ERA5 monthly averaged reanalysis", "target_crs": "EPSG:3035"},
    )
    for dataset in datasets:
        dataset.close()
    return output


def write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(-9999)}
        for name in dataset.data_vars if np.issubdtype(dataset[name].dtype, np.floating)
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=Path("data/processed/grid/chromap_nl_500m_grid.nc"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/predictors"))
    args = parser.parse_args()
    grid = xr.open_dataset(args.grid)
    shape = (grid.sizes["y"], grid.sizes["x"])
    transform = from_origin(float(grid.x.min()) - 250, float(grid.y.max()) + 250, 500, 500)

    pop = population(grid, transform, shape, Path("data/raw/eurostat/Eurostat_Census-GRID_2021_V1-0.zip"))
    dem, dem_tiles = elevation(grid, transform, shape, Path("data/raw/copernicus_dem/glo30"))
    cover, cover_tiles = land_cover(grid, transform, shape, Path("data/raw/esa_worldcover/2021_v200"))
    static = xr.Dataset(
        data_vars={
            "population_density_km2": (("y", "x"), pop, {"units": "persons km-2", "reference_year": 2021}),
            "log1p_population_density": (("y", "x"), np.log1p(pop).astype("float32"), {"units": "1"}),
            "elevation": (("y", "x"), dem, {"units": "m", "source": "Copernicus DEM GLO-30"}),
            **{name: (("y", "x"), data, {"units": "fraction", "reference_year": 2021}) for name, data in cover.items()},
        },
        coords={"y": grid.y.values, "x": grid.x.values},
        attrs={
            "title": "CHROMAP-NL static pilot predictors on fixed 500 m grid",
            "target_crs": "EPSG:3035", "dem_tiles_used": dem_tiles,
            "worldcover_tiles_used": cover_tiles,
            "landcover_fraction_method": "25 systematic 100 m nearest-neighbour samples per 500 m cell",
        },
    )
    static_path = args.output_dir / "static_predictors_nl_500m_2021.nc"
    write_netcdf(static, static_path)

    met = era5(grid, Path("data/raw/era5/era5_monthly_nl_buffer_2021"))
    met_path = args.output_dir / "era5_monthly_predictors_nl_500m_2021.nc"
    write_netcdf(met, met_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": str(args.grid),
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (static_path, met_path)
        ],
    }
    manifest_path = args.output_dir / "pilot_raster_predictors_2021.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
