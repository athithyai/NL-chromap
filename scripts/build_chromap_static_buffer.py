"""Build the paper-defined static CHROMAP features on a 500 m buffer grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import distance_transform_edt


BBOX = (0.0, 49.0, 11.0, 56.0)
TARGET_CRS = CRS.from_epsg(3035)
RESOLUTION = 500
CGLS_RECORD = 3518038
CGLS_FILES = {
    "GEO_Crop": "PROBAV_LC100_global_v3.0.1_2018-conso_Crops-CoverFraction-layer_EPSG-4326.tif",
    "GEO_Tree": "PROBAV_LC100_global_v3.0.1_2018-conso_Tree-CoverFraction-layer_EPSG-4326.tif",
    "GEO_Grass": "PROBAV_LC100_global_v3.0.1_2018-conso_Grass-CoverFraction-layer_EPSG-4326.tif",
    "GEO_Builtup": "PROBAV_LC100_global_v3.0.1_2018-conso_BuiltUp-CoverFraction-layer_EPSG-4326.tif",
    "GEO_Shrub": "PROBAV_LC100_global_v3.0.1_2018-conso_Shrub-CoverFraction-layer_EPSG-4326.tif",
    "GEO_Bare": "PROBAV_LC100_global_v3.0.1_2018-conso_Bare-CoverFraction-layer_EPSG-4326.tif",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_definition() -> tuple[np.ndarray, np.ndarray, object, tuple[int, int]]:
    left, bottom, right, top = transform_bounds(CRS.from_epsg(4326), TARGET_CRS,
                                                 *BBOX, densify_pts=41)
    left = math.floor(left / RESOLUTION) * RESOLUTION
    bottom = math.floor(bottom / RESOLUTION) * RESOLUTION
    right = math.ceil(right / RESOLUTION) * RESOLUTION
    top = math.ceil(top / RESOLUTION) * RESOLUTION
    width, height = int((right - left) / RESOLUTION), int((top - bottom) / RESOLUTION)
    transform = from_origin(left, top, RESOLUTION, RESOLUTION)
    x = left + RESOLUTION / 2 + np.arange(width) * RESOLUTION
    y = bottom + RESOLUTION / 2 + np.arange(height) * RESOLUTION
    return x, y, transform, (height, width)


def cgls_fraction(name: str, transform, shape: tuple[int, int]) -> np.ndarray:
    cache = Path("data/interim/chromap_cgls_500m") / f"{name}.npy"
    if cache.exists():
        return np.load(cache)
    filename = CGLS_FILES[name]
    url = (f"/vsicurl/https://zenodo.org/records/{CGLS_RECORD}/files/"
           f"{filename}?download=1")
    last_error = None
    for attempt in range(1, 6):
        try:
            north_up = np.full(shape, np.nan, dtype="float32")
            with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"):
                with rasterio.open(url) as source:
                    reproject(
                        source=rasterio.band(source, 1), destination=north_up,
                        src_transform=source.transform, src_crs=source.crs, src_nodata=255,
                        dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
                        resampling=Resampling.average,
                    )
            result = np.flipud(north_up)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, result)
            # A single warped read consumes most of Zenodo's anonymous request
            # allowance because the cloud TIFF is tiled into many byte ranges.
            time.sleep(61)
            return result
        except (OSError, rasterio.errors.RasterioError) as error:
            last_error = error
            print(f"retry {attempt}/5 for {name}: {error}", flush=True)
            time.sleep(attempt * 2)
    raise RuntimeError(f"failed remote CGLS read for {name}") from last_error


def ghsl_population(archive: Path, transform, shape: tuple[int, int]) -> np.ndarray:
    extracted = archive.parent / "extracted" / "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
    if extracted.exists():
        uri = str(extracted)
    else:
        with zipfile.ZipFile(archive) as zipped:
            member = next(name for name in zipped.namelist() if name.lower().endswith(".tif"))
        uri = f"zip://{archive.resolve()}!{member}"
    north_up = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(uri) as source:
        reproject(
            source=rasterio.band(source, 1), destination=north_up,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
            resampling=Resampling.sum,
        )
    return np.flipud(np.where(np.isfinite(north_up) & (north_up >= 0), north_up, 0))


def ghsl_smod(path: Path, transform, shape: tuple[int, int]) -> np.ndarray:
    north_up = np.full(shape, -200, dtype="int16")
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1), destination=north_up,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=-200,
            resampling=Resampling.nearest,
        )
    return np.flipud(north_up)


def elevation(directory: Path, transform, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    target_bounds = rasterio.transform.array_bounds(*shape, transform)
    total = np.zeros(shape, dtype="float64")
    count = np.zeros(shape, dtype="uint8")
    used = 0
    for path in sorted(directory.glob("*.tif")):
        with rasterio.open(path) as source:
            bounds = transform_bounds(source.crs, TARGET_CRS, *source.bounds, densify_pts=21)
            if not (bounds[0] < target_bounds[2] and bounds[2] > target_bounds[0]
                    and bounds[1] < target_bounds[3] and bounds[3] > target_bounds[1]):
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
    return np.flipud(np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)), used


def coast_distance(path: Path, transform, shape: tuple[int, int]) -> np.ndarray:
    land = gpd.read_file(path).to_crs(TARGET_CRS)
    mask = rasterio.features.rasterize(
        ((geometry, 1) for geometry in land.geometry if geometry is not None),
        out_shape=shape, transform=transform, fill=0, dtype="uint8",
    ).astype(bool)
    distance = distance_transform_edt(mask) * RESOLUTION
    return np.flipud(distance.astype("float32"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/predictors/chromap_static_buffer_500m_2021.nc"))
    args = parser.parse_args()
    x, y, transform, shape = grid_definition()
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(TARGET_CRS, 4326, always_xy=True).transform(xx, yy)
    variables: dict[str, tuple[tuple[str, str], np.ndarray, dict[str, str]]] = {}
    for name in CGLS_FILES:
        print(f"processing {name}", flush=True)
        variables[name] = (("y", "x"), cgls_fraction(name, transform, shape), {
            "units": "%", "source": "Copernicus Global Land Cover v3 epoch 2018",
        })
    pop_archive = Path("data/raw/ghsl_pop/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.zip")
    variables["GEO_PopDens"] = (("y", "x"), ghsl_population(pop_archive, transform, shape), {
        "units": "inhabitants per 500 m cell", "source": "GHS-POP R2023A epoch 2020, 100 m",
    })
    smod_path = Path("data/raw/ghsl_smod/extracted/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.tif")
    variables["GEO_UrbDeg"] = (("y", "x"), ghsl_smod(smod_path, transform, shape), {
        "units": "class", "source": "GHS-SMOD R2023A epoch 2020, nearest neighbour",
        "class_codes": "11,12,13,21,22,23,30; 10 is water",
    })
    altitude, dem_tiles = elevation(Path("data/raw/copernicus_dem/glo30"), transform, shape)
    variables["GEO_Alt"] = (("y", "x"), altitude.astype("float32"), {
        "units": "m", "source": "Copernicus DEM GLO-30 aggregated; paper used GLO-90",
    })
    coast = coast_distance(Path("data/raw/natural_earth/ne_10m_land/ne_10m_land.shp"),
                           transform, shape)
    variables["GEO_DistCoast"] = (("y", "x"), coast, {
        "units": "m", "source": "Natural Earth 1:10m land boundary calculation",
    })
    variables["GEO_Lon"] = (("y", "x"), lon.astype("float32"), {"units": "degrees_east"})
    variables["GEO_Lat"] = (("y", "x"), lat.astype("float32"), {"units": "degrees_north"})
    dataset = xr.Dataset(
        {name: xr.DataArray(data, dims=dims, attrs=attrs)
         for name, (dims, data, attrs) in variables.items()},
        coords={"x": x, "y": y},
        attrs={
            "title": "Paper-defined CHROMAP static predictor stack for January 2021 adaptation",
            "target_crs": "EPSG:3035", "resolution_m": RESOLUTION,
            "bbox_lonlat": BBOX, "dem_tiles_used": dem_tiles,
            "known_deviations": "Copernicus DEM GLO-30 aggregated instead of cited GLO-90; Natural Earth coastline",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name in dataset.data_vars:
        if np.issubdtype(dataset[name].dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
        else:
            encoding[name] = {"zlib": True, "complevel": 4}
    dataset.to_netcdf(args.output, engine="netcdf4", encoding=encoding)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "output": str(args.output),
        "bytes": args.output.stat().st_size, "sha256": sha256(args.output),
        "variables": list(dataset.data_vars), "shape": list(shape),
        "sources": {
            "cgls": f"https://zenodo.org/records/{CGLS_RECORD}",
            "ghsl_pop_sha256": sha256(pop_archive),
            "ghsl_smod_sha256": sha256(Path("data/raw/ghsl_smod/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.zip")),
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
