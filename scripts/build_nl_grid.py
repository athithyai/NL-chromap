"""Build the immutable 500 m EPSG:3035 Netherlands land grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import force_2d
from shapely.ops import unary_union


GRID_CRS = CRS.from_epsg(3035)
CELL_SIZE = 500.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    return (
        math.floor(xmin / CELL_SIZE) * CELL_SIZE,
        math.floor(ymin / CELL_SIZE) * CELL_SIZE,
        math.ceil(xmax / CELL_SIZE) * CELL_SIZE,
        math.ceil(ymax / CELL_SIZE) * CELL_SIZE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/pdok/WijkBuurtkaart_2026_v0.gpkg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/grid/chromap_nl_500m_grid.nc"),
    )
    args = parser.parse_args()

    land_parts = gpd.read_file(
        args.source,
        layer="buurten",
        columns=["water", "geometry"],
        where="water = 'NEE'",
    )
    if land_parts.empty or set(land_parts["water"]) != {"NEE"}:
        raise RuntimeError("PDOK land selection did not return the expected polygons")

    land_parts["geometry"] = land_parts.geometry.map(force_2d)
    land = unary_union(land_parts.to_crs(GRID_CRS).geometry)
    xmin, ymin, xmax, ymax = aligned_bounds(land.bounds)
    width = int((xmax - xmin) / CELL_SIZE)
    height = int((ymax - ymin) / CELL_SIZE)

    # Rasterio writes north-to-south; flip so the NetCDF y coordinate is ascending.
    transform = from_origin(xmin, ymax, CELL_SIZE, CELL_SIZE)
    north_down = rasterize(
        [(land, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    land_mask = np.flipud(north_down)
    x = xmin + CELL_SIZE * (np.arange(width) + 0.5)
    y = ymin + CELL_SIZE * (np.arange(height) + 0.5)
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(GRID_CRS, 4326, always_xy=True)
    longitude, latitude = transformer.transform(xx, yy)

    ds = xr.Dataset(
        data_vars={
            "land_mask": (
                ("y", "x"),
                land_mask,
                {
                    "long_name": "European Netherlands land-cell mask",
                    "flag_values": np.array([0, 1], dtype="uint8"),
                    "flag_meanings": "outside_or_major_water land",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "longitude": (("y", "x"), longitude.astype("float32"), {"units": "degrees_east"}),
            "latitude": (("y", "x"), latitude.astype("float32"), {"units": "degrees_north"}),
            "spatial_ref": ((), np.int32(0), GRID_CRS.to_cf()),
        },
        coords={
            "x": ("x", x, {"standard_name": "projection_x_coordinate", "units": "m"}),
            "y": ("y", y, {"standard_name": "projection_y_coordinate", "units": "m"}),
        },
        attrs={
            "title": "CHROMAP Netherlands fixed 500 m target grid",
            "source": "CBS Wijk- en buurtkaart 2026 v0 via PDOK; buurt water=NEE",
            "source_crs": str(land_parts.crs),
            "target_crs": "EPSG:3035",
            "cell_selection": "cell centre falls inside union of non-water neighbourhood polygons",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "land_mask": {"zlib": True, "complevel": 9, "dtype": "uint8"},
        "longitude": {"zlib": True, "complevel": 5},
        "latitude": {"zlib": True, "complevel": 5},
    }
    ds.to_netcdf(args.output, encoding=encoding)

    manifest = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "crs": "EPSG:3035",
        "cell_size_m": CELL_SIZE,
        "aligned_bounds": [xmin, ymin, xmax, ymax],
        "shape_yx": [height, width],
        "land_cells": int(land_mask.sum()),
        "represented_land_area_km2": float(land_mask.sum() * CELL_SIZE**2 / 1e6),
        "source_polygon_area_km2": float(land.area / 1e6),
        "selection": "PDOK buurten where water = NEE; cell-centre rasterization",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
