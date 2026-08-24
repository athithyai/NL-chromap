"""Build approximate 500 m road-density predictors from Geofabrik OSM PBF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pyogrio
import rasterio.features
import xarray as xr
from rasterio.transform import from_origin


ROAD_CLASSES = {
    "major": {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"},
    "secondary": {"secondary", "secondary_link", "tertiary", "tertiary_link"},
    "local": {"residential", "living_street", "unclassified", "service", "road"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbf", type=Path, default=Path("data/raw/osm_geofabrik/netherlands-latest.osm.pbf"))
    parser.add_argument("--grid", type=Path, default=Path("data/processed/grid/chromap_nl_500m_grid.nc"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictors/osm_road_density_nl_500m.nc"))
    args = parser.parse_args()

    grid = xr.open_dataset(args.grid)
    bbox = (
        float(np.nanmin(grid.longitude)) - 0.05, float(np.nanmin(grid.latitude)) - 0.05,
        float(np.nanmax(grid.longitude)) + 0.05, float(np.nanmax(grid.latitude)) + 0.05,
    )
    roads = pyogrio.read_dataframe(
        args.pbf, layer="lines", columns=["highway"],
        where="highway IS NOT NULL", bbox=bbox,
    )
    roads = roads[roads.highway.isin(set().union(*ROAD_CLASSES.values()))].to_crs(3035)
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()]

    factor = 10
    resolution = 50
    coarse_shape = (grid.sizes["y"], grid.sizes["x"])
    fine_shape = (coarse_shape[0] * factor, coarse_shape[1] * factor)
    transform = from_origin(float(grid.x.min()) - 250, float(grid.y.max()) + 250, resolution, resolution)
    mask = grid.land_mask.values == 1
    variables: dict[str, tuple[tuple[str, str], np.ndarray, dict[str, str]]] = {}
    counts: dict[str, int] = {}
    for name, highway_types in ROAD_CLASSES.items():
        selected = roads[roads.highway.isin(highway_types)]
        counts[name] = len(selected)
        burned = rasterio.features.rasterize(
            ((geometry, 1) for geometry in selected.geometry), out_shape=fine_shape,
            transform=transform, fill=0, dtype="uint8", all_touched=False,
        )
        pixels = burned.reshape(coarse_shape[0], factor, coarse_shape[1], factor).sum(axis=(1, 3))
        # North-up raster rows are reversed relative to the ascending grid y.
        length_km = np.flipud(pixels).astype("float32") * resolution / 1000
        density = length_km / 0.25
        density = np.where(mask, density, np.nan).astype("float32")
        variables[f"road_density_{name}"] = (
            ("y", "x"), density,
            {"units": "km km-2", "method": "50 m centreline raster length approximation"},
        )

    total = sum(variable[1] for variable in variables.values()).astype("float32")
    variables["road_density_motorized_total"] = (
        ("y", "x"), total,
        {"units": "km km-2", "method": "sum of major, secondary, and local classes"},
    )
    output = xr.Dataset(
        data_vars={name: xr.DataArray(data, dims=dims, attrs=attrs) for name, (dims, data, attrs) in variables.items()},
        coords={"y": grid.y.values, "x": grid.x.values},
        attrs={
            "title": "CHROMAP-NL OSM road-density predictors", "target_crs": "EPSG:3035",
            "source": str(args.pbf), "selected_feature_counts": json.dumps(counts),
            "caveat": "Static present-day proxy; parallel centrelines within one 50 m pixel may be undercounted.",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0} for name in output.data_vars}
    output.to_netcdf(args.output, engine="netcdf4", encoding=encoding)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.pbf),
        "source_sha256": sha256(args.pbf), "output": str(args.output),
        "output_bytes": args.output.stat().st_size, "output_sha256": sha256(args.output),
        "selected_feature_counts": counts,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
