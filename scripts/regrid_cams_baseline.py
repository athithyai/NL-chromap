"""Interpolate monthly CAMS fields onto the fixed 500 m Netherlands grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with xr.open_dataset(args.grid) as grid, xr.open_dataset(args.cams) as cams:
        lon = grid["longitude"].load()
        lat = grid["latitude"].load()
        mask = grid["land_mask"].load()
        output = xr.Dataset(coords={"time": cams.time.load(), "y": grid.y.load(), "x": grid.x.load()})
        for indicator in ["no2", "o3", "pm10", "pm2p5", "somo35"]:
            field = cams[indicator].interp(lon=lon, lat=lat, method="linear")
            field = field.where(mask == 1).astype("float32")
            field.attrs.update(cams[indicator].attrs)
            field.attrs["grid_mapping"] = "spatial_ref"
            field.attrs["processing_level"] = "CAMS 0.1-degree baseline interpolated to target grid"
            output[indicator] = field
        output["land_mask"] = mask
        output["longitude"] = lon
        output["latitude"] = lat
        output["spatial_ref"] = grid["spatial_ref"].load()

    output.attrs = {
        "title": "CHROMAP-NL CAMS baseline on the 500 m target grid",
        "product_status": "baseline, not statistical downscaling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_cams_file": args.cams.name,
        "source_grid_file": args.grid.name,
        "interpolation": "bilinear in geographic coordinates; land-cell mask applied",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 5}
        for name in output.data_vars
        if name != "spatial_ref"
    }
    encoding["land_mask"]["dtype"] = "uint8"
    output.to_netcdf(args.output, engine="netcdf4", encoding=encoding)

    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_cams": str(args.cams),
        "source_cams_sha256": sha256(args.cams),
        "source_grid": str(args.grid),
        "source_grid_sha256": sha256(args.grid),
        "variables": ["no2", "o3", "pm10", "pm2p5", "somo35"],
        "dimensions": {key: int(value) for key, value in output.sizes.items()},
        "status": "baseline_not_downscaled",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
