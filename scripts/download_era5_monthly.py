"""Download compact monthly ERA5 meteorological predictors for the pilot domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cdsapi


DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "boundary_layer_height",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "total_precipitation",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/era5"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"era5_monthly_nl_buffer_{args.year}"
    target = args.output_dir / f"{stem}.zip"
    legacy_target = args.output_dir / f"{stem}.nc"
    # Earlier cdsapi versions may save a multi-stream ZIP under the requested
    # .nc filename. Normalize that case without downloading the job again.
    if not target.exists() and legacy_target.exists() and zipfile.is_zipfile(legacy_target):
        legacy_target.replace(target)
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": [str(args.year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        # Required by the CDS monthly-means form even though the selected
        # product contains one monthly mean rather than an hourly series.
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [56.0, 0.0, 49.0, 11.0],
    }
    if not target.exists():
        cdsapi.Client(url="https://cds.climate.copernicus.eu/api").retrieve(
            DATASET, request, str(target)
        )
    if not zipfile.is_zipfile(target):
        raise RuntimeError(f"Expected CDS multi-stream ZIP, received {target}")
    extract_dir = args.output_dir / stem
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(extract_dir)
        member_paths = [extract_dir / name for name in archive.namelist()]
    members = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in member_paths
    ]
    manifest = {
        "dataset": DATASET,
        "doi": "10.24381/cds.f17050d7",
        "request": request,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "members": members,
    }
    (args.output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
