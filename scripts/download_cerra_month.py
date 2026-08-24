"""Download one month of the four CERRA variables used by CHROMAP."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cdsapi


DATASET = "reanalysis-cerra-single-levels"
VARIABLES = ["10m_wind_speed", "2m_temperature", "surface_pressure", "total_cloud_cover"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--month", type=int, default=1, choices=range(1, 13))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/cerra"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"cerra_chromap_{args.year}_{args.month:02d}.nc"
    request = {
        "variable": VARIABLES,
        "level_type": "surface_or_atmosphere",
        "data_type": ["reanalysis"],
        "product_type": "analysis",
        "year": [str(args.year)], "month": [f"{args.month:02d}"],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(0, 24, 3)],
        "data_format": "netcdf",
    }
    if not output.exists():
        cdsapi.Client(url="https://cds.climate.copernicus.eu/api").retrieve(
            DATASET, request, str(output)
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
        "doi": "10.24381/cds.622a565a", "request": request, "output": str(output),
        "bytes": output.stat().st_size, "sha256": sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
