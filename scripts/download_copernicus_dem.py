"""Download Copernicus DEM GLO-30 COG tiles intersecting the pilot buffer."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
WEST, EAST, SOUTH, NORTH = 0, 11, 49, 56


def tile_name(latitude: int, longitude: int) -> str:
    ns = "N" if latitude >= 0 else "S"
    ew = "E" if longitude >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(latitude):02d}_00_{ew}{abs(longitude):03d}_00_DEM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(latitude: int, longitude: int, directory: Path) -> dict | None:
    name = tile_name(latitude, longitude)
    url = f"{BASE}/{name}/{name}.tif"
    destination = directory / f"{name}.tif"
    if not destination.exists():
        response = requests.get(url, stream=True, timeout=(30, 300))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        partial = destination.with_suffix(".tif.part")
        with response, partial.open("wb") as target:
            for block in response.iter_content(1024 * 1024):
                if block:
                    target.write(block)
        partial.replace(destination)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    directory = Path("data/raw/copernicus_dem/glo30")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(lat, lon) for lat in range(SOUTH, NORTH) for lon in range(WEST, EAST)]
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, lat, lon, directory): (lat, lon) for lat, lon in jobs}
        for future in as_completed(futures):
            record = future.result()
            if record:
                records.append(record)
                print(f"Acquired {Path(record['path']).name}", flush=True)
    records.sort(key=lambda row: (row["latitude"], row["longitude"]))
    manifest = {
        "source": "Copernicus DEM GLO-30 Public, 2021 release",
        "source_registry": "https://registry.opendata.aws/copernicus-dem/",
        "bbox_west_east_south_north": [WEST, EAST, SOUTH, NORTH],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "tiles": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tiles"}, indent=2))


if __name__ == "__main__":
    main()
