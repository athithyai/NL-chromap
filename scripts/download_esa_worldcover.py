"""Download ESA WorldCover 2021 tiles covering the pilot buffer."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
TILES = [f"N{lat:02d}E{lon:03d}" for lat in (48, 51, 54) for lon in (0, 3, 6, 9)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(tile: str, directory: Path) -> dict | None:
    name = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    url = f"{BASE}/{name}"
    destination = directory / name
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
        "tile": tile,
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    directory = Path("data/raw/esa_worldcover/2021_v200")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, tile, directory): tile for tile in TILES}
        for future in as_completed(futures):
            record = future.result()
            if record:
                records.append(record)
                print(f"Acquired {record['tile']}", flush=True)
    records.sort(key=lambda row: row["tile"])
    manifest = {
        "source": "ESA WorldCover 10 m 2021 v200",
        "doi": "10.5281/zenodo.7254221",
        "licence": "CC BY 4.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "tiles": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tiles"}, indent=2))


if __name__ == "__main__":
    main()
