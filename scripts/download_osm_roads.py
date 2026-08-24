"""Download regional OSM PBF extracts covering the 150 km training buffer."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


URLS = {
    "netherlands": "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf",
    "belgium": "https://download.geofabrik.de/europe/belgium-latest.osm.pbf",
    "luxembourg": "https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf",
    "nordrhein-westfalen": "https://download.geofabrik.de/europe/germany/nordrhein-westfalen-latest.osm.pbf",
    "niedersachsen": "https://download.geofabrik.de/europe/germany/niedersachsen-latest.osm.pbf",
    "rheinland-pfalz": "https://download.geofabrik.de/europe/germany/rheinland-pfalz-latest.osm.pbf",
    "nord-pas-de-calais": "https://download.geofabrik.de/europe/france/nord-pas-de-calais-latest.osm.pbf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, url: str, directory: Path) -> dict:
    destination = directory / f"{name}-latest.osm.pbf"
    resolved_url = url
    if not destination.exists():
        response = requests.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()
        resolved_url = response.url
        partial = destination.with_suffix(".pbf.part")
        with response, partial.open("wb") as target:
            for block in response.iter_content(4 * 1024 * 1024):
                if block:
                    target.write(block)
        partial.replace(destination)
    return {
        "region": name,
        "source_url": url,
        "resolved_snapshot_url": resolved_url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    directory = Path("data/raw/osm_geofabrik")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fetch, name, url, directory): name for name, url in URLS.items()
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"Acquired {record['region']}", flush=True)
    records.sort(key=lambda row: row["region"])
    manifest = {
        "source": "OpenStreetMap regional extracts distributed by Geofabrik",
        "licence": "ODbL 1.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "use": "Static road-density proxy; clip to 150 km training buffer before gridding",
        "region_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
