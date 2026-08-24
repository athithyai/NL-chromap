"""Download and checksum the official Eurostat 2021 census population grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


URL = (
    "https://gisco-services.ec.europa.eu/pub/population-grid/2021/"
    "Eurostat_Census-GRID_2021_V1-0.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    destination = Path("data/raw/eurostat/Eurostat_Census-GRID_2021_V1-0.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(".zip.part")
        with requests.get(URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as target:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        target.write(block)
        partial.replace(destination)
    manifest = {
        "source": "Eurostat Census Grid 2021 v1.0",
        "source_url": URL,
        "crs": "EPSG:3035",
        "nominal_resolution": "1 km",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "usage_note": "Reuse subject to Eurostat population-grid download provisions and attribution.",
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
