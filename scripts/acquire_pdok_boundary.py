"""Download and checksum the official CBS/PDOK 2026 land-water geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


SOURCE_URL = (
    "https://service.pdok.nl/cbs/wijken-en-buurten-2026/atom/downloads/"
    "WijkBuurtkaart_2026_v0.gpkg"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/pdok/WijkBuurtkaart_2026_v0.gpkg"),
    )
    args = parser.parse_args()

    if not args.output.exists():
        download(SOURCE_URL, args.output)

    manifest = {
        "source": "CBS Wijk- en buurtkaart 2026 versie 0 via PDOK ATOM",
        "source_url": SOURCE_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "intended_use": "Select buurt polygons where water = NEE and union as land mask.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
