"""Acquire the official EEA sampling-point metadata extract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "https://ereporting.blob.core.windows.net/downloadservice/metadata.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    target = Path("data/raw/eea_air_quality/metadata.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with requests.get(URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for block in response.iter_content(4 * 1024 * 1024):
                    if block:
                        handle.write(block)
    manifest = {
        "source": URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": target.stat().st_size, "sha256": sha256(target),
        "note": "Daily metadata extract archived by EEA; server Last-Modified was 2022-11-03 at retrieval.",
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
