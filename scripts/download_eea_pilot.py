"""Acquire EEA verified station-series files used by the 2021 pilot.

EEA's classic endpoint rejects these requests as too large. The documented
URL-list endpoint returns immutable Parquet objects (full station series),
which are downloaded here in a restartable, hash-verified layout. The 2021
rows are selected in the downstream observation-preparation step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


URL_API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
COUNTRIES = ("BE", "DE", "FR", "LU")
POLLUTANTS = ("NO2", "O3", "PM10", "PM2.5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_urls(country: str, year: int) -> list[str]:
    payload = {
        "countries": [country], "cities": [], "pollutants": list(POLLUTANTS),
        "dataset": 2, "source": "api", "method": "urls",
        "dateTimeStart": f"{year}-01-01T00:00:00Z",
        "dateTimeEnd": f"{year}-12-31T23:59:59Z",
        "aggregationType": "hour", "compress": False,
    }
    response = requests.post(URL_API, json=payload, timeout=120)
    response.raise_for_status()
    return sorted({line.strip() for line in response.text.splitlines() if line.startswith("http")})


def download(item: tuple[str, Path]) -> dict[str, object]:
    url, destination = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, 6):
            try:
                with requests.get(url, stream=True, timeout=(45, 300)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                partial.replace(destination)
                break
            except (requests.RequestException, OSError):
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
    return {"url": url, "path": destination.as_posix(),
            "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/eea_air_quality"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, object]] = []
    for country in COUNTRIES:
        urls = get_urls(country, args.year)
        country_dir = args.output_dir / country
        (args.output_dir / f"urls_{country}.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        jobs = [(url, country_dir / Path(urlparse(url).path).name) for url in urls]
        print(f"{country}: {len(jobs)} station-series files", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, record in enumerate(executor.map(download, jobs), start=1):
                record["country"] = country
                all_records.append(record)
                if index % 100 == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)}", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "api": URL_API,
        "dataset": "verified E1a (dataset=2)", "requested_pilot_year": args.year,
        "note": "URL-list objects contain full station series; filter to pilot year downstream.",
        "pollutants": list(POLLUTANTS), "countries": list(COUNTRIES),
        "file_count": len(all_records),
        "total_bytes": sum(int(record["bytes"]) for record in all_records),
        "records": all_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({manifest['total_bytes']:,} bytes)")


if __name__ == "__main__":
    main()
