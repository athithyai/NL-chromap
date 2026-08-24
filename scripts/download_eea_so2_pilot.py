"""Download verified EEA SO2 station-series files for the pilot countries."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from download_eea_pilot import download


URL_API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
COUNTRIES = ("BE", "DE", "FR", "LU")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def urls(country: str, year: int) -> list[str]:
    payload = {
        "countries": [country], "cities": [], "pollutants": ["SO2"],
        "dataset": 2, "source": "api", "method": "urls",
        "dateTimeStart": f"{year}-01-01T00:00:00Z",
        "dateTimeEnd": f"{year}-12-31T23:59:59Z",
        "aggregationType": "hour", "compress": False,
    }
    response = requests.post(URL_API, json=payload, timeout=120)
    response.raise_for_status()
    return sorted({line.strip() for line in response.text.splitlines() if line.startswith("http")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/raw/eea_air_quality_so2"))
    args = parser.parse_args()
    records = []
    for country in COUNTRIES:
        country_urls = urls(country, args.year)
        jobs = [(url, args.output_dir / country / Path(urlparse(url).path).name)
                for url in country_urls]
        print(f"{country}: {len(jobs)} files", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record in executor.map(download, jobs):
                record["country"] = country
                records.append(record)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "api": URL_API,
        "dataset": "verified E1a (dataset=2)", "requested_year": args.year,
        "pollutants": ["SO2"], "countries": list(COUNTRIES),
        "file_count": len(records), "total_bytes": sum(int(row["bytes"]) for row in records),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
