"""Acquire and process ratified RIVM observations for a range of years."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from chromap_nl.rivm import (
    build_monthly_targets,
    coverage_summary,
    read_measurements,
    read_metadata,
)


COMPONENT_FILES = ["NO2", "PM25", "PM10", "O3"]
BASE_URL = "https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(8 * 1024 * 1024):
                if block:
                    stream.write(block)
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    for directory in [args.raw_dir, args.staging_dir, args.output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(
        args.metadata_dir / "luchtmeetnet_meetlocaties.csv",
        args.metadata_dir / "luchtmeetnet_meetreeksen.csv",
    )

    manifests = []
    all_targets = []
    staging_root = args.staging_dir.resolve()
    for year in range(args.start_year, args.end_year + 1):
        archive = args.raw_dir / f"{year}.zip"
        url = f"{BASE_URL}/{year}/{year}.zip"
        print(f"Acquiring {year}: {url}", flush=True)
        download(url, archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"invalid zip archive: {archive}")

        year_stage = (args.staging_dir / str(year)).resolve()
        if staging_root not in year_stage.parents:
            raise RuntimeError(f"unsafe staging path: {year_stage}")
        year_stage.mkdir(parents=True, exist_ok=True)

        parts = []
        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
            required = [f"{year}_{component}.csv" for component in COMPONENT_FILES]
            missing = set(required) - names
            if missing:
                raise ValueError(f"missing files in {archive}: {sorted(missing)}")
            for member in required:
                target = year_stage / member
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                measurements = read_measurements(target)
                parts.append(build_monthly_targets(measurements, metadata))

        targets = pd.concat(parts, ignore_index=True)
        targets_path = args.output_dir / f"rivm_station_month_targets_{year}.csv.gz"
        coverage_path = args.output_dir / f"rivm_station_month_coverage_{year}.csv"
        targets.to_csv(targets_path, index=False, compression="gzip")
        coverage_summary(targets).to_csv(coverage_path, index=False)
        all_targets.append(targets)
        manifests.append(
            {
                "year": year,
                "source_url": url,
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "target_rows": len(targets),
                "valid_target_rows": int(targets["is_valid"].sum()),
            }
        )
        shutil.rmtree(year_stage)
        print(f"Processed {year}", flush=True)

    combined = pd.concat(all_targets, ignore_index=True)
    combined.to_csv(
        args.output_dir
        / f"rivm_station_month_targets_{args.start_year}_{args.end_year}.csv.gz",
        index=False,
        compression="gzip",
    )
    coverage_summary(combined).to_csv(
        args.output_dir
        / f"rivm_station_month_coverage_{args.start_year}_{args.end_year}.csv",
        index=False,
    )
    manifest_path = args.output_dir / "rivm_history_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

