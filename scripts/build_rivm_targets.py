"""Build pilot station-month targets from ratified RIVM hourly files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chromap_nl.rivm import (
    build_monthly_targets,
    coverage_summary,
    read_measurements,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(
        args.metadata_dir / "luchtmeetnet_meetlocaties.csv",
        args.metadata_dir / "luchtmeetnet_meetreeksen.csv",
    )

    files = [
        args.raw_dir / f"{args.year}_NO2.csv",
        args.raw_dir / f"{args.year}_PM25.csv",
        args.raw_dir / f"{args.year}_PM10.csv",
        args.raw_dir / f"{args.year}_O3.csv",
    ]
    parts = []
    for path in files:
        print(f"Reading {path.name}", flush=True)
        measurements = read_measurements(path)
        parts.append(build_monthly_targets(measurements, metadata))

    targets = pd.concat(parts, ignore_index=True)
    targets_path = args.output_dir / f"rivm_station_month_targets_{args.year}.csv.gz"
    coverage_path = args.output_dir / f"rivm_station_month_coverage_{args.year}.csv"
    targets.to_csv(targets_path, index=False, compression="gzip")
    coverage_summary(targets).to_csv(coverage_path, index=False)
    print(f"Wrote {targets_path}")
    print(f"Wrote {coverage_path}")


if __name__ == "__main__":
    main()

