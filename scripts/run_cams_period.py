"""Run CAMS acquisition, monthly processing, baseline gridding, and validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(*arguments: object) -> None:
    command = [sys.executable, *map(str, arguments)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def archive_path(raw_dir: Path, year: int, month: int, kind: str) -> Path:
    return raw_dir / f"cams_ensemble_{kind}_{year}_{month:02d}.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="First month, YYYY-MM")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cams"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cams"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/processed/baseline"))
    parser.add_argument("--validation-dir", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--staging-dir", type=Path, default=Path("data/staging/cams"))
    parser.add_argument("--grid", type=Path, default=Path("data/processed/grid/chromap_nl_500m_grid.nc"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/processed/observations/rivm_station_month_targets_2013_2025.csv.gz"),
    )
    parser.add_argument(
        "--prune-used-raw",
        action="store_true",
        help="Delete each previous archive only after the next month is processed successfully",
    )
    args = parser.parse_args()

    months = pd.period_range(args.start, args.end, freq="M")
    previous = months[0] - 1
    previous_kind = "validated_reanalysis" if previous.year <= 2024 else "interim_reanalysis"
    previous_archive = archive_path(args.raw_dir, previous.year, previous.month, previous_kind)
    if not previous_archive.exists():
        run(
            "scripts/download_cams_reanalysis.py",
            "--year", previous.year,
            "--month", previous.month,
            "--type", previous_kind,
            "--output-dir", args.raw_dir,
        )

    for month in months:
        kind = "validated_reanalysis" if month.year <= 2024 else "interim_reanalysis"
        archive = archive_path(args.raw_dir, month.year, month.month, kind)
        run(
            "scripts/download_cams_reanalysis.py",
            "--year", month.year,
            "--month", month.month,
            "--type", kind,
            "--output-dir", args.raw_dir,
        )
        predictor = args.processed_dir / f"cams_monthly_predictors_{month.year}_{month.month:02d}.nc"
        run(
            "scripts/process_cams_month.py",
            "--archive", archive,
            "--previous-archive", previous_archive,
            "--output", predictor,
            "--staging-dir", args.staging_dir,
        )
        baseline = args.baseline_dir / f"cams_baseline_nl_500m_{month.year}_{month.month:02d}.nc"
        run(
            "scripts/regrid_cams_baseline.py",
            "--cams", predictor,
            "--grid", args.grid,
            "--output", baseline,
        )
        run(
            "scripts/evaluate_cams_baseline.py",
            "--targets", args.targets,
            "--cams", predictor,
            "--month", str(month),
            "--output-dir", args.validation_dir,
        )
        if args.prune_used_raw and previous_archive.exists():
            previous_archive.unlink()
            print(f"Removed reproducible used cache: {previous_archive}", flush=True)
        previous_archive = archive


if __name__ == "__main__":
    main()
