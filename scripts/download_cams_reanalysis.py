"""Download CAMS European ensemble reanalysis monthly files from the ADS."""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi


VARIABLES = [
    "nitrogen_dioxide",
    "ozone",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, choices=range(1, 13), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--type",
        choices=["validated_reanalysis", "interim_reanalysis"],
        default="validated_reanalysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"cams_ensemble_{args.type}_{args.year}_{args.month:02d}.zip"
    if target.exists() and target.stat().st_size > 0:
        print(f"Already present: {target}")
        return

    request = {
        "variable": VARIABLES,
        "model": ["ensemble"],
        "level": ["0"],
        "type": [args.type],
        "year": [str(args.year)],
        "month": [f"{args.month:02d}"],
    }
    client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")
    client.retrieve("cams-europe-air-quality-reanalyses", request, str(target))
    print(f"Downloaded {target}")


if __name__ == "__main__":
    main()

