"""Download one CAMS European ensemble reanalysis month for SO2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, choices=range(1, 13), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=["validated_reanalysis", "interim_reanalysis"],
                        default="validated_reanalysis")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 0:
        print(f"Already present: {args.output}")
        return
    request = {
        "variable": ["sulphur_dioxide"], "model": ["ensemble"], "level": ["0"],
        "type": [args.type], "year": [str(args.year)], "month": [f"{args.month:02d}"],
    }
    cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api").retrieve(
        "cams-europe-air-quality-reanalyses", request, str(args.output)
    )
    print(args.output)


if __name__ == "__main__":
    main()

