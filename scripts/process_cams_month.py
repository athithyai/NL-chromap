"""Crop one CAMS monthly archive and derive monthly chemical predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr


MEMBER_TO_INDICATOR = {
    "no2": "no2",
    "o3": "o3",
    "pm10": "pm10",
    "pm2p5": "pm2p5",
}

LONG_NAMES = {
    "no2": "monthly mean mass concentration of nitrogen dioxide in air",
    "o3": "monthly mean mass concentration of ozone in air",
    "pm10": "monthly mean mass concentration of particulate matter below 10 micrometres",
    "pm2p5": "monthly mean mass concentration of particulate matter below 2.5 micrometres",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--previous-archive",
        type=Path,
        help="Previous calendar month's archive, used for the first day's O3 8-hour windows",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--west", type=float, default=0.5)
    parser.add_argument("--east", type=float, default=10.5)
    parser.add_argument("--south", type=float, default=49.0)
    parser.add_argument("--north", type=float, default=55.5)
    return parser.parse_args()


def load_previous_o3_tail(
    archive_path: Path,
    staging_dir: Path,
    west: float,
    east: float,
    south: float,
    north: float,
) -> xr.DataArray:
    """Load the final seven O3 hours from the previous calendar month."""

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if member_indicator(name) == "o3"]
        if len(members) != 1:
            raise ValueError(f"expected one previous-month O3 file, found {members}")
        staged = staging_dir / ("previous_" + Path(members[0]).name)
        with archive.open(members[0]) as source, staged.open("wb") as target:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                target.write(block)
        try:
            with xr.open_dataset(staged) as dataset:
                return dataset["o3"].sel(
                    lon=slice(west, east),
                    lat=slice(south, north),
                ).isel(time=slice(-7, None)).load()
        finally:
            staged.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_indicator(name: str) -> str | None:
    for token, indicator in MEMBER_TO_INDICATOR.items():
        if f".{token}." in name:
            return indicator
    return None


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    archive_hash = sha256(args.archive)
    previous_archive_hash = sha256(args.previous_archive) if args.previous_archive else None
    previous_o3_tail = (
        load_previous_o3_tail(
            args.previous_archive,
            args.staging_dir,
            args.west,
            args.east,
            args.south,
            args.north,
        )
        if args.previous_archive
        else None
    )
    monthly_arrays = []
    month_start = None

    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if member_indicator(name)]
        if len(members) != 4:
            raise ValueError(f"expected four pollutant files, found: {members}")

        for member in sorted(members):
            indicator = member_indicator(member)
            assert indicator is not None
            staged = args.staging_dir / Path(member).name
            print(f"Extracting {member}", flush=True)
            with archive.open(member) as source, staged.open("wb") as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)

            try:
                with xr.open_dataset(staged) as dataset:
                    if indicator not in dataset.data_vars:
                        raise ValueError(
                            f"variable {indicator} not found in {member}: "
                            f"{list(dataset.data_vars)}"
                        )
                    subset = dataset[indicator].sel(
                        lon=slice(args.west, args.east),
                        lat=slice(args.south, args.north),
                    )
                    if subset.sizes.get("lon", 0) == 0 or subset.sizes.get("lat", 0) == 0:
                        raise ValueError("requested geographic subset is empty")
                    times = pd.DatetimeIndex(subset.time.values)
                    current_month = pd.Timestamp(times[0]).to_period("M")
                    if not (times.to_period("M") == current_month).all():
                        raise ValueError(f"file crosses calendar months: {member}")
                    if month_start is None:
                        month_start = current_month.to_timestamp()
                    elif month_start != current_month.to_timestamp():
                        raise ValueError("archive members contain different months")

                    mean = subset.mean("time", skipna=True).load()
                    mean.name = indicator
                    mean.attrs.update(subset.attrs)
                    mean.attrs["long_name"] = LONG_NAMES[indicator]
                    mean.attrs["units"] = "ug m-3"
                    mean.attrs["aggregation"] = "calendar-month mean of hourly CAMS fields"
                    monthly_arrays.append(mean)

                    if indicator == "o3":
                        ozone_for_rolling = (
                            xr.concat([previous_o3_tail, subset], dim="time")
                            if previous_o3_tail is not None
                            else subset
                        )
                        rolling_8h = ozone_for_rolling.rolling(time=8, min_periods=6).mean()
                        daily_max_8h = rolling_8h.resample(time="1D").max(skipna=True)
                        daily_max_8h = daily_max_8h.sel(
                            time=(daily_max_8h.time.dt.year == current_month.year)
                            & (daily_max_8h.time.dt.month == current_month.month)
                        )
                        somo35 = (daily_max_8h - 70.0).clip(min=0).sum(
                            "time", skipna=True
                        ).load()
                        somo35.name = "somo35"
                        somo35.attrs = {
                            "long_name": "monthly SOMO35 contribution",
                            "units": "ug m-3 day",
                            "threshold": "70 ug m-3",
                            "aggregation": (
                                "sum of positive daily maximum running 8-hour "
                                "O3 means above 70 ug m-3"
                            ),
                            "month_boundary_complete": str(previous_o3_tail is not None).lower(),
                        }
                        monthly_arrays.append(somo35)
            finally:
                staged.unlink(missing_ok=True)

    if month_start is None:
        raise RuntimeError("no CAMS data processed")
    result = xr.merge(monthly_arrays).expand_dims(time=[month_start])
    result.attrs = {
        "title": "Monthly CAMS chemical predictors for CHROMAP-NL",
        "source_dataset": "cams-europe-air-quality-reanalyses",
        "source_doi": "10.24381/7cc0465a",
        "source_archive": args.archive.name,
        "source_archive_sha256": archive_hash,
        "previous_archive": args.previous_archive.name if args.previous_archive else "not supplied",
        "previous_archive_sha256": previous_archive_hash or "not supplied",
        "somo35_month_boundary_complete": str(previous_o3_tail is not None).lower(),
        "reanalysis_type": "validated_reanalysis"
        if "validated_reanalysis" in args.archive.name
        else "interim_reanalysis",
        "subset_bounds_west_east_south_north": (
            f"{args.west},{args.east},{args.south},{args.north}"
        ),
    }
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in result.data_vars
    }
    result.to_netcdf(args.output, engine="netcdf4", encoding=encoding)

    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_archive": str(args.archive),
        "source_archive_sha256": archive_hash,
        "previous_archive": str(args.previous_archive) if args.previous_archive else None,
        "previous_archive_sha256": previous_archive_hash,
        "somo35_month_boundary_complete": previous_o3_tail is not None,
        "variables": list(result.data_vars),
        "dimensions": dict(result.sizes),
        "bounds": {
            "west": args.west,
            "east": args.east,
            "south": args.south,
            "north": args.north,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
