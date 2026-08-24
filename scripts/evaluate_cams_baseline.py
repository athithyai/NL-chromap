"""Evaluate monthly CAMS predictors at ratified RIVM station locations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cams", type=Path, required=True)
    parser.add_argument("--month", required=True, help="Calendar month in YYYY-MM form")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metrics(group: pd.DataFrame) -> pd.Series:
    observed = group["observed"].to_numpy(dtype=float)
    predicted = group["cams"].to_numpy(dtype=float)
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    mean_observed = float(np.mean(observed))
    denominator = float(np.sum((observed - mean_observed) ** 2))
    r2 = float(1 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan
    return pd.Series(
        {
            "n": len(group),
            "mean_observed": mean_observed,
            "mean_cams": float(np.mean(predicted)),
            "rmse": rmse,
            "rrmse": rmse / mean_observed if mean_observed != 0 else np.nan,
            "mae": mae,
            "bias_cams_minus_observed": float(np.mean(residual)),
            "r2": r2,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets)
    targets = targets.loc[
        (targets["month"] == args.month)
        & targets["is_valid"]
        & targets["value"].notna()
    ].copy()
    targets = targets.rename(columns={"value": "observed"})

    with xr.open_dataset(args.cams) as dataset:
        predictions = []
        for indicator, group in targets.groupby("indicator", sort=False):
            values = dataset[indicator].isel(time=0).interp(
                lon=xr.DataArray(group["longitude"].to_numpy(), dims="station"),
                lat=xr.DataArray(group["latitude"].to_numpy(), dims="station"),
                method="linear",
            )
            part = group.copy()
            part["cams"] = values.to_numpy()
            predictions.append(part)

    comparison = pd.concat(predictions, ignore_index=True).dropna(
        subset=["observed", "cams"]
    )
    overall = comparison.groupby("indicator").apply(metrics, include_groups=False)
    overall = overall.reset_index()
    by_type = comparison.groupby(["indicator", "stationstype"]).apply(
        metrics, include_groups=False
    )
    by_type = by_type.reset_index()

    comparison.to_csv(
        args.output_dir / f"cams_station_comparison_{args.month}.csv", index=False
    )
    overall.to_csv(
        args.output_dir / f"cams_baseline_metrics_{args.month}.csv", index=False
    )
    by_type.to_csv(
        args.output_dir / f"cams_baseline_metrics_by_station_type_{args.month}.csv",
        index=False,
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

