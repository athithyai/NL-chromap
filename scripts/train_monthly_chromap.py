"""Train a direct CHROMAP-style monthly ensemble and spatially validate it.

Inputs are a station table with value, x, y and feature columns, plus a 500 m
grid NetCDF containing those same named features. This runner deliberately has
no CAMS-plus-residual alternative. CAMS may be one predictor in the ensemble.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chromap_models import PAPER_NO2_FEATURES, PILOT_FEATURES, chromap_models


def scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias_pred_minus_obs": float(np.mean(predicted - observed)),
        "r2": float(r2_score(observed, predicted)),
    }


def fit_predict(indicator: str, feature_names: list[str], train: pd.DataFrame,
                target: pd.DataFrame) -> tuple[np.ndarray, int]:
    preprocess = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())])
    x_train = preprocess.fit_transform(train[feature_names])
    x_target = preprocess.transform(target[feature_names])
    y_train = train.value.to_numpy()
    members = []
    mlp_warnings = 0
    for name, estimator in chromap_models(indicator).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x_train, y_train)
        if name == "mlp":
            mlp_warnings += any(issubclass(item.category, ConvergenceWarning)
                                for item in caught)
        members.append(np.maximum(estimator.predict(x_target), float(y_train.min())))
    return np.median(np.vstack(members), axis=0), mlp_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pollutant", required=True, choices=("no2", "o3", "pm10", "pm2p5", "so2"))
    parser.add_argument("--stations", required=True, type=Path, help="CSV/CSV.GZ with value, x, y and features")
    parser.add_argument("--grid-features", required=True, type=Path, help="NetCDF with y, x, land_mask and features")
    parser.add_argument("--grid-mask", type=Path, help="NetCDF with land_mask and spatial_ref, if not in --grid-features")
    parser.add_argument("--feature-set", required=True, choices=("paper-no2", "monthly-pilot"))
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--engine", default="h5netcdf", help="xarray NetCDF read/write engine")
    args = parser.parse_args()
    if args.feature_set == "paper-no2" and args.pollutant != "no2":
        parser.error("paper-no2 feature set is only valid for NO2")
    feature_names = PAPER_NO2_FEATURES if args.feature_set == "paper-no2" else PILOT_FEATURES

    station = pd.read_csv(args.stations, parse_dates=["month"])
    if "indicator" in station:
        station = station[station.indicator.eq(args.pollutant)]
    station = station[station.month.dt.strftime("%Y-%m").eq(args.month)]
    if "is_valid" in station:
        station = station[station.is_valid.eq(True)]
    required = feature_names + ["value", "x", "y"]
    station = station[np.isfinite(station[required].to_numpy(dtype=float)).all(axis=1)].copy().reset_index(drop=True)
    if len(station) < 30:
        raise ValueError(f"Only {len(station)} eligible stations; check monthly targets and features")
    if {"source", "station_id"}.issubset(station) and station.duplicated(["source", "station_id"]).any():
        raise ValueError("Duplicate station rows after monthly filtering")
    station["fold"] = KMeans(n_clusters=5, random_state=42, n_init=50).fit_predict(station[["x", "y"]])
    station["prediction_held_out"] = np.nan
    fold_rows = []
    nonconverged = 0
    for fold in range(5):
        train = station[station.fold.ne(fold)]
        test = station[station.fold.eq(fold)]
        prediction, warnings_count = fit_predict(args.pollutant, feature_names, train, test)
        station.loc[test.index, "prediction_held_out"] = prediction
        nonconverged += warnings_count
        fold_rows.append({"fold": fold + 1, **scores(test.value.to_numpy(), prediction)})
        print(f"Completed spatial fold {fold + 1}/5", flush=True)

    with xr.open_dataset(args.grid_features, engine=args.engine) as ds:
        ds = ds.load()
    if args.grid_mask is not None:
        with xr.open_dataset(args.grid_mask, engine=args.engine) as grid:
            grid = grid.load()
    else:
        grid = ds
    if not ds.x.equals(grid.x) or not ds.y.equals(grid.y):
        raise ValueError("Grid feature and mask coordinates differ")
    mask = grid.land_mask.values.astype(bool)
    if mask.ndim != 2 or mask.sum() == 0:
        raise ValueError("Expected a non-empty 2D land_mask on the feature grid")
    flat = pd.DataFrame({feature: np.asarray(ds[feature].squeeze().values)[mask]
                         for feature in feature_names})
    final_prediction, final_warnings = fit_predict(args.pollutant, feature_names, station, flat)
    field = np.full(mask.shape, np.nan, dtype="float32")
    field[mask] = final_prediction.astype("float32")
    result = xr.Dataset(
        {args.pollutant: (("time", "y", "x"), field[None],
                          {"units": "ug m-3", "grid_mapping": "spatial_ref"}),
         "land_mask": (("y", "x"), mask.astype("uint8"))},
        coords={"time": [np.datetime64(args.month + "-01")], "x": ds.x, "y": ds.y},
        attrs={"method": "CHROMAP direct four-model median; monthly retraining",
               "feature_set": args.feature_set,
               "so2_preset": "project-defined, not author-supplied" if args.pollutant == "so2" else "not applicable"},
    )
    if "spatial_ref" in grid:
        result["spatial_ref"] = grid.spatial_ref
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.pollutant}_{args.month.replace('-', '_')}_chromap.nc"
    result.to_netcdf(output, engine=args.engine)
    station.to_csv(args.output_dir / "held_out_station_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary = {"pollutant": args.pollutant, "month": args.month, "feature_set": args.feature_set,
               "stations": len(station), "five_fold_held_out": scores(station.value.to_numpy(), station.prediction_held_out.to_numpy()),
               "mlp_nonconverged_folds": int(nonconverged),
               "mlp_nonconverged_final_fit": int(final_warnings),
               "validation_note": "Each held-out station was excluded from its fold's training; the final map uses all eligible stations."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
