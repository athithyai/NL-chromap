"""Spatially validate and fit 2021 CHROMAP-NL residual models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "cams_value", "log1p_population_density", "elevation",
    "landcover_tree_fraction", "landcover_shrub_fraction", "landcover_grass_fraction",
    "landcover_cropland_fraction", "landcover_built_up_fraction",
    "landcover_bare_sparse_fraction", "landcover_permanent_water_fraction",
    "landcover_wetland_fraction", "road_density_major", "road_density_secondary",
    "road_density_local", "road_density_motorized_total", "era5_u10", "era5_v10",
    "era5_t2m", "era5_blh", "era5_sp", "era5_tcc", "era5_ssrd", "era5_tp",
    "era5_wind_speed", "month_sin", "month_cos",
]
INDICATORS = ("no2", "o3", "pm10", "pm2p5", "somo35")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model(kind: str, trees: int, seed: int):
    common = dict(
        n_estimators=trees, min_samples_leaf=3, max_features=0.7,
        n_jobs=-1, random_state=seed,
    )
    if kind == "random_forest":
        return RandomForestRegressor(bootstrap=True, **common)
    return ExtraTreesRegressor(bootstrap=False, **common)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": len(observed), "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias": float(np.mean(predicted - observed)), "r2": float(r2_score(observed, predicted)),
    }


def main() -> None:
    source = Path("data/processed/model/station_predictors_2021.csv.gz")
    output_dir = Path("data/processed/model/validated_2021")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, parse_dates=["month"])
    station = frame.drop_duplicates(["source", "station_id"])[["source", "station_id", "x", "y"]].copy()
    station["spatial_fold"] = KMeans(n_clusters=5, random_state=42, n_init=50).fit_predict(station[["x", "y"]])
    frame = frame.merge(station[["source", "station_id", "spatial_fold"]], on=["source", "station_id"])
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    selected: dict[str, str] = {}

    for indicator in INDICATORS:
        data = frame[(frame.indicator == indicator) & frame.value.notna() & frame.is_valid].copy()
        data = data[np.isfinite(data[FEATURES]).all(axis=1)]
        baseline = np.clip(data.cams_value.values, 0, None)
        baseline_metrics = metrics(data.value.values, baseline)
        metric_rows.append({"indicator": indicator, "model": "cams_baseline", **baseline_metrics})
        data["pred_cams_baseline"] = baseline

        candidate_rmse: dict[str, float] = {"cams_baseline": baseline_metrics["rmse"]}
        for kind in ("random_forest", "extra_trees"):
            oof = np.full(len(data), np.nan)
            for fold in sorted(data.spatial_fold.unique()):
                train = data.spatial_fold != fold
                test = ~train
                estimator = model(kind, trees=150, seed=42 + int(fold))
                estimator.fit(data.loc[train, FEATURES], data.loc[train, "value"] - data.loc[train, "cams_value"])
                residual = estimator.predict(data.loc[test, FEATURES])
                oof[test] = np.clip(data.loc[test, "cams_value"].values + residual, 0, None)
            data[f"pred_{kind}"] = oof
            result = metrics(data.value.values, oof)
            candidate_rmse[kind] = result["rmse"]
            metric_rows.append({"indicator": indicator, "model": kind, **result})

        best = min(candidate_rmse, key=candidate_rmse.get)
        selected[indicator] = best
        data["selected_model"] = best
        data["prediction"] = data[f"pred_{best}"]
        predictions.append(data)

        if best != "cams_baseline":
            final = model(best, trees=400, seed=42)
            final.fit(data[FEATURES], data.value - data.cams_value)
            joblib.dump({"model": final, "features": FEATURES, "indicator": indicator}, output_dir / f"{indicator}_{best}.joblib")
            importance = pd.DataFrame({"feature": FEATURES, "importance": final.feature_importances_}).sort_values("importance", ascending=False)
            importance.to_csv(output_dir / f"{indicator}_{best}_importance.csv", index=False)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_path = output_dir / "spatial_cv_predictions_2021.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
    metric_frame = pd.DataFrame(metric_rows)
    metric_path = output_dir / "spatial_cv_metrics_2021.csv"
    metric_frame.to_csv(metric_path, index=False)
    selected_path = output_dir / "selected_models.json"
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "features": FEATURES, "spatial_folds": 5, "selected_models": selected,
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (prediction_path, metric_path, selected_path)
        ],
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metric_frame.to_string(index=False))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
