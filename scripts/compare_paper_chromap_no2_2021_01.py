"""Paired validation and mapped comparison using the paper-defined NO2 stack."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_paper_no2_features_2021_01 import FEATURES
from compare_no2_2021_01_approaches import chromap_models
from train_pilot_models_2021 import FEATURES as NL_FEATURES


ROOT = Path("outputs/paper_chromap_no2_2021_01")
COLORS = ["#4f83b5", "#3d9fc0", "#55babe", "#83cbb5", "#bed8ae",
          "#e8dfa5", "#f3bd7b", "#e87b49", "#c94835"]


def metric(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"n": int(len(obs)), "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
            "mae": float(mean_absolute_error(obs, pred)),
            "bias": float(np.mean(pred - obs)), "r2": float(r2_score(obs, pred))}


def residual_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=.7,
                               bootstrap=False, n_jobs=-1, random_state=seed)


def direct_predict(x_train: pd.DataFrame, y_train: np.ndarray,
                   x_test: pd.DataFrame) -> tuple[np.ndarray, int]:
    preprocessing = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())])
    train = preprocessing.fit_transform(x_train)
    test = preprocessing.transform(x_test)
    predictions, mlp_warning = [], 0
    for name, estimator in chromap_models("no2").items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(train, y_train)
        if name == "mlp" and any(issubclass(item.category, ConvergenceWarning) for item in caught):
            mlp_warning += 1
        predictions.append(np.maximum(estimator.predict(test), float(y_train.min())))
    return np.median(np.vstack(predictions), axis=0), mlp_warning


def write_map(path: Path, values: np.ndarray, grid: xr.Dataset, title: str) -> None:
    cmap = LinearSegmentedColormap.from_list("reference", COLORS, N=256)
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6.4, 7.1))
    extent = [float(grid.x.min()) / 1000, float(grid.x.max()) / 1000,
              float(grid.y.min()) / 1000, float(grid.y.max()) / 1000]
    image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                      norm=Normalize(0, 40, clip=True))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("EPSG:3035 easting (km)")
    ax.set_ylabel("EPSG:3035 northing (km)")
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=.08,
                 label="Monthly mean NO₂ (µg m⁻³); common 0–40 scale")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    paper = pd.read_csv("data/processed/model/paper_no2_2021_01/paper_no2_station_features_2021_01.csv.gz",
                        parse_dates=["month"])
    required = FEATURES + NL_FEATURES + ["value"]
    paper = paper[np.isfinite(paper[required]).all(axis=1)].reset_index(drop=True)
    paper["fold"] = KMeans(n_clusters=5, random_state=42, n_init=50).fit_predict(paper[["x", "y"]])
    methods = ["cams", "paper_chromap", "nl_original_features", "nl_same_paper_features"]
    predictions = {name: np.full(len(paper), np.nan) for name in methods}
    fold_rows, mlp_warnings = [], 0
    for fold in range(5):
        train, test = paper.fold != fold, paper.fold == fold
        y_train = paper.loc[train, "value"].values
        cams_train = paper.loc[train, "CHEM_CTM_no2"].values
        predictions["cams"][test] = paper.loc[test, "CHEM_CTM_no2"]
        predictions["paper_chromap"][test], warned = direct_predict(
            paper.loc[train, FEATURES], y_train, paper.loc[test, FEATURES])
        mlp_warnings += warned
        original = residual_model(100 + fold)
        original.fit(paper.loc[train, NL_FEATURES], y_train - cams_train)
        predictions["nl_original_features"][test] = np.clip(
            paper.loc[test, "CHEM_CTM_no2"].values + original.predict(paper.loc[test, NL_FEATURES]), 0, None)
        same = residual_model(200 + fold)
        same.fit(paper.loc[train, FEATURES], y_train - cams_train)
        predictions["nl_same_paper_features"][test] = np.clip(
            paper.loc[test, "CHEM_CTM_no2"].values + same.predict(paper.loc[test, FEATURES]), 0, None)
        for name in methods:
            fold_rows.append({"fold": fold + 1, "method": name,
                              **metric(paper.loc[test, "value"].values, predictions[name][test])})
        print(f"completed fold {fold + 1}", flush=True)
    for name, values in predictions.items():
        paper[f"pred_{name}"] = values
    metrics = {name: metric(paper.value.values, values) for name, values in predictions.items()}
    paper[["source", "country", "station_id", "x", "y", "fold", "value"] +
          [f"pred_{name}" for name in methods]].to_csv(ROOT / "paired_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "fold_metrics.csv", index=False)

    feature_grid = xr.open_dataset("data/processed/model/paper_no2_2021_01/paper_no2_grid_features_2021_01.nc")
    grid = xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc")
    mask = grid.land_mask.values == 1
    grid_table = pd.DataFrame({name: feature_grid[name].values[mask] for name in FEATURES})
    paper_map, _ = direct_predict(paper[FEATURES], paper.value.values, grid_table)
    same = residual_model(242)
    same.fit(paper[FEATURES], paper.value.values - paper.CHEM_CTM_no2.values)
    same_map = np.clip(grid_table.CHEM_CTM_no2.values + same.predict(grid_table), 0, None)
    map_fields = {}
    for name, values in (("paper_chromap", paper_map), ("nl_same_paper_features", same_map)):
        field = np.full(mask.shape, np.nan, dtype="float32")
        field[mask] = values
        map_fields[name] = field
    with xr.open_dataset("outputs/pilot_2021/chromap_nl_500m_2021_01.nc") as existing:
        map_fields["nl_original_features"] = existing.no2.squeeze().values.astype("float32")

    coords = {"time": [np.datetime64("2021-01-01")], "y": grid.y.values, "x": grid.x.values}
    out = xr.Dataset({
        "no2_paper_chromap": (("time", "y", "x"), map_fields["paper_chromap"][None]),
        "no2_nl_original_features": (("time", "y", "x"), map_fields["nl_original_features"][None]),
        "no2_nl_same_paper_features": (("time", "y", "x"), map_fields["nl_same_paper_features"][None]),
        "spatial_ref": grid.spatial_ref.copy(deep=True),
    }, coords=coords, attrs={
        "title": "January 2021 NO2: paper-faithful monthly CHROMAP adaptation versus NL residual methods",
        "units": "ug m-3", "crs": "EPSG:3035", "resolution_m": 500,
        "paper_scope_deviation": "Paper maps are annual 2019-2023; this experiment applies its documented NO2 stack and fixed estimators to January 2021.",
    })
    for name in out.data_vars:
        if name != "spatial_ref":
            out[name].attrs.update(units="ug m-3", grid_mapping="spatial_ref")
    nc_path = ROOT / "no2_2021_01_paper_chromap_vs_nl.nc"
    out.to_netcdf(nc_path, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0} for name in out.data_vars if name != "spatial_ref"})
    write_map(ROOT / "no2_2021_01_paper_chromap.png", map_fields["paper_chromap"], grid,
              "January 2021 NO₂ — paper-faithful CHROMAP adaptation")
    write_map(ROOT / "no2_2021_01_nl_original.png", map_fields["nl_original_features"], grid,
              "January 2021 NO₂ — NL residual approach")

    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "station_rows": len(paper), "features": FEATURES, "metrics": metrics,
               "mlp_nonconverged_folds": mlp_warnings,
               "interpretation": {"primary_comparison": ["paper_chromap", "nl_original_features"],
                                  "controlled_model_comparison": ["paper_chromap", "nl_same_paper_features"]}}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
