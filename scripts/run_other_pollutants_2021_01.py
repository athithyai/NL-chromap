"""Create and validate January 2021 O3, PM10, PM2.5, and SO2 comparisons."""

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
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compare_no2_2021_01_approaches import (
    COLORS,
    FEATURES,
    POSITIONS,
    chromap_models,
    field,
    grid_feature_table,
    statistics,
)
from validate_no2_2021_01_approaches import metrics, paired_bootstrap


ROOT = Path("outputs/comparison_2021_01_other_pollutants")
INDICATORS = ("o3", "pm10", "pm2p5", "so2")
META = {
    "o3": {"label": "O₃", "math_label": "O$_3$", "long_name": "ozone", "unit": "µg m⁻³", "unit_nc": "ug m-3", "vmax": 60},
    "pm10": {"label": "PM₁₀", "math_label": "PM$_{10}$", "long_name": "particulate matter below 10 micrometres", "unit": "µg m⁻³", "unit_nc": "ug m-3", "vmax": 40},
    "pm2p5": {"label": "PM₂.₅", "math_label": "PM$_{2.5}$", "long_name": "particulate matter below 2.5 micrometres", "unit": "µg m⁻³", "unit_nc": "ug m-3", "vmax": 25},
    "so2": {"label": "SO₂", "math_label": "SO$_2$", "long_name": "sulphur dioxide", "unit": "µg m⁻³", "unit_nc": "ug m-3", "vmax": 10},
}
MODEL_LABELS = {
    "cams_baseline": "CAMS baseline",
    "chromap_style": "CHROMAP-style",
    "nl_residual": "NL residual candidate",
}
MODEL_COLORS = {"cams_baseline": "#8a8a8a", "chromap_style": "#d95f02", "nl_residual": "#1b9e77"}


def station_data(indicator: str) -> pd.DataFrame:
    path = ("data/processed/model/so2_station_predictors_2021_01.csv.gz" if indicator == "so2"
            else "data/processed/model/station_predictors_2021.csv.gz")
    data = pd.read_csv(path, parse_dates=["month"])
    data = data[(data.indicator == indicator) & (data.month.dt.month == 1)
                & data.value.notna() & data.is_valid].copy()
    data = data[np.isfinite(data[FEATURES]).all(axis=1)].reset_index(drop=True)
    if data.duplicated(["source", "station_id"]).any():
        raise ValueError(f"duplicate January station rows for {indicator}")
    return data


def validate(indicator: str, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data = data.copy()
    data["fold"] = KMeans(n_clusters=5, random_state=42, n_init=50).fit_predict(data[["x", "y"]])
    predictions = np.full((len(data), 3), np.nan)
    mlp_nonconverged = 0
    for fold_number in sorted(data.fold.unique()):
        train = data.fold != fold_number
        test = ~train
        x_train, x_test = data.loc[train, FEATURES], data.loc[test, FEATURES]
        y_train = data.loc[train, "value"].to_numpy()
        cams_test = data.loc[test, "cams_value"].to_numpy()
        predictions[test, 0] = np.clip(cams_test, 0, None)

        preprocessing = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])
        x_train_scaled = preprocessing.fit_transform(x_train)
        x_test_scaled = preprocessing.transform(x_test)
        direct = []
        for name, estimator in chromap_models(indicator).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train_scaled, y_train)
            if name == "mlp" and any(issubclass(item.category, ConvergenceWarning) for item in caught):
                mlp_nonconverged += 1
            direct.append(np.maximum(estimator.predict(x_test_scaled), float(y_train.min())))
        predictions[test, 1] = np.median(np.vstack(direct), axis=0)

        residual = ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.7,
            bootstrap=False, n_jobs=-1, random_state=42 + int(fold_number),
        )
        residual.fit(x_train, y_train - data.loc[train, "cams_value"].to_numpy())
        predictions[test, 2] = np.clip(cams_test + residual.predict(x_test), 0, None)
        print(f"{indicator}: fold {int(fold_number) + 1}/5", flush=True)

    for name, values in zip(MODEL_LABELS, predictions.T):
        data[f"pred_{name}"] = values
    overall = {name: metrics(data.value.to_numpy(), data[f"pred_{name}"].to_numpy())
               for name in MODEL_LABELS}
    selected = min(("cams_baseline", "nl_residual"), key=lambda name: overall[name]["rmse"])
    data["pred_nl_selected"] = data[f"pred_{selected}"]
    overall["nl_selected"] = metrics(data.value.to_numpy(), data.pred_nl_selected.to_numpy())
    fold_rows = []
    for fold_number in sorted(data.fold.unique()):
        group = data[data.fold == fold_number]
        for name in MODEL_LABELS:
            fold_rows.append({"indicator": indicator, "fold": int(fold_number) + 1,
                              "model": name, **metrics(group.value.to_numpy(), group[f"pred_{name}"].to_numpy())})
    fold_frame = pd.DataFrame(fold_rows)
    bootstrap = paired_bootstrap(data.value.to_numpy(), data.pred_chromap_style.to_numpy(),
                                 data.pred_nl_selected.to_numpy())
    bootstrap["definition"] = (
        "CHROMAP-style error minus NL-selected error; positive favors NL-selected"
    )
    winner = min(("chromap_style", "nl_selected"), key=lambda name: overall[name]["rmse"])
    summary = {
        "n": len(data), "fold_sizes": {str(int(k) + 1): int(v) for k, v in data.fold.value_counts().sort_index().items()},
        "overall_metrics": overall, "nl_selected_model": selected,
        "head_to_head_winner": winner, "paired_bootstrap": bootstrap,
        "mlp_nonconverged_folds": mlp_nonconverged,
        "chromap_so2_note": ("The supplied CHROMAP archive has no SO2 preset; a conservative fixed four-model extension was used."
                              if indicator == "so2" else None),
    }
    return data, fold_frame, summary


def final_maps(indicator: str, data: pd.DataFrame, selected: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc") as grid:
        mask = grid.land_mask.values == 1
    grid_table = grid_feature_table(mask, indicator)
    preprocessing = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])
    grid_scaled = preprocessing.fit_transform(grid_table)
    station_scaled = preprocessing.transform(data[FEATURES])
    direct_members = {}
    for name, estimator in chromap_models(indicator).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(station_scaled, data.value)
        direct_members[name] = np.maximum(estimator.predict(grid_scaled), float(data.value.min()))
    stack = np.vstack(list(direct_members.values()))
    chromap = field(np.median(stack, axis=0), mask)
    spread = field(np.std(stack, axis=0), mask)
    cams_values = grid_table.cams_value.to_numpy()
    if selected == "nl_residual":
        residual = ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.7,
            bootstrap=False, n_jobs=-1, random_state=42,
        )
        residual.fit(data[FEATURES], data.value - data.cams_value)
        nl_values = np.clip(cams_values + residual.predict(grid_table[FEATURES]), 0, None)
    else:
        nl_values = np.clip(cams_values, 0, None)
    nl = field(nl_values, mask)
    return chromap, nl, spread, {name: field(values, mask) for name, values in direct_members.items()}


def write_products(indicator: str, chromap: np.ndarray, nl: np.ndarray, spread: np.ndarray,
                   members: dict[str, np.ndarray], selected: str, selected_rmse: float,
                   stations: pd.DataFrame) -> None:
    directory = ROOT / indicator
    directory.mkdir(parents=True, exist_ok=True)
    meta = META[indicator]
    with xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc") as grid:
        x, y = grid.x.load(), grid.y.load()
        spatial_ref = grid.spatial_ref.load()
        mask = grid.land_mask.values == 1
    coords = {"time": [np.datetime64("2021-01-01")], "y": y, "x": x}
    chromap_vars = {
        indicator: (("time", "y", "x"), chromap[None], {"units": meta["unit_nc"], "grid_mapping": "spatial_ref"}),
        f"{indicator}_ensemble_spread": (("time", "y", "x"), spread[None], {"units": meta["unit_nc"], "grid_mapping": "spatial_ref"}),
        "spatial_ref": spatial_ref,
    }
    for name, values in members.items():
        chromap_vars[f"{indicator}_{name}"] = (("time", "y", "x"), values[None],
                                                {"units": meta["unit_nc"], "grid_mapping": "spatial_ref"})
    chromap_ds = xr.Dataset(chromap_vars, coords=coords, attrs={
        "title": f"January 2021 {meta['label']}: CHROMAP-style monthly adaptation",
        "method": "Direct Ridge, Random Forest, XGBoost, and MLP median ensemble",
        "target_crs": "EPSG:3035", "resolution_m": 500,
    })
    iy, ix = np.where(mask)
    station_tree = cKDTree(stations[["x", "y"]].values)
    distances, _ = station_tree.query(np.column_stack((x.values[ix], y.values[iy])))
    uncertainty_values = selected_rmse * (1 + np.minimum(distances / 1000, 100) / 200)
    uncertainty = field(uncertainty_values, mask)
    nl_ds = xr.Dataset({
        indicator: (("time", "y", "x"), nl[None], {"units": meta["unit_nc"], "grid_mapping": "spatial_ref"}),
        f"{indicator}_uncertainty": (("time", "y", "x"), uncertainty[None],
                                      {"units": meta["unit_nc"], "grid_mapping": "spatial_ref",
                                       "long_name": "Distance-inflated spatial-CV RMSE proxy"}),
        "spatial_ref": spatial_ref,
    }, coords=coords, attrs={
        "title": f"January 2021 {meta['label']}: NL validated hybrid",
        "selected_model": selected, "target_crs": "EPSG:3035", "resolution_m": 500,
    })
    for path, dataset in (
        (directory / f"{indicator}_2021_01_chromap_style.nc", chromap_ds),
        (directory / f"{indicator}_2021_01_nl_selected.nc", nl_ds),
    ):
        encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
                    for name in dataset.data_vars if name != "spatial_ref"}
        dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    render_maps(indicator, chromap, nl, x.values, y.values, selected, directory)


def render_maps(indicator: str, chromap: np.ndarray, nl: np.ndarray, x: np.ndarray,
                y: np.ndarray, selected: str, directory: Path) -> None:
    meta = META[indicator]
    cmap = LinearSegmentedColormap.from_list("reference_blue_green_orange",
                                             list(zip(POSITIONS, COLORS)), N=256)
    cmap.set_bad("white")
    extent = [x.min() / 1000, x.max() / 1000, y.min() / 1000, y.max() / 1000]
    norm = Normalize(0, meta["vmax"], clip=True)
    for values, title, filename in (
        (chromap, "CHROMAP-style direct ensemble", f"{indicator}_2021_01_chromap_style.png"),
        (nl, f"NL selected: {MODEL_LABELS[selected]}", f"{indicator}_2021_01_nl_selected.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor="white")
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Netherlands {meta['label']} — January 2021\n{title}", fontweight="bold")
        ax.set_xlabel("EPSG:3035 easting (km)"); ax.set_ylabel("EPSG:3035 northing (km)")
        fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.09,
                     label=f"Monthly mean {meta['label']} ({meta['unit']}); common 0–{meta['vmax']} scale")
        fig.tight_layout(); fig.savefig(directory / filename, dpi=220, bbox_inches="tight"); plt.close(fig)
    difference = chromap - nl
    limit = max(float(np.percentile(np.abs(difference[np.isfinite(difference)]), 98)), 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.82, wspace=0.16)
    for ax, values, title in ((axes[0], chromap, "CHROMAP-style direct ensemble"),
                              (axes[1], nl, f"NL selected: {MODEL_LABELS[selected]}")):
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontweight="bold", pad=10)
    axes[0].set_ylabel("Northing (km)")
    diff_image = axes[2].imshow(difference, origin="lower", extent=extent, cmap="RdBu_r",
                                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
    axes[2].set_title("Difference: CHROMAP-style − NL", fontweight="bold", pad=10)
    fig.colorbar(image, ax=axes[:2], orientation="horizontal", fraction=0.055, pad=0.08,
                 label=f"Monthly mean {meta['label']} ({meta['unit']}); common 0–{meta['vmax']} scale")
    fig.colorbar(diff_image, ax=axes[2], orientation="horizontal", fraction=0.055, pad=0.08,
                 label=f"Difference ({meta['unit']})")
    fig.suptitle(f"Netherlands {meta['label']} — January 2021 modelling comparison",
                 fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(directory / f"{indicator}_2021_01_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_validation_summary(summaries: dict[str, object]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    for ax, indicator in zip(axes.ravel(), INDICATORS):
        metrics_by_model = summaries[indicator]["overall_metrics"]
        names = ["cams_baseline", "chromap_style", "nl_residual"]
        values = [metrics_by_model[name]["rmse"] for name in names]
        bars = ax.bar([MODEL_LABELS[name] for name in names], values,
                      color=[MODEL_COLORS[name] for name in names])
        ax.set_ylim(bottom=0); ax.set_ylabel("RMSE (µg m$^{-3}$)")
        ax.set_title(f"{META[indicator]['math_label']} — {summaries[indicator]['n']} held-out stations",
                     fontweight="bold")
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle("January 2021 paired five-fold spatial validation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "all_pollutants_spatial_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    all_predictions, all_folds = [], []
    for indicator in INDICATORS:
        data = station_data(indicator)
        validated, fold_frame, summary = validate(indicator, data)
        chromap, nl, spread, members = final_maps(indicator, data, summary["nl_selected_model"])
        write_products(indicator, chromap, nl, spread, members, summary["nl_selected_model"],
                       summary["overall_metrics"]["nl_selected"]["rmse"], data)
        summary["map_statistics"] = {
            "chromap_style": statistics(chromap), "nl_selected": statistics(nl),
            "difference_chromap_minus_nl": statistics(chromap - nl),
        }
        validated.insert(0, "indicator_name", indicator)
        all_predictions.append(validated)
        all_folds.append(fold_frame)
        summaries[indicator] = summary
        print(f"completed {indicator}", flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(ROOT / "paired_oof_predictions.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(ROOT / "fold_metrics.csv", index=False)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "indicators": summaries,
              "scope": "January 2021 only", "somo35_excluded": True}
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_validation_summary(summaries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
