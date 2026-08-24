"""Compare CHROMAP-style direct ensemble and NL residual NO2 for January 2021.

The supplied CHROMAP v1.0 implementation was designed for annual inputs. This
script makes a transparent monthly adaptation: it trains the same four direct
regression families with the supplied NO2 hyperparameters on January station
targets, applies the reference lower-bound rule, and takes their median. The NL
result is the already validated pooled-2021 CAMS + Extra Trees residual product.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train_pilot_models_2021 import FEATURES


OUTPUT_DIR = Path("outputs/comparison_2021_01_no2")
COLORS = [
    "#4f83b5", "#3d9fc0", "#55babe", "#83cbb5", "#bed8ae",
    "#e8dfa5", "#f3bd7b", "#e87b49", "#c94835",
]
POSITIONS = [0.00, 0.14, 0.28, 0.43, 0.58, 0.70, 0.82, 0.92, 1.00]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_feature_table(mask: np.ndarray, indicator: str = "no2") -> pd.DataFrame:
    """Build January grid predictors in the same order used at stations."""
    with (
        xr.open_dataset("data/processed/predictors/static_predictors_nl_500m_2021.nc") as static,
        xr.open_dataset("data/processed/predictors/osm_road_density_nl_500m.nc") as roads,
        xr.open_dataset("data/processed/predictors/era5_monthly_predictors_nl_500m_2021.nc") as met,
        xr.open_dataset(
            "data/processed/baseline/cams_baseline_nl_500m_2021_01_so2.nc"
            if indicator == "so2"
            else "data/processed/baseline/cams_baseline_nl_500m_2021_01.nc"
        ) as cams,
    ):
        values = {
            "cams_value": cams[indicator].squeeze().values[mask],
            "log1p_population_density": static.log1p_population_density.values[mask],
            "elevation": static.elevation.values[mask],
            "landcover_tree_fraction": static.landcover_tree_fraction.values[mask],
            "landcover_shrub_fraction": static.landcover_shrub_fraction.values[mask],
            "landcover_grass_fraction": static.landcover_grass_fraction.values[mask],
            "landcover_cropland_fraction": static.landcover_cropland_fraction.values[mask],
            "landcover_built_up_fraction": static.landcover_built_up_fraction.values[mask],
            "landcover_bare_sparse_fraction": static.landcover_bare_sparse_fraction.values[mask],
            "landcover_permanent_water_fraction": static.landcover_permanent_water_fraction.values[mask],
            "landcover_wetland_fraction": static.landcover_wetland_fraction.values[mask],
            "road_density_major": roads.road_density_major.values[mask],
            "road_density_secondary": roads.road_density_secondary.values[mask],
            "road_density_local": roads.road_density_local.values[mask],
            "road_density_motorized_total": roads.road_density_motorized_total.values[mask],
            "era5_u10": met.u10.isel(time=0).values[mask],
            "era5_v10": met.v10.isel(time=0).values[mask],
            "era5_t2m": met.t2m.isel(time=0).values[mask],
            "era5_blh": met.blh.isel(time=0).values[mask],
            "era5_sp": met.sp.isel(time=0).values[mask],
            "era5_tcc": met.tcc.isel(time=0).values[mask],
            "era5_ssrd": met.ssrd.isel(time=0).values[mask],
            "era5_tp": met.tp.isel(time=0).values[mask],
            "era5_wind_speed": met.wind_speed_10m.isel(time=0).values[mask],
            "month_sin": np.full(mask.sum(), np.sin(2 * np.pi / 12)),
            "month_cos": np.full(mask.sum(), np.cos(2 * np.pi / 12)),
        }
    return pd.DataFrame(values)[FEATURES]


def chromap_models(indicator: str = "no2") -> dict[str, object]:
    """Use the supplied CHROMAP v1.0 fixed pollutant hyperparameters."""
    if indicator == "o3":
        ridge = {"alpha": 3.5}
        forest = dict(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=0.7, sampling_method="uniform", n_estimators=1000,
                       min_child_weight=5, max_depth=2, max_delta_step=3,
                       learning_rate=0.3, reg_lambda=100, gamma=0.4,
                       colsample_bytree=0.5, reg_alpha=100)
        mlp = dict(solver="lbfgs", momentum=0.7, learning_rate_init=0.01,
                   learning_rate="constant", hidden_layer_sizes=(100, 100, 100),
                   alpha=0.5, activation="identity")
    elif indicator == "somo35":
        ridge = {"alpha": 10.9}
        forest = dict(n_estimators=1500, min_samples_split=5, min_samples_leaf=2,
                      max_features=1.0, max_depth=50, bootstrap=True)
        boosted = dict(subsample=1.0, sampling_method="uniform", n_estimators=1500,
                       min_child_weight=1, max_depth=6, max_delta_step=0,
                       learning_rate=0.3, reg_lambda=1, gamma=0,
                       colsample_bytree=1.0, reg_alpha=0)
        mlp = dict(solver="lbfgs", momentum=0.5, learning_rate_init=0.2,
                   learning_rate="constant", hidden_layer_sizes=(50,),
                   alpha=0.1, activation="identity")
    elif indicator in {"pm10", "pm2p5"}:
        ridge = {"alpha": 2.0}
        forest = dict(n_estimators=1000, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=30, bootstrap=True)
        boosted = dict(subsample=0.5, sampling_method="uniform", n_estimators=1500,
                       min_child_weight=7, max_depth=5, max_delta_step=7,
                       learning_rate=0.01, reg_lambda=10, gamma=0.4,
                       colsample_bytree=1.0, reg_alpha=1)
        mlp = dict(solver="adam", momentum=0.9, learning_rate_init=0.001,
                   learning_rate="invscaling", hidden_layer_sizes=(150,),
                   alpha=0.5, activation="tanh")
    elif indicator == "no2":
        ridge = {"alpha": 2.9}
        forest = dict(n_estimators=500, min_samples_split=10, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=1.0, sampling_method="uniform", n_estimators=1000,
                       min_child_weight=1, max_depth=5, max_delta_step=5,
                       learning_rate=0.01, reg_lambda=10, gamma=1,
                       colsample_bytree=0.5, reg_alpha=100)
        mlp = dict(solver="adam", momentum=0.9, learning_rate_init=0.001,
                   learning_rate="constant", hidden_layer_sizes=(150, 150, 150),
                   alpha=0.0001, activation="logistic")
    elif indicator == "so2":
        # The supplied archive has no fixed SO2 preset. This conservative direct-
        # ensemble extension stays inside its model families and parameter ranges.
        ridge = {"alpha": 3.0}
        forest = dict(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=0.8, sampling_method="uniform", n_estimators=500,
                       min_child_weight=3, max_depth=2, max_delta_step=3,
                       learning_rate=0.03, reg_lambda=10, gamma=0.1,
                       colsample_bytree=0.7, reg_alpha=1)
        mlp = dict(solver="lbfgs", momentum=0.9, learning_rate_init=0.001,
                   learning_rate="constant", hidden_layer_sizes=(50,),
                   alpha=0.1, activation="relu")
    else:
        raise ValueError(f"unsupported CHROMAP pollutant: {indicator}")
    return {
        "ridge": Ridge(**ridge),
        "random_forest": RandomForestRegressor(**forest, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(**boosted, random_state=42, n_jobs=-1, tree_method="hist"),
        "mlp": MLPRegressor(**mlp, random_state=42, max_iter=200),
    }


def field(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(mask.shape, np.nan, dtype="float32")
    result[mask] = values.astype("float32")
    return result


def statistics(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    return {
        "count": int(valid.size), "min": float(valid.min()),
        "p05": float(np.percentile(valid, 5)), "mean": float(valid.mean()),
        "median": float(np.median(valid)), "p95": float(np.percentile(valid, 95)),
        "max": float(valid.max()), "std": float(valid.std()),
    }


def write_netcdf(path: Path, dataset: xr.Dataset) -> None:
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
        for name in dataset.data_vars if name != "spatial_ref"
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def render_map(path: Path, values: np.ndarray, extent: list[float], title: str) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "reference_blue_green_orange", list(zip(POSITIONS, COLORS)), N=256
    )
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor="white")
    image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                      norm=Normalize(0, 40, clip=True), interpolation="nearest")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("EPSG:3035 easting (km)")
    ax.set_ylabel("EPSG:3035 northing (km)")
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.09)
    colorbar.set_label("Monthly mean NO₂ (µg m⁻³); common 0–40 scale")
    colorbar.set_ticks([0, 10, 20, 30, 40])
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_comparison(path: Path, chromap: np.ndarray, nl: np.ndarray,
                      extent: list[float]) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "reference_blue_green_orange", list(zip(POSITIONS, COLORS)), N=256
    )
    cmap.set_bad("#ffffff")
    difference = chromap - nl
    limit = float(np.percentile(np.abs(difference[np.isfinite(difference)]), 98))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.82, wspace=0.16)
    for ax, values, title in (
        (axes[0], chromap, "CHROMAP-style direct ensemble"),
        (axes[1], nl, "NL CAMS + residual correction"),
    ):
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                          norm=Normalize(0, 40, clip=True), interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylabel("Northing (km)")
    diff_image = axes[2].imshow(
        difference, origin="lower", extent=extent, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest",
    )
    axes[2].set_title("Difference: CHROMAP-style − NL", fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(image, ax=axes[:2], orientation="horizontal", fraction=0.055, pad=0.08,
                 label="Monthly mean NO₂ (µg m⁻³); common 0–40 scale")
    fig.colorbar(diff_image, ax=axes[2], orientation="horizontal", fraction=0.055, pad=0.08,
                 label="Difference (µg m⁻³)")
    fig.suptitle("Netherlands NO₂ — January 2021 modelling comparison",
                 fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc") as grid:
        mask = grid.land_mask.values == 1
        x = grid.x.values.copy()
        y = grid.y.values.copy()
        spatial_ref = grid.spatial_ref.copy(deep=True)

    stations = pd.read_csv(
        "data/processed/model/station_predictors_2021.csv.gz", parse_dates=["month"]
    )
    stations = stations[
        (stations.indicator == "no2") & (stations.month.dt.month == 1)
        & stations.value.notna() & stations.is_valid
    ].copy()
    stations = stations[np.isfinite(stations[FEATURES]).all(axis=1)]
    grid_table = grid_feature_table(mask)

    # The reference implementation standardizes predictors before fitting.
    transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])
    grid_scaled = transformer.fit_transform(grid_table)
    station_scaled = transformer.transform(stations[FEATURES])
    observed = stations.value.to_numpy()
    lower_bound = float(observed.min())

    member_predictions: dict[str, np.ndarray] = {}
    model_diagnostics: dict[str, object] = {}
    for name, estimator in chromap_models().items():
        estimator.fit(station_scaled, observed)
        member_predictions[name] = np.maximum(estimator.predict(grid_scaled), lower_bound)
        if name == "mlp":
            model_diagnostics["mlp_iterations"] = int(estimator.n_iter_)
            model_diagnostics["mlp_converged"] = bool(estimator.n_iter_ < estimator.max_iter)
        print(f"fitted {name}")
    stack = np.vstack(list(member_predictions.values()))
    chromap_values = np.median(stack, axis=0)
    chromap_spread = np.std(stack, axis=0)
    chromap_field = field(chromap_values, mask)

    with xr.open_dataset("outputs/pilot_2021/chromap_nl_500m_2021_01.nc") as source:
        nl_field = source.no2.squeeze().values.astype("float32")
        nl_uncertainty = source.no2_uncertainty.squeeze().values.astype("float32")

    created = datetime.now(timezone.utc).isoformat()
    coords = {
        "time": [np.datetime64("2021-01-01")],
        "y": xr.DataArray(y, dims="y", attrs={
            "standard_name": "projection_y_coordinate", "units": "m", "axis": "Y",
        }),
        "x": xr.DataArray(x, dims="x", attrs={
            "standard_name": "projection_x_coordinate", "units": "m", "axis": "X",
        }),
    }
    chromap_vars = {
        "no2": (("time", "y", "x"), chromap_field[None], {
            "units": "ug m-3", "long_name": "Monthly mean nitrogen dioxide",
            "grid_mapping": "spatial_ref",
        }),
        "no2_ensemble_spread": (("time", "y", "x"), field(chromap_spread, mask)[None],
                                {"units": "ug m-3", "long_name": "Standard deviation across four direct models",
                                 "grid_mapping": "spatial_ref"}),
        "spatial_ref": spatial_ref,
    }
    for name, values in member_predictions.items():
        chromap_vars[f"no2_{name}"] = (("time", "y", "x"), field(values, mask)[None],
                                       {"units": "ug m-3", "grid_mapping": "spatial_ref"})
    chromap_ds = xr.Dataset(chromap_vars, coords=coords, attrs={
        "title": "January 2021 NO2: CHROMAP-style monthly adaptation",
        "method": "Direct Ridge, Random Forest, XGBoost, and MLP predictions; median ensemble",
        "training_scope": "258 valid January 2021 station targets",
        "reference_difference": "Supplied CHROMAP v1.0 is annual; this is a monthly adaptation on the NL feature stack",
        "target_crs": "EPSG:3035", "resolution_m": 500, "created_utc": created,
    })
    nl_ds = xr.Dataset({
        "no2": (("time", "y", "x"), nl_field[None], {
            "units": "ug m-3", "long_name": "Monthly mean nitrogen dioxide",
            "grid_mapping": "spatial_ref",
        }),
        "no2_uncertainty": (("time", "y", "x"), nl_uncertainty[None], {
            "units": "ug m-3", "grid_mapping": "spatial_ref",
        }),
        "spatial_ref": spatial_ref,
    }, coords=coords, attrs={
        "title": "January 2021 NO2: NL residual-downscaling approach",
        "method": "CAMS baseline plus pooled-2021 spatially validated Extra Trees residual correction",
        "target_crs": "EPSG:3035", "resolution_m": 500, "created_utc": created,
    })

    chromap_path = OUTPUT_DIR / "no2_2021_01_chromap_style.nc"
    nl_path = OUTPUT_DIR / "no2_2021_01_nl_residual.nc"
    write_netcdf(chromap_path, chromap_ds)
    write_netcdf(nl_path, nl_ds)

    extent = [float(x.min()) / 1000, float(x.max()) / 1000,
              float(y.min()) / 1000, float(y.max()) / 1000]
    render_map(OUTPUT_DIR / "no2_2021_01_chromap_style.png", chromap_field, extent,
               "Netherlands NO₂ — January 2021\nCHROMAP-style direct median ensemble")
    render_map(OUTPUT_DIR / "no2_2021_01_nl_residual.png", nl_field, extent,
               "Netherlands NO₂ — January 2021\nNL CAMS + Extra Trees residual correction")
    render_comparison(OUTPUT_DIR / "no2_2021_01_comparison.png",
                      chromap_field, nl_field, extent)

    difference = chromap_field - nl_field
    summary = {
        "created_utc": created,
        "training_station_rows_chromap_style": int(len(stations)),
        "model_diagnostics": model_diagnostics,
        "chromap_style": statistics(chromap_field),
        "nl_residual": statistics(nl_field),
        "difference_chromap_minus_nl": statistics(difference),
        "spatial_pearson_correlation": float(np.corrcoef(
            chromap_field[mask], nl_field[mask]
        )[0, 1]),
        "files": [],
        "caveat": "Method comparison, not an accuracy ranking; CHROMAP-style is a monthly adaptation and was not separately spatially validated here.",
    }
    for path in sorted(OUTPUT_DIR.glob("*")):
        if path.name != "summary.json":
            summary["files"].append({"path": str(path), "bytes": path.stat().st_size,
                                     "sha256": sha256(path)})
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
