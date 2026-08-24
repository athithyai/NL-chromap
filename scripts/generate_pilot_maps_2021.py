"""Generate selected-model 2021 monthly CHROMAP-NL maps and uncertainty proxies."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


INDICATORS = ("no2", "o3", "pm10", "pm2p5", "somo35")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    model_dir = Path("data/processed/model/validated_2021")
    output_dir = Path("outputs/pilot_2021")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((model_dir / "selected_models.json").read_text())
    metrics = pd.read_csv(model_dir / "spatial_cv_metrics_2021.csv")
    cv = pd.read_csv(model_dir / "spatial_cv_predictions_2021.csv.gz")
    grid = xr.open_dataset("data/processed/grid/chromap_nl_500m_grid.nc")
    static = xr.open_dataset("data/processed/predictors/static_predictors_nl_500m_2021.nc")
    roads = xr.open_dataset("data/processed/predictors/osm_road_density_nl_500m.nc")
    met = xr.open_dataset("data/processed/predictors/era5_monthly_predictors_nl_500m_2021.nc")
    mask = grid.land_mask.values == 1
    iy, ix = np.where(mask)
    grid_points = np.column_stack((grid.x.values[ix], grid.y.values[iy]))

    model_objects = {}
    for indicator, name in selected.items():
        if name != "cams_baseline":
            model_objects[indicator] = joblib.load(model_dir / f"{indicator}_{name}.joblib")

    distance_fields: dict[str, np.ndarray] = {}
    rmse: dict[str, float] = {}
    for indicator, name in selected.items():
        stations = cv[cv.indicator == indicator].drop_duplicates(["source", "station_id"])
        distance, _ = cKDTree(stations[["x", "y"]].values).query(grid_points)
        field = np.full(mask.shape, np.nan, dtype="float32")
        field[mask] = distance.astype("float32") / 1000
        distance_fields[indicator] = field
        rmse[indicator] = float(metrics[(metrics.indicator == indicator) & (metrics.model == name)].rmse.iloc[0])

    outputs = []
    for month in range(1, 13):
        cams = xr.open_dataset(f"data/processed/baseline/cams_baseline_nl_500m_2021_{month:02d}.nc")
        feature_base = {
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
            "era5_u10": met.u10.isel(time=month - 1).values[mask],
            "era5_v10": met.v10.isel(time=month - 1).values[mask],
            "era5_t2m": met.t2m.isel(time=month - 1).values[mask],
            "era5_blh": met.blh.isel(time=month - 1).values[mask],
            "era5_sp": met.sp.isel(time=month - 1).values[mask],
            "era5_tcc": met.tcc.isel(time=month - 1).values[mask],
            "era5_ssrd": met.ssrd.isel(time=month - 1).values[mask],
            "era5_tp": met.tp.isel(time=month - 1).values[mask],
            "era5_wind_speed": met.wind_speed_10m.isel(time=month - 1).values[mask],
            "month_sin": np.full(mask.sum(), np.sin(2 * np.pi * month / 12)),
            "month_cos": np.full(mask.sum(), np.cos(2 * np.pi * month / 12)),
        }
        variables = {}
        for indicator in INDICATORS:
            baseline = cams[indicator].squeeze().values[mask]
            prediction = baseline.copy()
            if selected[indicator] != "cams_baseline":
                fitted = model_objects[indicator]
                features = fitted["features"]
                table = pd.DataFrame({**feature_base, "cams_value": baseline})
                residual = fitted["model"].predict(table[features])
                prediction = np.clip(baseline + residual, 0, None)
            field = np.full(mask.shape, np.nan, dtype="float32")
            field[mask] = prediction.astype("float32")
            units = "ug m-3 day" if indicator == "somo35" else "ug m-3"
            variables[indicator] = (("time", "y", "x"), field[None], {
                "units": units, "selected_model": selected[indicator],
                "long_name": f"Monthly {indicator} selected-model estimate",
            })
            # Empirical CV RMSE inflated smoothly with distance from a training station.
            uncertainty = rmse[indicator] * (1 + np.minimum(distance_fields[indicator], 100) / 200)
            variables[f"{indicator}_uncertainty"] = (("time", "y", "x"), uncertainty[None].astype("float32"), {
                "units": units, "long_name": "Distance-inflated spatial-CV RMSE uncertainty proxy",
            })
        dataset = xr.Dataset(
            data_vars=variables,
            coords={"time": [np.datetime64(f"2021-{month:02d}-01")], "y": grid.y.values, "x": grid.x.values},
            attrs={
                "title": "CHROMAP-NL 2021 spatially validated pilot",
                "target_crs": "EPSG:3035", "resolution_m": 500,
                "selected_models": json.dumps(selected),
                "uncertainty_note": "Proxy based on selected-model spatial-CV RMSE and nearest training-station distance; not a probabilistic confidence interval.",
                "status": "validated_2021_pilot",
            },
        )
        path = output_dir / f"chromap_nl_500m_2021_{month:02d}.nc"
        encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0} for name in dataset.data_vars}
        dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
        outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        cams.close()
        print(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "year": 2021,
        "selected_models": selected, "spatial_cv_rmse": rmse, "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
