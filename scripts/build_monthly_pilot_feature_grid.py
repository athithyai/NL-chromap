"""Combine aligned January 2021 pilot predictors for CHROMAP-style training.

This is the project predictor stack for O3, PM10, PM2.5 and SO2, not the
paper-defined 22-feature NO2 stack. All inputs must already share the frozen
500 m EPSG:3035 Netherlands grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from chromap_models import PILOT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pollutant", required=True, choices=("o3", "pm10", "pm2p5", "so2"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", default="h5netcdf")
    args = parser.parse_args()
    base = Path("data/processed")
    cams_path = base / "baseline" / (
        "cams_baseline_nl_500m_2021_01_so2.nc" if args.pollutant == "so2"
        else "cams_baseline_nl_500m_2021_01.nc")
    paths = {
        "grid": base / "grid/chromap_nl_500m_grid.nc",
        "static": base / "predictors/static_predictors_nl_500m_2021.nc",
        "roads": base / "predictors/osm_road_density_nl_500m.nc",
        "met": base / "predictors/era5_monthly_predictors_nl_500m_2021.nc",
        "cams": cams_path,
    }
    datasets = {name: xr.open_dataset(path, engine=args.engine) for name, path in paths.items()}
    try:
        grid, static, roads, met, cams = (datasets[name] for name in paths)
        mask = grid.land_mask.load()
        shape = mask.shape
        source = {
            "cams_value": cams[args.pollutant].squeeze(),
            "log1p_population_density": static.log1p_population_density,
            "elevation": static.elevation,
            "landcover_tree_fraction": static.landcover_tree_fraction,
            "landcover_shrub_fraction": static.landcover_shrub_fraction,
            "landcover_grass_fraction": static.landcover_grass_fraction,
            "landcover_cropland_fraction": static.landcover_cropland_fraction,
            "landcover_built_up_fraction": static.landcover_built_up_fraction,
            "landcover_bare_sparse_fraction": static.landcover_bare_sparse_fraction,
            "landcover_permanent_water_fraction": static.landcover_permanent_water_fraction,
            "landcover_wetland_fraction": static.landcover_wetland_fraction,
            "road_density_major": roads.road_density_major,
            "road_density_secondary": roads.road_density_secondary,
            "road_density_local": roads.road_density_local,
            "road_density_motorized_total": roads.road_density_motorized_total,
            "era5_u10": met.u10.isel(time=0), "era5_v10": met.v10.isel(time=0),
            "era5_t2m": met.t2m.isel(time=0), "era5_blh": met.blh.isel(time=0),
            "era5_sp": met.sp.isel(time=0), "era5_tcc": met.tcc.isel(time=0),
            "era5_ssrd": met.ssrd.isel(time=0), "era5_tp": met.tp.isel(time=0),
            "era5_wind_speed": met.wind_speed_10m.isel(time=0),
            "month_sin": xr.DataArray(np.full(shape, np.sin(2 * np.pi / 12), dtype="float32"), dims=("y", "x")),
            "month_cos": xr.DataArray(np.full(shape, np.cos(2 * np.pi / 12), dtype="float32"), dims=("y", "x")),
        }
        result = xr.Dataset({name: source[name].squeeze().load() for name in PILOT_FEATURES},
                            coords={"x": grid.x, "y": grid.y})
        result["land_mask"] = mask
        result["spatial_ref"] = grid.spatial_ref.load()
        for name in PILOT_FEATURES:
            if result[name].shape != shape:
                raise ValueError(f"{name} has {result[name].shape}; expected {shape}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(args.output, engine=args.engine)
        print(args.output)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
