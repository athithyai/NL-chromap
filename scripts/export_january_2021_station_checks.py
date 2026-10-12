"""Export January 2021 training-station locations and station/map comparisons.

The final maps were fitted using these stations. Their sampled values are
descriptive; the saved spatial-fold predictions are the validation results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "station_checks_2021_01"
NO2_OOF = ROOT / "outputs/paper_chromap_no2_2021_01/paired_oof_predictions.csv"
NO2_META = ROOT / "data/processed/model/paper_no2_2021_01/paper_no2_station_features_2021_01.csv.gz"
OTHER_OOF = ROOT / "outputs/comparison_2021_01_other_pollutants/paired_oof_predictions.csv"
NO2_MAP = ROOT / "outputs/paper_chromap_no2_2021_01/no2_2021_01_paper_chromap_vs_nl.nc"
OTHER_MAP_ROOT = ROOT / "outputs/comparison_2021_01_other_pollutants"
GRID_MANIFEST = ROOT / "data/processed/grid/chromap_nl_500m_grid.manifest.json"
POLLUTANTS = ("no2", "o3", "pm10", "pm2p5", "so2")
STATION_COLUMNS = (
    "pollutant", "month", "source", "country", "station_id", "sampling_id",
    "station_name", "station_type", "area_type", "longitude", "latitude",
    "x_epsg3035_m", "y_epsg3035_m", "observed_ug_m3", "valid_days",
    "expected_days", "coverage", "spatial_fold", "cams_heldout_ug_m3",
    "chromap_heldout_ug_m3", "nl_heldout_ug_m3", "chromap_map_ug_m3",
    "nl_map_ug_m3", "chromap_heldout_error_ug_m3", "nl_heldout_error_ug_m3",
    "chromap_map_error_ug_m3", "nl_map_error_ug_m3",
)


def load_station_predictions() -> pd.DataFrame:
    no2 = pd.read_csv(NO2_OOF)
    meta = pd.read_csv(NO2_META)
    key = ["source", "country", "station_id"]
    if no2.duplicated(key).any() or meta.duplicated(key).any():
        raise ValueError("NO2 station key is not unique")
    extra = ["sampling_id", "station_name", "station_type", "area_type",
             "longitude", "latitude", "valid_days", "expected_days", "coverage",
             "month", "is_valid"]
    no2 = no2.merge(meta[key + extra], on=key, validate="one_to_one", indicator=True)
    if len(no2) != 258 or not no2._merge.eq("both").all():
        raise ValueError("NO2 station metadata did not match all validation rows")
    no2 = no2.drop(columns="_merge").rename(columns={
        "pred_cams": "cams_heldout_ug_m3",
        "pred_paper_chromap": "chromap_heldout_ug_m3",
        "pred_nl_original_features": "nl_heldout_ug_m3",
    })
    no2["pollutant"] = "no2"

    other = pd.read_csv(OTHER_OOF)
    other = other[other.indicator.isin(POLLUTANTS[1:])].copy()
    if other.duplicated(["indicator", *key]).any():
        raise ValueError("Other-pollutant station key is not unique")
    other = other.rename(columns={
        "indicator": "pollutant", "pred_cams_baseline": "cams_heldout_ug_m3",
        "pred_chromap_style": "chromap_heldout_ug_m3",
        "pred_nl_selected": "nl_heldout_ug_m3",
    })
    frame = pd.concat([no2, other], ignore_index=True)
    frame = frame.rename(columns={
        "x": "x_epsg3035_m", "y": "y_epsg3035_m",
        "value": "observed_ug_m3", "fold": "spatial_fold",
    })
    frame["month"] = pd.to_datetime(frame.month).dt.strftime("%Y-%m")
    if not frame.month.eq("2021-01").all() or not frame.is_valid.eq(True).all():
        raise ValueError("Unexpected month or invalid station target")
    if frame.duplicated(["pollutant", *key]).any():
        raise ValueError("Duplicate station/pollutant row")
    required = ["longitude", "latitude", "x_epsg3035_m", "y_epsg3035_m",
                "observed_ug_m3", "chromap_heldout_ug_m3", "nl_heldout_ug_m3"]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing coordinates, observations, or held-out predictions")
    return frame


def read_map(path: Path, variable: str, scratch: Path,
             gdal_translate: Path | None) -> np.ndarray:
    """Read a map as north-up rows, with GDAL as an optional NetCDF reader."""
    if gdal_translate is None:
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            south_up = dataset[variable].squeeze(drop=True).transpose("y", "x").values
        return np.flipud(np.asarray(south_up, dtype="float32"))

    output = scratch / f"{path.stem}_{variable}.bin"
    source = f"NETCDF:{path}:{variable}"
    subprocess.run([str(gdal_translate), "-q", "-of", "ENVI", source, str(output)],
                   check=True, capture_output=True, text=True)
    height, width = json.loads(GRID_MANIFEST.read_text(encoding="utf-8"))["shape_yx"]
    raster = np.fromfile(output, dtype="<f4")
    if raster.size != height * width:
        raise ValueError(f"Unexpected raster shape for {source}: {raster.size}")
    return raster.reshape(height, width)


def sample_maps(frame: pd.DataFrame, pollutant: str, scratch: Path,
                gdal_translate: Path | None) -> pd.DataFrame:
    group = frame.loc[frame.pollutant.eq(pollutant)].copy()
    if pollutant == "no2":
        fields = {
            "chromap": read_map(NO2_MAP, "no2_paper_chromap", scratch, gdal_translate),
            "nl": read_map(NO2_MAP, "no2_nl_original_features", scratch, gdal_translate),
        }
    else:
        folder = OTHER_MAP_ROOT / pollutant
        fields = {
            "chromap": read_map(folder / f"{pollutant}_2021_01_chromap_style.nc",
                                pollutant, scratch, gdal_translate),
            "nl": read_map(folder / f"{pollutant}_2021_01_nl_selected.nc",
                           pollutant, scratch, gdal_translate),
        }

    grid = json.loads(GRID_MANIFEST.read_text(encoding="utf-8"))
    xmin, ymin, xmax, ymax = grid["aligned_bounds"]
    step = grid["cell_size_m"]
    x = group.x_epsg3035_m.to_numpy()
    y = group.y_epsg3035_m.to_numpy()
    col = np.floor((x - xmin) / step).astype(int)
    row = np.floor((ymax - y) / step).astype(int)  # GDAL presents north-up rows.
    inside = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
    for name, field in fields.items():
        if field.shape != tuple(grid["shape_yx"]):
            raise ValueError(f"Map grid mismatch for {pollutant}")
        sampled = np.full(len(group), np.nan)
        sampled[inside] = field[row[inside], col[inside]]
        sampled[np.isclose(sampled, -9999.0)] = np.nan
        group[f"{name}_map_ug_m3"] = sampled
        group[f"{name}_map_error_ug_m3"] = group[f"{name}_map_ug_m3"] - group.observed_ug_m3
        group[f"{name}_heldout_error_ug_m3"] = group[f"{name}_heldout_ug_m3"] - group.observed_ug_m3
    return group


def score(group: pd.DataFrame, column: str) -> dict[str, float | int]:
    errors = (group[column] - group.observed_ug_m3).dropna().to_numpy()
    return {
        "n": len(errors),
        "rmse_ug_m3": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else np.nan,
        "mae_ug_m3": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "bias_ug_m3": float(np.mean(errors)) if len(errors) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gdal-translate", type=Path,
        help="Optional path to gdal_translate for systems where Python NetCDF reading fails",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_station_predictions()
    if args.gdal_translate is not None and not args.gdal_translate.is_file():
        raise FileNotFoundError(f"GDAL executable not found: {args.gdal_translate}")
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        scratch = Path(temporary).resolve()
        if not scratch.is_relative_to(OUT.resolve()):
            raise ValueError("Temporary directory escaped output directory")
        result = pd.concat([sample_maps(base, name, scratch, args.gdal_translate)
                            for name in POLLUTANTS],
                           ignore_index=True)
    result = result.loc[:, STATION_COLUMNS].sort_values(
        ["pollutant", "country", "source", "station_id"]
    ).reset_index(drop=True)
    assert len(result) == 824

    result.to_csv(OUT / "all_pollutants_station_checks_2021_01.csv", index=False,
                  float_format="%.8f")
    rows = []
    for pollutant in POLLUTANTS:
        group = result[result.pollutant.eq(pollutant)]
        group.to_csv(OUT / f"{pollutant}_station_locations_and_checks_2021_01.csv",
                     index=False, float_format="%.8f")
        for subset_name, subset in (("all_training_stations", group),
                                    ("netherlands_stations", group[group.country.eq("NL")])):
            for model, column in (
                ("CAMS held-out", "cams_heldout_ug_m3"),
                ("CHROMAP held-out", "chromap_heldout_ug_m3"),
                ("NL held-out", "nl_heldout_ug_m3"),
                ("CHROMAP final map at station", "chromap_map_ug_m3"),
                ("NL final map at station", "nl_map_ug_m3"),
            ):
                rows.append({"pollutant": pollutant, "scope": subset_name,
                             "comparison": model, **score(subset, column)})
    pd.DataFrame(rows).to_csv(OUT / "accuracy_summary_2021_01.csv", index=False,
                              float_format="%.4f")

    locations = gpd.GeoDataFrame(
        result.drop(columns=["chromap_heldout_error_ug_m3", "nl_heldout_error_ug_m3",
                             "chromap_map_error_ug_m3", "nl_map_error_ug_m3"]),
        geometry=gpd.points_from_xy(result.longitude, result.latitude), crs="EPSG:4326",
    )
    locations.to_file(OUT / "ground_stations_2021_01.gpkg", layer="stations", driver="GPKG")
    print(f"Exported {len(result)} pollutant-station records to {OUT}")
    print(result.groupby(["pollutant", "country"]).size().to_string())
    print(pd.DataFrame(rows).query("scope == 'all_training_stations' and comparison == 'CHROMAP held-out'").to_string(index=False))


if __name__ == "__main__":
    main()
