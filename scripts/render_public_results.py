"""Render Dutch-only station measurements and CHROMAP January 2021 maps.

Reads local, untracked inputs. Commits only PNG figures; never publishes
station-level records or downloaded provider rasters.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results"
OBSERVATIONS = ROOT / "output/station_checks_2021_01/all_pollutants_station_checks_2021_01.csv"
BOUNDARY = ROOT / "data/raw/pdok/WijkBuurtkaart_2026_v0.gpkg"
MAP_ROOT = ROOT / "outputs"
CAMS_NO2 = ROOT / "data/processed/baseline/cams_baseline_nl_500m_2021_01.nc"
META = {
    "no2": ("NO₂", 40), "o3": ("O₃", 60), "pm10": ("PM₁₀", 40),
    "pm2p5": ("PM₂.₅", 25), "so2": ("SO₂", 10),
}
COLORS = ["#4f83b5", "#3d9fc0", "#55babe", "#83cbb5", "#bed8ae",
          "#e8dfa5", "#f3bd7b", "#e87b49", "#c94835"]
PALETTE = LinearSegmentedColormap.from_list("blue_green_orange", COLORS, N=256)
PALETTE.set_bad("white")


def load_field(path: Path, variable: str) -> tuple[np.ndarray, list[float]]:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        values = ds[variable].squeeze().values.astype("float32")
        extent = [float(ds.x.min()) / 1000, float(ds.x.max()) / 1000,
                  float(ds.y.min()) / 1000, float(ds.y.max()) / 1000]
    values[values <= -9999] = np.nan
    return values, extent


def render_map(ax, values: np.ndarray, extent: list[float], title: str,
               vmax: float) -> None:
    im = ax.imshow(values, origin="lower", extent=extent, cmap=PALETTE,
                   norm=Normalize(0, vmax), interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("EPSG:3035 easting (km)")
    ax.set_ylabel("EPSG:3035 northing (km)")
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(OBSERVATIONS)
    outline = gpd.read_file(BOUNDARY, layer="gemeenten", columns=["geometry"])
    outline = gpd.GeoSeries([outline.geometry.union_all()], crs=outline.crs).to_crs(4326)
    for code, (label, vmax) in META.items():
        dutch = stations[stations.pollutant.eq(code) & stations.country.eq("NL")]
        fig, ax = plt.subplots(figsize=(5.4, 6.0), facecolor="white")
        outline.plot(ax=ax, facecolor="#f1f4f4", edgecolor="#8a969e", linewidth=.55)
        points = ax.scatter(dutch.longitude, dutch.latitude, c=dutch.observed_ug_m3,
                            cmap=PALETTE, norm=Normalize(0, vmax), s=52,
                            edgecolor="white", linewidth=.5, zorder=3)
        ax.set_xlim(3.25, 7.3)
        ax.set_ylim(50.45, 53.75)
        ax.set_aspect(1 / np.cos(np.deg2rad(52)))
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{label} ground measurements · January 2021\n{len(dutch)} Dutch stations", fontweight="bold")
        fig.colorbar(points, ax=ax, orientation="horizontal", pad=.08,
                     label=f"Monthly mean {label} (µg/m³); 0–{vmax}")
        fig.tight_layout()
        fig.savefig(OUT / f"{code}_2021_01_dutch_measurements.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        path = MAP_ROOT / f"chromap_{code}_2021_01" / f"{code}_2021_01_chromap.nc"
        values, extent = load_field(path, code)
        fig, ax = plt.subplots(figsize=(5.4, 6.0), facecolor="white")
        im = render_map(ax, values, extent, f"{label} CHROMAP adaptation · January 2021", vmax)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=.08,
                     label=f"Monthly mean {label} (µg/m³); 0–{vmax}")
        fig.tight_layout()
        fig.savefig(OUT / f"{code}_2021_01_chromap_map.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    cams, cams_extent = load_field(CAMS_NO2, "no2")
    chromap, chromap_extent = load_field(
        MAP_ROOT / "chromap_no2_2021_01/no2_2021_01_chromap.nc", "no2")
    if not np.allclose(cams_extent, chromap_extent):
        raise ValueError("CAMS and CHROMAP grid extents differ")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), facecolor="white")
    render_map(axes[0], cams, cams_extent, "CAMS reanalysis baseline", 40)
    im = render_map(axes[1], chromap, chromap_extent, "Monthly CHROMAP adaptation", 40)
    fig.suptitle("Netherlands NO₂ · January 2021 · same 0–40 µg/m³ scale", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.21, wspace=.24)
    colorbar_axis = fig.add_axes([.32, .075, .36, .035])
    fig.colorbar(im, cax=colorbar_axis, orientation="horizontal",
                 label="Monthly mean NO₂ (µg/m³)")
    fig.savefig(OUT / "no2_2021_01_chromap_vs_cams.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered 11 public figures in {OUT}")


if __name__ == "__main__":
    main()
