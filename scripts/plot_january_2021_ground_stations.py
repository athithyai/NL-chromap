"""Plot the January 2021 monitoring sites used for each pollutant."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/station_checks_2021_01/all_pollutants_station_checks_2021_01.csv"
BOUNDARY = ROOT / "data/raw/pdok/WijkBuurtkaart_2026_v0.gpkg"
OUT = ROOT / "docs/results"
POLLUTANTS = (("no2", "NO₂"), ("o3", "O₃"), ("pm10", "PM₁₀"),
              ("pm2p5", "PM₂.₅"), ("so2", "SO₂"))
NL_COLOR = "#0072B2"
NEIGHBOR_COLOR = "#D55E00"


def make_figure(data: pd.DataFrame, outline: gpd.GeoSeries, *, nl_only: bool) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), facecolor="white")
    panels = axes.ravel()
    extent = (3.25, 7.3, 50.45, 53.75) if nl_only else (1.25, 9.5, 49.25, 54.6)
    for ax, (code, label) in zip(panels, POLLUTANTS):
        group = data[data.pollutant.eq(code)]
        local = group[group.country.eq("NL")]
        nearby = group[group.country.ne("NL")]
        outline.plot(ax=ax, facecolor="#EDF2F3", edgecolor="#61717B",
                     linewidth=0.9, zorder=1)
        if not nl_only:
            ax.scatter(nearby.longitude, nearby.latitude, s=23, c=NEIGHBOR_COLOR,
                       alpha=0.78, edgecolor="white", linewidth=0.3, zorder=2)
        ax.scatter(local.longitude, local.latitude, s=31, c=NL_COLOR,
                   alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(52)))
        ax.set_title(f"{label}  |  {len(local)} NL" +
                     ("" if nl_only else f" + {len(nearby)} neighbouring"),
                     loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.grid(color="#D9E0E3", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors="#52616A")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#AAB7BC")
    panels[-1].axis("off")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8,
                      markerfacecolor=NL_COLOR, markeredgecolor="white",
                      label="Netherlands monitoring station")]
    if not nl_only:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=8,
                              markerfacecolor=NEIGHBOR_COLOR, markeredgecolor="white",
                              label="Neighbouring-country training station"))
    panels[-1].legend(handles=handles, loc="center", frameon=False, fontsize=12)
    scope = "Netherlands stations" if nl_only else "Netherlands and neighbouring-country stations"
    fig.suptitle(f"Ground stations used in the January 2021 models\n{scope}",
                 fontsize=18, fontweight="bold", x=0.08, ha="left", y=0.99)
    fig.supxlabel("Longitude (°E)", fontsize=11, y=0.04)
    fig.supylabel("Latitude (°N)", fontsize=11, x=0.035)
    fig.text(0.08, 0.015,
             "Each point is a station with a valid January monthly mean. Outline: PDOK Netherlands municipalities (2026).",
             fontsize=9.5, color="#56656E")
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.91), h_pad=2.5, w_pad=1.4)
    path = OUT / ("ground_stations_netherlands_2021_01.png" if nl_only
                  else "ground_stations_all_training_2021_01.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    data = pd.read_csv(DATA)
    expected = {"no2": 258, "o3": 159, "pm10": 207, "pm2p5": 152, "so2": 48}
    if data.groupby("pollutant").size().to_dict() != expected:
        raise ValueError("Station counts do not match the January validation population")
    municipalities = gpd.read_file(BOUNDARY, layer="gemeenten", columns=["geometry"])
    outline = gpd.GeoSeries([municipalities.geometry.union_all()], crs=municipalities.crs).to_crs(4326)
    for local_only in (False, True):
        print(make_figure(data, outline, nl_only=local_only))


if __name__ == "__main__":
    main()
