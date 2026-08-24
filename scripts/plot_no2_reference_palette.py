"""Render 2021 NO2 maps with the user-provided blue-to-red reference palette."""

from __future__ import annotations

import json
from calendar import month_name
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize


COLORS = [
    "#4f83b5",  # deep blue
    "#3d9fc0",  # cyan-blue
    "#55babe",  # turquoise
    "#83cbb5",  # sea green
    "#bed8ae",  # pale green
    "#e8dfa5",  # cream-yellow
    "#f3bd7b",  # light orange
    "#e87b49",  # orange-red
    "#c94835",  # hotspot red
]
POSITIONS = [0.00, 0.14, 0.28, 0.43, 0.58, 0.70, 0.82, 0.92, 1.00]
VMIN = 0.0
VMAX = 40.0


def draw(ax, field, extent, cmap, norm, title):
    ax.set_facecolor(COLORS[0])
    image = ax.imshow(
        field, origin="lower", extent=extent, cmap=cmap, norm=norm,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    source = Path("outputs/pilot_2021/chromap_nl_500m_monthly_2021.nc")
    output = Path("outputs/pilot_2021/no2_reference_palette")
    output.mkdir(parents=True, exist_ok=True)
    dataset = xr.open_dataset(source)
    cmap = LinearSegmentedColormap.from_list(
        "reference_blue_green_orange", list(zip(POSITIONS, COLORS)), N=256
    )
    cmap.set_bad(COLORS[0])
    norm = Normalize(VMIN, VMAX, clip=True)
    extent = [
        float(dataset.x.min()) / 1000, float(dataset.x.max()) / 1000,
        float(dataset.y.min()) / 1000, float(dataset.y.max()) / 1000,
    ]

    for month in range(1, 13):
        fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor="white")
        image = draw(
            ax, dataset.no2.isel(time=month - 1).values, extent, cmap, norm,
            f"Netherlands NO₂ — {month_name[month]} 2021",
        )
        colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.025)
        colorbar.set_label("Monthly mean NO₂ (µg m⁻³)")
        colorbar.set_ticks([0, 10, 20, 30, 40])
        fig.tight_layout()
        fig.savefig(output / f"no2_2021_{month:02d}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(13, 13), facecolor="white", constrained_layout=True)
    for month, ax in enumerate(axes.ravel(), start=1):
        image = draw(
            ax, dataset.no2.isel(time=month - 1).values, extent, cmap, norm,
            month_name[month],
        )
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.025, pad=0.025)
    colorbar.set_label("Monthly mean NO₂ (µg m⁻³); common 0–40 scale")
    colorbar.set_ticks([0, 10, 20, 30, 40])
    fig.suptitle("CHROMAP-NL monthly NO₂ — 2021", fontsize=18, fontweight="bold")
    fig.savefig(output / "no2_2021_all_months.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    palette = {
        "name": "reference_blue_green_orange", "colors": COLORS,
        "positions": POSITIONS, "vmin_ug_m3": VMIN, "vmax_ug_m3": VMAX,
        "note": "Visual styling only; the NetCDF concentration values are unchanged.",
    }
    (output / "palette.json").write_text(json.dumps(palette, indent=2), encoding="utf-8")
    print(output / "no2_2021_all_months.png")
    print(f"Rendered 12 monthly PNGs in {output}")


if __name__ == "__main__":
    main()
