"""Create a publication-style QA plate for one monthly gridded product."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PANELS = [
    ("no2", r"NO$_2$ monthly mean", "magma", r"ug m$^{-3}$"),
    ("pm2p5", r"PM$_{2.5}$ monthly mean", "inferno", r"ug m$^{-3}$"),
    ("pm10", r"PM$_{10}$ monthly mean", "inferno", r"ug m$^{-3}$"),
    ("o3", r"O$_3$ monthly mean", "viridis", r"ug m$^{-3}$"),
    ("somo35", "SOMO35 monthly contribution", "plasma", r"ug m$^{-3}$ days"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with xr.open_dataset(args.input) as source:
        ds = source.load()

    month = pd.Timestamp(ds.time.values[0]).strftime("%B %Y")
    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, (name, title, cmap, unit) in zip(axes, PANELS):
        values = ds[name].isel(time=0).to_numpy()
        finite = values[np.isfinite(values)]
        vmin, vmax = np.nanpercentile(finite, [1, 99])
        image = ax.pcolormesh(ds.x, ds.y, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
        colorbar.set_label(unit, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    info = axes[-1]
    info.axis("off")
    info.set_title("Station benchmark", fontsize=12, fontweight="bold")
    if args.metrics:
        metrics = pd.read_csv(args.metrics).set_index("indicator")
        rows = []
        for name, label in [("no2", "NO2"), ("pm2p5", "PM2.5"), ("pm10", "PM10"), ("o3", "O3"), ("somo35", "SOMO35")]:
            row = metrics.loc[name]
            rows.append([label, int(row["n"]), f'{row["rmse"]:.2f}', f'{row["bias_cams_minus_observed"]:+.2f}'])
        table = info.table(
            cellText=rows,
            colLabels=["Indicator", "n", "RMSE", "Bias"],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        info.text(0.5, 0.16, "CAMS bilinear baseline vs ratified RIVM targets", ha="center", fontsize=9)

    fig.suptitle(
        f"Netherlands 500 m air-quality baseline — {month}",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "CAMS European air-quality reanalysis interpolated to the fixed CBS/PDOK land grid. "
        "QA baseline; not yet CHROMAP statistical downscaling.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
