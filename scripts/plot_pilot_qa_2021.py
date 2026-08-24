"""Create spatial and validation QA figures for the 2021 pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


INDICATORS = ("no2", "o3", "pm10", "pm2p5", "somo35")
LABELS = {"no2": "NO₂", "o3": "O₃", "pm10": "PM₁₀", "pm2p5": "PM₂.₅", "somo35": "SOMO35"}


def main() -> None:
    output = Path("outputs/pilot_2021/qa")
    output.mkdir(parents=True, exist_ok=True)
    months = (1, 7, 12)
    datasets = {month: xr.open_dataset(f"outputs/pilot_2021/chromap_nl_500m_2021_{month:02d}.nc") for month in months}
    x = datasets[1].x.values / 1000
    y = datasets[1].y.values / 1000
    extent = [x.min(), x.max(), y.min(), y.max()]
    fig, axes = plt.subplots(5, 3, figsize=(11, 16), constrained_layout=True)
    for row, indicator in enumerate(INDICATORS):
        all_values = np.concatenate([datasets[month][indicator].values.ravel() for month in months])
        vmin, vmax = np.nanpercentile(all_values, [2, 98])
        for column, month in enumerate(months):
            ax = axes[row, column]
            image = ax.imshow(
                datasets[month][indicator].squeeze(), origin="lower", extent=extent,
                vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest",
            )
            if row == 0:
                ax.set_title({1: "January", 7: "July", 12: "December"}[month], fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{LABELS[indicator]}\nNorthing (km)")
            else:
                ax.set_yticklabels([])
            if row == len(INDICATORS) - 1:
                ax.set_xlabel("Easting (km)")
            else:
                ax.set_xticklabels([])
            ax.grid(False)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
            colorbar.ax.tick_params(labelsize=7)
    fig.suptitle("CHROMAP-NL 2021 pilot: seasonal spatial QA", fontsize=16, fontweight="bold")
    fig.savefig(output / "seasonal_spatial_qa.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.read_csv("data/processed/model/validated_2021/spatial_cv_metrics_2021.csv")
    selected = pd.read_json("data/processed/model/validated_2021/selected_models.json", typ="series")
    rows = []
    for indicator in INDICATORS:
        baseline = metrics[(metrics.indicator == indicator) & (metrics.model == "cams_baseline")].iloc[0]
        chosen = metrics[(metrics.indicator == indicator) & (metrics.model == selected[indicator])].iloc[0]
        rows.append({"indicator": LABELS[indicator], "CAMS": baseline.rmse, "Selected": chosen.rmse})
    comparison = pd.DataFrame(rows).set_index("indicator")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    comparison.plot.bar(ax=ax, color=["#9aa0a6", "#1769aa"], width=0.72)
    ax.set_title("Spatial validation improves NO₂ and PM₁₀; CAMS retained elsewhere", fontweight="bold")
    ax.set_ylabel("Spatial-CV RMSE")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "spatial_cv_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output / "seasonal_spatial_qa.png")
    print(output / "spatial_cv_model_comparison.png")


if __name__ == "__main__":
    main()
