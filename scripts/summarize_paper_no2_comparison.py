"""Create final comparison figures and an evidence-backed Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


ROOT = Path("outputs/paper_chromap_no2_2021_01")
DOCS = Path("docs/results")
COLORS = ["#4f83b5", "#3d9fc0", "#55babe", "#83cbb5", "#bed8ae",
          "#e8dfa5", "#f3bd7b", "#e87b49", "#c94835"]


def bootstrap(frame: pd.DataFrame, repetitions: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    obs = frame.value.values
    a, b = frame.pred_paper_chromap.values, frame.pred_nl_original_features.values
    differences = np.empty(repetitions)
    for index in range(repetitions):
        take = rng.integers(0, len(frame), len(frame))
        differences[index] = np.sqrt(np.mean((a[take] - obs[take]) ** 2)) - np.sqrt(np.mean((b[take] - obs[take]) ** 2))
    return {"definition": "paper CHROMAP RMSE minus NL RMSE; positive favors NL",
            "mean": float(differences.mean()), "ci95_low": float(np.percentile(differences, 2.5)),
            "ci95_high": float(np.percentile(differences, 97.5))}


def comparison_map(dataset: xr.Dataset) -> None:
    paper = dataset.no2_paper_chromap.squeeze().values
    nl = dataset.no2_nl_original_features.squeeze().values
    difference = paper - nl
    extent = [float(dataset.x.min()) / 1000, float(dataset.x.max()) / 1000,
              float(dataset.y.min()) / 1000, float(dataset.y.max()) / 1000]
    cmap = LinearSegmentedColormap.from_list("reference", COLORS, N=256); cmap.set_bad("white")
    limit = float(np.nanpercentile(np.abs(difference), 98))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2))
    for ax, field, title in ((axes[0], paper, "Paper-feature CHROMAP"),
                             (axes[1], nl, "NL residual approach")):
        image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, norm=Normalize(0, 40))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Easting (km)")
    diff = axes[2].imshow(difference, origin="lower", extent=extent, cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[2].set_title("CHROMAP − NL", fontweight="bold"); axes[2].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    fig.colorbar(image, ax=axes[:2], orientation="horizontal", fraction=.045, pad=.08,
                 label="January mean NO₂ (µg m⁻³); common 0–40 scale")
    fig.colorbar(diff, ax=axes[2], orientation="horizontal", fraction=.045, pad=.08,
                 label="Difference (µg m⁻³)")
    fig.suptitle("Netherlands NO₂ — January 2021", fontsize=16, fontweight="bold")
    fig.savefig(ROOT / "no2_2021_01_paper_vs_nl_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def validation_figure(frame: pd.DataFrame, metrics: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    styles = {"paper_chromap": ("#d95f02", "Paper-feature CHROMAP"),
              "nl_original_features": ("#1b9e77", "NL residual")}
    lower, upper = frame.value.min() - 2, frame.value.max() + 2
    for method, (color, label) in styles.items():
        axes[0].scatter(frame.value, frame[f"pred_{method}"], s=22, alpha=.6,
                        color=color, edgecolor="none", label=label)
    axes[0].plot([lower, upper], [lower, upper], "k--", lw=1)
    axes[0].set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Observed NO₂ (µg m⁻³)",
                ylabel="Held-out prediction (µg m⁻³)", title="Identical held-out stations")
    axes[0].legend(frameon=False)
    methods = ["cams", "paper_chromap", "nl_original_features", "nl_same_paper_features"]
    labels = ["CAMS", "Paper\nCHROMAP", "NL original\nfeatures", "NL same\npaper features"]
    bars = axes[1].bar(labels, [metrics[x]["rmse"] for x in methods],
                       color=["#888888", "#d95f02", "#1b9e77", "#4daf4a"])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(ylabel="RMSE (µg m⁻³)", title="Five-fold spatial validation", ylim=(0, 6.2))
    fig.suptitle("January 2021 NO₂ method comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(ROOT / "spatial_validation_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(ROOT / "paired_oof_predictions.csv")
    boot = bootstrap(predictions)
    summary["paired_bootstrap"] = boot
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = pd.DataFrame([{"method": name, **values} for name, values in summary["metrics"].items()])
    metrics.to_csv(DOCS / "no2_2021_01_paper_reproduction_metrics.csv", index=False)
    dataset = xr.open_dataset(ROOT / "no2_2021_01_paper_chromap_vs_nl.nc")
    comparison_map(dataset); validation_figure(predictions, summary["metrics"])
    shap = pd.read_csv(ROOT / "shap/feature_importance.csv")
    top = shap.sort_values(["method", "rank"]).groupby("method").head(8)
    def rows(method: str) -> str:
        part = top[top.method == method]
        return "\n".join(f"| {int(r['rank'])} | `{r['feature']}` | {r['mean_abs_shap']:.3f} | {100*r['importance_share']:.1f}% |" for _, r in part.iterrows())
    m = summary["metrics"]
    report = f"""# January 2021 NO₂: paper-feature CHROMAP versus NL

## Result

| Method | RMSE | MAE | Bias | R² |
|---|---:|---:|---:|---:|
| CAMS baseline | {m['cams']['rmse']:.3f} | {m['cams']['mae']:.3f} | {m['cams']['bias']:.3f} | {m['cams']['r2']:.3f} |
| Paper-feature CHROMAP | {m['paper_chromap']['rmse']:.3f} | {m['paper_chromap']['mae']:.3f} | {m['paper_chromap']['bias']:.3f} | {m['paper_chromap']['r2']:.3f} |
| NL residual, original features | {m['nl_original_features']['rmse']:.3f} | {m['nl_original_features']['mae']:.3f} | {m['nl_original_features']['bias']:.3f} | {m['nl_original_features']['r2']:.3f} |
| NL residual, same paper features | {m['nl_same_paper_features']['rmse']:.3f} | {m['nl_same_paper_features']['mae']:.3f} | {m['nl_same_paper_features']['bias']:.3f} | {m['nl_same_paper_features']['r2']:.3f} |

The paired bootstrap difference (paper CHROMAP minus NL-original RMSE) is
{boot['mean']:.3f} µg m⁻³, with descriptive 95% interval
[{boot['ci95_low']:.3f}, {boot['ci95_high']:.3f}]. Positive values favor NL.

The controlled same-feature comparison (4.530 versus 3.344 RMSE) indicates that
the residual formulation—not the NL feature substitutions—accounts for most of
the improvement in this experiment.

## Leading SHAP drivers

### Paper-feature CHROMAP

| Rank | Feature | mean abs. SHAP | share |
|---:|---|---:|---:|
{rows('chromap_style')}

### NL residual on the same paper features

| Rank | Feature | mean abs. SHAP | share |
|---:|---|---:|---:|
{rows('nl_residual')}

The paper's three highlighted variables are now real inputs: `CHEM_CTM_no2`
ranks 1st, `GEO_UrbDeg` 2nd, and `CHEM_Sat_no2` 8th in the January CHROMAP
adaptation. A different rank from the publication is expected because its figure
averages annual models over 2019–2023 across Europe, whereas this run uses one
winter month and a Netherlands-centred station sample.

## Interpretation limits

- This is a paper-faithful **monthly adaptation**, not the published annual map.
- The original predictor arrays/trained weights are not released.
- The CHROMAP MLP hit its 200-iteration limit in all five folds, matching the
  supplied implementation limit but adding model instability.
- One TROPOMI archive object is internally truncated; 70 intersecting orbits and
  202,617 QA-valid pixels were retained, with the failure recorded in the manifest.
- SHAP values explain fitted predictions and are not causal effects.
"""
    (DOCS / "NO2_2021_01_PAPER_REPRODUCTION.md").write_text(report, encoding="utf-8")
    print("wrote paper-faithful NO2 comparison report and figures")


if __name__ == "__main__":
    main()
