"""Paired spatial validation of January 2021 NO2 modelling approaches.

Both approaches use the same valid station rows, predictor columns, and five
geographically clustered folds. The CHROMAP-style candidate directly predicts
NO2 with a median Ridge/RF/XGBoost/MLP ensemble. The NL candidate predicts the
CAMS residual with Extra Trees and adds it back to CAMS. All preprocessing and
model fitting happen inside each training fold.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compare_no2_2021_01_approaches import FEATURES, chromap_models


OUTPUT_DIR = Path("outputs/comparison_2021_01_no2/spatial_validation")
MODEL_LABELS = {
    "cams_baseline": "CAMS baseline",
    "chromap_style": "CHROMAP-style ensemble",
    "nl_residual": "NL residual correction",
}
COLORS = {
    "cams_baseline": "#8a8a8a",
    "chromap_style": "#d95f02",
    "nl_residual": "#1b9e77",
}


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias_pred_minus_obs": float(np.mean(predicted - observed)),
        "r2": float(r2_score(observed, predicted)),
    }


def paired_bootstrap(observed: np.ndarray, chromap: np.ndarray, nl: np.ndarray,
                     repetitions: int = 10_000) -> dict[str, float]:
    """Station-paired descriptive bootstrap; positive differences favor NL."""
    rng = np.random.default_rng(42)
    n = len(observed)
    rmse_difference = np.empty(repetitions)
    mae_difference = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, n, n)
        obs = observed[sample]
        pred_chromap = chromap[sample]
        pred_nl = nl[sample]
        rmse_difference[index] = (
            np.sqrt(np.mean((pred_chromap - obs) ** 2))
            - np.sqrt(np.mean((pred_nl - obs) ** 2))
        )
        mae_difference[index] = (
            np.mean(np.abs(pred_chromap - obs))
            - np.mean(np.abs(pred_nl - obs))
        )
    return {
        "definition": "CHROMAP-style error minus NL residual error; positive favors NL",
        "rmse_difference": float(np.mean(rmse_difference)),
        "rmse_difference_ci95_low": float(np.percentile(rmse_difference, 2.5)),
        "rmse_difference_ci95_high": float(np.percentile(rmse_difference, 97.5)),
        "mae_difference": float(np.mean(mae_difference)),
        "mae_difference_ci95_low": float(np.percentile(mae_difference, 2.5)),
        "mae_difference_ci95_high": float(np.percentile(mae_difference, 97.5)),
        "caveat": "Rows are resampled as independent stations; the interval is descriptive and does not model spatial dependence.",
    }


def render(predictions: pd.DataFrame, fold_metrics: pd.DataFrame,
           summary: dict[str, object], path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")
    fig.subplots_adjust(hspace=0.31, wspace=0.23, top=0.91)

    # Identical axes make calibration and spread directly comparable.
    lower = float(min(predictions.observed.min(), predictions.pred_chromap_style.min(),
                      predictions.pred_nl_residual.min()))
    upper = float(max(predictions.observed.max(), predictions.pred_chromap_style.max(),
                      predictions.pred_nl_residual.max()))
    padding = (upper - lower) * 0.05
    limits = (lower - padding, upper + padding)
    ax = axes[0, 0]
    for model in ("chromap_style", "nl_residual"):
        ax.scatter(predictions.observed, predictions[f"pred_{model}"], s=24,
                   alpha=0.65, color=COLORS[model], edgecolor="none",
                   label=MODEL_LABELS[model])
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.2, label="1:1")
    ax.set(xlim=limits, ylim=limits, xlabel="Observed NO$_2$ (µg m$^{-3}$)",
           ylabel="Held-out prediction (µg m$^{-3}$)",
           title="Same held-out stations for both methods")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    positions = np.arange(5)
    width = 0.25
    for offset, model in zip((-width, 0, width), MODEL_LABELS):
        values = fold_metrics[fold_metrics.model == model].sort_values("fold").rmse
        ax.bar(positions + offset, values, width=width, color=COLORS[model],
               label=MODEL_LABELS[model])
    ax.set_xticks(positions, [f"Fold {number}" for number in range(1, 6)])
    ax.set_ylabel("RMSE (µg m$^{-3}$)")
    ax.set_title("RMSE varies substantially by geographic fold")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    residual_data = [
        predictions.pred_cams_baseline - predictions.observed,
        predictions.pred_chromap_style - predictions.observed,
        predictions.pred_nl_residual - predictions.observed,
    ]
    boxes = ax.boxplot(residual_data, tick_labels=["CAMS", "CHROMAP-style", "NL residual"],
                       patch_artist=True, showfliers=True)
    for patch, model in zip(boxes["boxes"], MODEL_LABELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Prediction − observation (µg m$^{-3}$)")
    ax.set_title("Held-out residual distributions")

    ax = axes[1, 1]
    squared_advantage = (
        (predictions.pred_chromap_style - predictions.observed) ** 2
        - (predictions.pred_nl_residual - predictions.observed) ** 2
    )
    limit = float(np.percentile(np.abs(squared_advantage), 95))
    scatter = ax.scatter(predictions.x / 1000, predictions.y / 1000,
                         c=np.clip(squared_advantage, -limit, limit), cmap="RdBu_r",
                         vmin=-limit, vmax=limit, s=30, edgecolor="black", linewidth=0.15)
    ax.set_xlabel("EPSG:3035 easting (km)")
    ax.set_ylabel("EPSG:3035 northing (km)")
    ax.set_title("Where each method has lower squared error")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("CHROMAP² error − NL² error\nred: NL better; blue: CHROMAP better")

    overall = summary["overall_metrics"]
    subtitle = (
        f"Overall RMSE — CAMS {overall['cams_baseline']['rmse']:.2f}, "
        f"CHROMAP-style {overall['chromap_style']['rmse']:.2f}, "
        f"NL residual {overall['nl_residual']['rmse']:.2f} µg m$^{{-3}}$"
    )
    fig.suptitle("January 2021 NO$_2$: paired five-fold spatial validation\n" + subtitle,
                 fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, object], path: Path) -> None:
    metrics_by_model = summary["overall_metrics"]
    bootstrap = summary["paired_bootstrap"]
    winner = summary["decision"]["recommended_method"]
    text = f"""# January 2021 NO₂ head-to-head validation

## Overall assessment: Share with noted caveats

Both methods were evaluated on the same {summary['n_station_rows']} valid station
rows and the same five geographically clustered folds. Every station appears once
as held-out data, and preprocessing was fitted within each training fold.

| Method | RMSE | MAE | Bias (prediction − observation) | R² |
|---|---:|---:|---:|---:|
| CAMS baseline | {metrics_by_model['cams_baseline']['rmse']:.3f} | {metrics_by_model['cams_baseline']['mae']:.3f} | {metrics_by_model['cams_baseline']['bias_pred_minus_obs']:.3f} | {metrics_by_model['cams_baseline']['r2']:.3f} |
| CHROMAP-style ensemble | {metrics_by_model['chromap_style']['rmse']:.3f} | {metrics_by_model['chromap_style']['mae']:.3f} | {metrics_by_model['chromap_style']['bias_pred_minus_obs']:.3f} | {metrics_by_model['chromap_style']['r2']:.3f} |
| NL residual correction | {metrics_by_model['nl_residual']['rmse']:.3f} | {metrics_by_model['nl_residual']['mae']:.3f} | {metrics_by_model['nl_residual']['bias_pred_minus_obs']:.3f} | {metrics_by_model['nl_residual']['r2']:.3f} |

## Decision

Recommended for this January-only experiment: **{MODEL_LABELS[winner]}**.

The paired bootstrap RMSE difference (CHROMAP-style minus NL) is
{bootstrap['rmse_difference']:.3f} µg m⁻³, with a descriptive 95% interval of
[{bootstrap['rmse_difference_ci95_low']:.3f}, {bootstrap['rmse_difference_ci95_high']:.3f}].
Positive values favor the NL approach.

## Required caveats

- This evaluates January 2021 only; it does not establish the winner for other months or pollutants.
- Folds are geographic K-means clusters, not fixed-distance exclusion buffers.
- The supplied CHROMAP configuration was designed for annual inputs and is adapted here to monthly targets.
- The CHROMAP MLP reached its iteration limit in {summary['diagnostics']['mlp_nonconverged_folds']} of five folds.
- CAMS assimilates monitoring information, so CAMS-based results are not fully independent of observations.
- The bootstrap resamples stations independently and does not represent residual spatial dependence.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("data/processed/model/station_predictors_2021.csv.gz",
                       parse_dates=["month"])
    data = data[
        (data.indicator == "no2") & (data.month.dt.month == 1)
        & data.value.notna() & data.is_valid
    ].copy()
    data = data[np.isfinite(data[FEATURES]).all(axis=1)].reset_index(drop=True)
    if data.duplicated(["source", "station_id"]).any():
        raise ValueError("January input must contain one row per source/station")

    data["fold"] = KMeans(n_clusters=5, random_state=42, n_init=50).fit_predict(data[["x", "y"]])
    predictions = np.full((len(data), 3), np.nan)
    mlp_nonconverged = 0

    for fold in sorted(data.fold.unique()):
        train = data.fold != fold
        test = ~train
        x_train = data.loc[train, FEATURES]
        x_test = data.loc[test, FEATURES]
        y_train = data.loc[train, "value"].to_numpy()

        predictions[test, 0] = np.clip(data.loc[test, "cams_value"].to_numpy(), 0, None)

        preprocessing = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])
        x_train_scaled = preprocessing.fit_transform(x_train)
        x_test_scaled = preprocessing.transform(x_test)
        direct_members = []
        for name, estimator in chromap_models().items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train_scaled, y_train)
            if name == "mlp" and any(issubclass(item.category, ConvergenceWarning) for item in caught):
                mlp_nonconverged += 1
            member = np.maximum(estimator.predict(x_test_scaled), float(y_train.min()))
            direct_members.append(member)
        predictions[test, 1] = np.median(np.vstack(direct_members), axis=0)

        residual_model = ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.7,
            bootstrap=False, n_jobs=-1, random_state=42 + int(fold),
        )
        residual_model.fit(x_train, y_train - data.loc[train, "cams_value"].to_numpy())
        correction = residual_model.predict(x_test)
        predictions[test, 2] = np.clip(data.loc[test, "cams_value"].to_numpy() + correction, 0, None)
        print(f"completed fold {int(fold) + 1}: train={int(train.sum())}, test={int(test.sum())}")

    if not np.isfinite(predictions).all():
        raise RuntimeError("Every station must receive all three out-of-fold predictions")
    for column, values in zip(
        ("pred_cams_baseline", "pred_chromap_style", "pred_nl_residual"), predictions.T
    ):
        data[column] = values

    overall = {
        model: metrics(data.value.to_numpy(), data[f"pred_{model}"].to_numpy())
        for model in MODEL_LABELS
    }
    fold_rows = []
    for fold in sorted(data.fold.unique()):
        group = data[data.fold == fold]
        for model in MODEL_LABELS:
            fold_rows.append({"fold": int(fold) + 1, "model": model,
                              **metrics(group.value.to_numpy(), group[f"pred_{model}"].to_numpy())})
    fold_frame = pd.DataFrame(fold_rows)
    bootstrap = paired_bootstrap(
        data.value.to_numpy(), data.pred_chromap_style.to_numpy(), data.pred_nl_residual.to_numpy()
    )
    compared = {name: overall[name]["rmse"] for name in ("chromap_style", "nl_residual")}
    winner = min(compared, key=compared.get)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "January-only paired geographic out-of-fold accuracy comparison",
        "n_station_rows": int(len(data)),
        "n_unique_stations": int(data[["source", "station_id"]].drop_duplicates().shape[0]),
        "fold_sizes": {str(int(key) + 1): int(value) for key, value in data.fold.value_counts().sort_index().items()},
        "overall_metrics": overall,
        "paired_bootstrap": bootstrap,
        "diagnostics": {"mlp_nonconverged_folds": mlp_nonconverged},
        "decision": {
            "recommended_method": winner,
            "basis": "lowest RMSE on identical held-out station rows",
            "scope": "January 2021 NO2 only",
        },
    }

    prediction_columns = [
        "source", "country", "station_id", "station_type", "x", "y", "fold",
        "value", "cams_value", "pred_cams_baseline", "pred_chromap_style", "pred_nl_residual",
    ]
    data[prediction_columns].rename(columns={"value": "observed"}).to_csv(
        OUTPUT_DIR / "paired_oof_predictions.csv", index=False
    )
    fold_frame.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render(data.rename(columns={"value": "observed"}), fold_frame, summary,
           OUTPUT_DIR / "spatial_validation_comparison.png")
    write_report(summary, OUTPUT_DIR / "VALIDATION_REPORT.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
