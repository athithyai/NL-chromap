"""Explain two January 2021 model families on the current NL predictor stack.

This implementation estimates interventional Shapley values directly so it does
not depend on the optional compiled SHAP/numba stack. Random permutations are
paired with their reverse order to reduce Monte Carlo ordering variance.

This is not an original-CHROMAP predictor reproduction: the current stack omits
TROPOMI total-column NO2 (CHEM_Sat_no2) and GHS-SMOD urbanisation class
(GEO_UrbDeg). Its CAMS monthly mean is the CHEM_CTM_no2 analogue.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compare_no2_2021_01_approaches import FEATURES, chromap_models


OUTPUT_DIR = Path("outputs/feature_importance_2021_01_no2")
INPUT_PATH = Path("data/processed/model/station_predictors_2021.csv.gz")
BACKGROUND_SIZE = 24
PERMUTATIONS = 16
SEED = 42
METHODS = {
    "chromap_style": "CHROMAP-style direct ensemble",
    "nl_residual": "NL CAMS + residual",
}
COLORS = {"chromap_style": "#d95f02", "nl_residual": "#1b9e77"}
DISPLAY_NAMES = {
    "cams_value": "CAMS NO$_2$",
    "log1p_population_density": "Population density (log1p)",
    "elevation": "Elevation",
    "landcover_tree_fraction": "Tree cover",
    "landcover_shrub_fraction": "Shrub cover",
    "landcover_grass_fraction": "Grass cover",
    "landcover_cropland_fraction": "Cropland",
    "landcover_built_up_fraction": "Built-up cover",
    "landcover_bare_sparse_fraction": "Bare/sparse cover",
    "landcover_permanent_water_fraction": "Permanent water",
    "landcover_wetland_fraction": "Wetland",
    "road_density_major": "Major-road density",
    "road_density_secondary": "Secondary-road density",
    "road_density_local": "Local-road density",
    "road_density_motorized_total": "Total motor-road density",
    "era5_u10": "Eastward wind (ERA5)",
    "era5_v10": "Northward wind (ERA5)",
    "era5_t2m": "2 m temperature (ERA5)",
    "era5_blh": "Boundary-layer height (ERA5)",
    "era5_sp": "Surface pressure (ERA5)",
    "era5_tcc": "Cloud cover (ERA5)",
    "era5_ssrd": "Solar radiation (ERA5)",
    "era5_tp": "Precipitation (ERA5)",
    "era5_wind_speed": "Wind speed (ERA5)",
    "month_sin": "Month sine",
    "month_cos": "Month cosine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_PATH, parse_dates=["month"])
    data = data[
        (data.indicator == "no2")
        & (data.month.dt.year == 2021)
        & (data.month.dt.month == 1)
        & data.value.notna()
        & data.is_valid
    ].copy()
    data = data[np.isfinite(data[FEATURES]).all(axis=1)].reset_index(drop=True)
    if len(data) != 258:
        raise ValueError(f"expected 258 complete January NO2 station rows, found {len(data)}")
    if data.duplicated(["source", "station_id"]).any():
        raise ValueError("January input must contain one row per source/station")
    return data


def fit_prediction_functions(
    data: pd.DataFrame,
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, object]]:
    x = data[FEATURES]
    y = data.value.to_numpy()
    preprocessing = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])
    transformed = preprocessing.fit_transform(x)
    direct_members = {}
    mlp_converged = True
    for name, estimator in chromap_models("no2").items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(transformed, y)
        if name == "mlp" and any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ):
            mlp_converged = False
        direct_members[name] = estimator
    lower_bound = float(y.min())

    def predict_chromap(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=FEATURES)
        scaled = preprocessing.transform(frame)
        member_predictions = [
            np.maximum(estimator.predict(scaled), lower_bound)
            for estimator in direct_members.values()
        ]
        return np.median(np.vstack(member_predictions), axis=0)

    residual = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.7,
        bootstrap=False,
        n_jobs=-1,
        random_state=SEED,
    )
    residual.fit(x, y - data.cams_value.to_numpy())

    def predict_nl(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=FEATURES)
        prediction = frame.cams_value.to_numpy() + residual.predict(frame)
        return np.clip(prediction, 0, None)

    diagnostics = {
        "chromap_member_models": list(direct_members),
        "chromap_lower_bound": lower_bound,
        "chromap_mlp_converged": mlp_converged,
        "nl_residual_model": "ExtraTreesRegressor",
        "nl_residual_trees": 400,
    }
    return {"chromap_style": predict_chromap, "nl_residual": predict_nl}, diagnostics


def antithetic_permutations(n_features: int, count: int, seed: int) -> np.ndarray:
    if count % 2:
        raise ValueError("permutation count must be even for antithetic pairing")
    rng = np.random.default_rng(seed)
    first = np.vstack([rng.permutation(n_features) for _ in range(count // 2)])
    return np.vstack([first, first[:, ::-1]])


def permutation_shap(
    predict: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    background: np.ndarray,
    permutations: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate interventional SHAP values for every row and verify efficiency."""
    n_rows, n_features = values.shape
    n_permutations = len(permutations)
    n_background = len(background)
    base_value = float(np.mean(predict(background)))
    shap_values = np.zeros((n_rows, n_features), dtype=np.float64)
    predictions = np.zeros(n_rows, dtype=np.float64)
    maximum_additivity_error = 0.0

    for row_index, row in enumerate(values):
        paths = np.empty(
            (n_permutations, n_features + 1, n_background, n_features),
            dtype=np.float64,
        )
        for order_index, order in enumerate(permutations):
            current = background.copy()
            paths[order_index, 0] = current
            for step, feature_index in enumerate(order, start=1):
                current[:, feature_index] = row[feature_index]
                paths[order_index, step] = current
        path_predictions = predict(paths.reshape(-1, n_features)).reshape(
            n_permutations, n_features + 1, n_background
        ).mean(axis=2)
        deltas = np.diff(path_predictions, axis=1)
        contributions = np.zeros((n_permutations, n_features), dtype=np.float64)
        for order_index, order in enumerate(permutations):
            contributions[order_index, order] = deltas[order_index]
        shap_values[row_index] = contributions.mean(axis=0)
        predictions[row_index] = float(predict(row.reshape(1, -1))[0])
        reconstructed = base_value + shap_values[row_index].sum()
        maximum_additivity_error = max(
            maximum_additivity_error, abs(predictions[row_index] - reconstructed)
        )
        if (row_index + 1) % 10 == 0 or row_index + 1 == n_rows:
            print(f"{method}: explained {row_index + 1}/{n_rows} stations", flush=True)
    return shap_values, predictions, base_value, maximum_additivity_error


def importance_table(method: str, shap_values: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame({
        "method": method,
        "method_label": METHODS[method],
        "feature": FEATURES,
        "feature_label": [DISPLAY_NAMES[name] for name in FEATURES],
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "median_abs_shap": np.median(np.abs(shap_values), axis=0),
        "mean_signed_shap": np.mean(shap_values, axis=0),
        "std_shap": np.std(shap_values, axis=0),
    })
    result["importance_share"] = result.mean_abs_shap / result.mean_abs_shap.sum()
    result["rank"] = result.mean_abs_shap.rank(method="first", ascending=False).astype(int)
    return result.sort_values("rank")


def write_long_values(
    data: pd.DataFrame,
    feature_values: np.ndarray,
    explanations: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    base_values: dict[str, float],
) -> None:
    identifiers = data[["source", "country", "station_id", "value"]].reset_index(drop=True)
    frames = []
    for method, shap_values in explanations.items():
        repeated = identifiers.loc[identifiers.index.repeat(len(FEATURES))].reset_index(drop=True)
        repeated.insert(0, "method", method)
        repeated["prediction"] = np.repeat(predictions[method], len(FEATURES))
        repeated["base_value"] = base_values[method]
        repeated["feature"] = np.tile(FEATURES, len(data))
        repeated["feature_value"] = feature_values.reshape(-1)
        repeated["shap_value"] = shap_values.reshape(-1)
        frames.append(repeated)
    pd.concat(frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "station_feature_shap_values.csv.gz", index=False
    )


def render_importance(importance: pd.DataFrame) -> None:
    pivot = importance.pivot(index="feature_label", columns="method", values="mean_abs_shap")
    order = pivot.max(axis=1).sort_values(ascending=True).tail(15).index
    pivot = pivot.loc[order]
    y = np.arange(len(pivot))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 8), facecolor="white")
    ax.barh(y - height / 2, pivot.chromap_style, height, color=COLORS["chromap_style"],
            label=METHODS["chromap_style"])
    ax.barh(y + height / 2, pivot.nl_residual, height, color=COLORS["nl_residual"],
            label=METHODS["nl_residual"])
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("Mean absolute permutation-SHAP value (µg m$^{-3}$)")
    ax.set_title("What drives January 2021 NO$_2$ predictions?\nSame stations and features",
                 fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "no2_feature_importance_comparison.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def render_beeswarm(values: np.ndarray, shap_values: np.ndarray, method: str) -> None:
    ranks = np.argsort(np.mean(np.abs(shap_values), axis=0))[-15:]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(10.5, 8), facecolor="white")
    norm = Normalize(0, 1)
    scatter = None
    for position, feature_index in enumerate(ranks):
        feature = values[:, feature_index]
        low, high = np.percentile(feature, [5, 95])
        scaled = np.full(len(feature), 0.5) if high <= low else np.clip((feature - low) / (high - low), 0, 1)
        jitter = rng.normal(0, 0.09, len(feature))
        scatter = ax.scatter(shap_values[:, feature_index], position + jitter, c=scaled,
                             cmap="coolwarm", norm=norm, s=20, alpha=0.72,
                             edgecolor="none")
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(np.arange(len(ranks)), [DISPLAY_NAMES[FEATURES[index]] for index in ranks])
    ax.set_xlabel("Permutation-SHAP contribution to NO$_2$ (µg m$^{-3}$)")
    ax.set_title(f"{METHODS[method]} — January 2021 NO$_2$\n"
                 "Positive values raise the prediction; negative values lower it",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2); ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Relative feature value\nlow → high")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"no2_{method}_shap_beeswarm.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    x = data[FEATURES].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    background_indices = np.sort(rng.choice(len(data), BACKGROUND_SIZE, replace=False))
    background = x[background_indices]
    permutations = antithetic_permutations(len(FEATURES), PERMUTATIONS, SEED)
    predictors, diagnostics = fit_prediction_functions(data)

    explanations = {}
    predictions = {}
    base_values = {}
    additivity_errors = {}
    importance_frames = []
    for method, predict in predictors.items():
        shap_values, fitted, base, error = permutation_shap(
            predict, x, background, permutations, method
        )
        explanations[method] = shap_values
        predictions[method] = fitted
        base_values[method] = base
        additivity_errors[method] = error
        importance_frames.append(importance_table(method, shap_values))
        render_beeswarm(x, shap_values, method)

    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    write_long_values(data, x, explanations, predictions, base_values)
    render_importance(importance)

    source_counts = data.groupby("source").size().astype(int).to_dict()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "January 2021 NO2; final models fitted to all complete station rows",
        "predictor_scope": "Current NL predictor stack; not the complete original CHROMAP NO2 stack",
        "original_chromap_features_missing": ["CHEM_Sat_no2", "GEO_UrbDeg"],
        "original_chromap_feature_analogue": {"CHEM_CTM_no2": "cams_value"},
        "station_rows": len(data),
        "source_counts": source_counts,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "method": "Monte Carlo interventional permutation SHAP with antithetic orders",
        "background_size": BACKGROUND_SIZE,
        "background_station_rows": background_indices.tolist(),
        "permutations_per_station": PERMUTATIONS,
        "random_seed": SEED,
        "base_values_ug_m3": base_values,
        "maximum_additivity_error_ug_m3": additivity_errors,
        "model_diagnostics": diagnostics,
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "interpretation": (
            "SHAP values are prediction attributions relative to the mean model output over "
            "the background stations; they are not causal effects. Correlated predictors can "
            "share or exchange attribution under an independent interventional masker."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
