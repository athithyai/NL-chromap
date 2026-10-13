"""Fixed CHROMAP v1.0 pollutant presets used by the monthly adaptation.

SO2 is explicitly a project-defined extension: the supplied archive has no
SO2 preset. These are new models, not pretrained CHROMAP weights.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


PILOT_FEATURES = [
    "cams_value", "log1p_population_density", "elevation",
    "landcover_tree_fraction", "landcover_shrub_fraction", "landcover_grass_fraction",
    "landcover_cropland_fraction", "landcover_built_up_fraction",
    "landcover_bare_sparse_fraction", "landcover_permanent_water_fraction",
    "landcover_wetland_fraction", "road_density_major", "road_density_secondary",
    "road_density_local", "road_density_motorized_total", "era5_u10", "era5_v10",
    "era5_t2m", "era5_blh", "era5_sp", "era5_tcc", "era5_ssrd", "era5_tp",
    "era5_wind_speed", "month_sin", "month_cos",
]

PAPER_NO2_FEATURES = [
    "CHEM_CTM_no2", "CHEM_Sat_no2", "MET_Precip", "MET_Rad", "MET_Pbl",
    "MET_Winds", "MET_Temp", "MET_SurfP", "MET_CloudC", "GEO_Crop",
    "GEO_Tree", "GEO_Grass", "GEO_Builtup", "GEO_Shrub", "GEO_Bare",
    "GEO_Lat", "GEO_Lon", "GEO_Alt", "GEO_DistCoast", "GEO_RoadNet",
    "GEO_PopDens", "GEO_UrbDeg",
]


def chromap_models(indicator: str) -> dict[str, object]:
    """Return unfitted Ridge, RF, XGBoost and MLP estimators in ensemble order."""
    if indicator == "o3":
        ridge = dict(alpha=3.5)
        forest = dict(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=.7, sampling_method="uniform", n_estimators=1000,
                       min_child_weight=5, max_depth=2, max_delta_step=3,
                       learning_rate=.3, reg_lambda=100, gamma=.4,
                       colsample_bytree=.5, reg_alpha=100)
        mlp = dict(solver="lbfgs", momentum=.7, learning_rate_init=.01,
                   learning_rate="constant", hidden_layer_sizes=(100, 100, 100),
                   alpha=.5, activation="identity")
    elif indicator in {"pm10", "pm2p5"}:
        ridge = dict(alpha=2.0)
        forest = dict(n_estimators=1000, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=30, bootstrap=True)
        boosted = dict(subsample=.5, sampling_method="uniform", n_estimators=1500,
                       min_child_weight=7, max_depth=5, max_delta_step=7,
                       learning_rate=.01, reg_lambda=10, gamma=.4,
                       colsample_bytree=1., reg_alpha=1)
        mlp = dict(solver="adam", momentum=.9, learning_rate_init=.001,
                   learning_rate="invscaling", hidden_layer_sizes=(150,),
                   alpha=.5, activation="tanh")
    elif indicator == "no2":
        ridge = dict(alpha=2.9)
        forest = dict(n_estimators=500, min_samples_split=10, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=1., sampling_method="uniform", n_estimators=1000,
                       min_child_weight=1, max_depth=5, max_delta_step=5,
                       learning_rate=.01, reg_lambda=10, gamma=1,
                       colsample_bytree=.5, reg_alpha=100)
        mlp = dict(solver="adam", momentum=.9, learning_rate_init=.001,
                   learning_rate="constant", hidden_layer_sizes=(150, 150, 150),
                   alpha=.0001, activation="logistic")
    elif indicator == "so2":
        ridge = dict(alpha=3.)
        forest = dict(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                      max_features="sqrt", max_depth=10, bootstrap=True)
        boosted = dict(subsample=.8, sampling_method="uniform", n_estimators=500,
                       min_child_weight=3, max_depth=2, max_delta_step=3,
                       learning_rate=.03, reg_lambda=10, gamma=.1,
                       colsample_bytree=.7, reg_alpha=1)
        mlp = dict(solver="lbfgs", momentum=.9, learning_rate_init=.001,
                   learning_rate="constant", hidden_layer_sizes=(50,),
                   alpha=.1, activation="relu")
    else:
        raise ValueError(f"Unsupported pollutant: {indicator}")
    return {
        "ridge": Ridge(**ridge),
        "random_forest": RandomForestRegressor(**forest, random_state=42, n_jobs=-1),
        "xgboost": XGBRegressor(**boosted, random_state=42, n_jobs=-1, tree_method="hist"),
        "mlp": MLPRegressor(**mlp, random_state=42, max_iter=200),
    }
