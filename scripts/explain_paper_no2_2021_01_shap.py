"""Run the audited permutation-SHAP explainer on the paper's 22 NO2 features."""

from pathlib import Path

import explain_no2_2021_01_shap as explainer
from build_paper_no2_features_2021_01 import FEATURES


explainer.OUTPUT_DIR = Path("outputs/paper_chromap_no2_2021_01/shap")
explainer.INPUT_PATH = Path(
    "data/processed/model/paper_no2_2021_01/paper_no2_station_features_2021_01.csv.gz"
)
explainer.FEATURES = FEATURES
explainer.BASELINE_FEATURE = "CHEM_CTM_no2"
explainer.PREDICTOR_SCOPE = "Complete 22-feature NO2 stack documented in the CHROMAP paper"
explainer.ORIGINAL_CHROMAP_FEATURES_MISSING = []
explainer.ORIGINAL_CHROMAP_FEATURE_ANALOGUE = {}
explainer.METHODS = {
    "chromap_style": "Paper-feature CHROMAP direct ensemble",
    "nl_residual": "NL residual model on the same paper features",
}
explainer.DISPLAY_NAMES = {
    "CHEM_CTM_no2": "CHEM_CTM_no2 (CAMS ensemble)",
    "CHEM_Sat_no2": "CHEM_Sat_no2 (TROPOMI)",
    "MET_Precip": "Precipitation (ERA5)", "MET_Rad": "Solar radiation (ERA5)",
    "MET_Pbl": "Boundary-layer height (ERA5)", "MET_Winds": "Wind speed (CERRA)",
    "MET_Temp": "2 m temperature (CERRA)", "MET_SurfP": "Surface pressure (CERRA)",
    "MET_CloudC": "Cloud cover (CERRA)", "GEO_Crop": "Cropland fraction",
    "GEO_Tree": "Tree fraction", "GEO_Grass": "Grass fraction",
    "GEO_Builtup": "Built-up fraction", "GEO_Shrub": "Shrub fraction",
    "GEO_Bare": "Bare fraction", "GEO_Lat": "Latitude", "GEO_Lon": "Longitude",
    "GEO_Alt": "Altitude", "GEO_DistCoast": "Distance to coast",
    "GEO_RoadNet": "Selected-road length", "GEO_PopDens": "Population",
    "GEO_UrbDeg": "GHS-SMOD urban degree",
}


if __name__ == "__main__":
    explainer.main()
