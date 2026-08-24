# CHROMAP-NL 2021 pilot results

## Outcome

The 2021 pilot produced twelve monthly 500 m Netherlands maps for NO2, O3, PM10, PM2.5, and SOMO35, plus an uncertainty-proxy layer for each indicator. The release is a spatially validated hybrid: statistical residual downscaling is used only where it improves held-out spatial performance; CAMS is retained where the candidate models perform worse.

## Training and validation

- Observations: ratified RIVM data plus verified EEA E1a observations from Belgium, Germany, France, and Luxembourg within 150 km of the Netherlands.
- Harmonized target table: 12,795 unique station-month-indicator rows at 334 stations.
- Predictors: CAMS, ERA5, Eurostat population, Copernicus DEM, ESA WorldCover fractions, and OSM road density.
- Validation: five geographically clustered station folds. All months from a station remain in the same fold.
- Models compared: raw CAMS, random-forest residual correction, and extremely randomized-tree residual correction.

| Indicator | CAMS RMSE | Selected RMSE | Selected product |
|---|---:|---:|---|
| NO2 | 7.27 | 4.73 | Extra Trees residual correction |
| O3 | 4.13 | 4.13 | CAMS baseline |
| PM10 | 3.37 | 3.11 | Extra Trees residual correction |
| PM2.5 | 1.62 | 1.62 | CAMS baseline |
| SOMO35 | 59.69 | 59.69 | CAMS baseline |

The go decision is therefore **yes for the hybrid pipeline**, but not for unconditional statistical correction. NO2 and PM10 pass the improvement gate; O3, PM2.5, and SOMO35 remain CAMS-based until a stronger model demonstrates spatial-validation improvement.

## Outputs

- Twelve monthly NetCDF files: `outputs/pilot_2021/chromap_nl_500m_2021_MM.nc`.
- Combined annual collection: `outputs/pilot_2021/chromap_nl_500m_monthly_2021.nc`.
- Spatial validation metrics: `data/processed/model/validated_2021/spatial_cv_metrics_2021.csv`.
- Out-of-fold predictions: `data/processed/model/validated_2021/spatial_cv_predictions_2021.csv.gz`.
- Model artifacts and feature importance: `data/processed/model/validated_2021/`.
- QA figures: `outputs/pilot_2021/qa/`.

## Interpretation limits

- The uncertainty variables are distance-inflated spatial-CV RMSE proxies, not probabilistic confidence intervals.
- Population and land-cover inputs are frozen 2021 spatial proxies. Historical production must use documented temporal treatment rather than imply they are annual measurements.
- OSM roads are a present-day static proxy.
- The maps remain partly or wholly dependent on CAMS and must not be described as independent of CAMS.
- This validates the 2021 pilot only. The 2013-2025 production run still requires annual/monthly acquisition, validation checks, and provisional labelling where applicable.
