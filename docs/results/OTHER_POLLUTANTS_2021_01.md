# January 2021 comparison: O3, PM10, PM2.5, and SO2

This run extends the paired January 2021 comparison to four pollutants. SOMO35 is deliberately excluded. Every reported score is based on the same five geographic folds for the competing methods within a pollutant; no January validation station is used to train its own prediction.

## Methods

- **CAMS baseline:** January 2021 CAMS validated ensemble reanalysis, sampled directly at held-out stations.
- **CHROMAP-style:** direct prediction from the supplied CHROMAP model families (Ridge, random forest, XGBoost, and MLP), combined by the median. The supplied archive has no SO2 preset, so SO2 uses a conservative fixed extension within those same model families and parameter ranges.
- **NL residual candidate:** Extra Trees learns station residuals relative to CAMS. The NL improvement gate retains either the residual model or unmodified CAMS according to lower out-of-fold RMSE.

Predictors combine CAMS, monthly meteorology, static geography, land cover, population, elevation, roads, coast distance, and coordinates. Targets are January monthly means from ratified RIVM observations and verified EEA cross-border observations. SO2 uses 14 RIVM and 34 EEA stations after filtering.

## Paired spatial-validation result

| Pollutant | Stations | CAMS RMSE | CHROMAP-style RMSE | NL residual RMSE | NL selected | Interpretation |
|---|---:|---:|---:|---:|---|---|
| O3 | 159 | 3.189 | 2.892 | 3.135 | residual | CHROMAP-style is nominally lowest, but its paired bootstrap interval against NL crosses zero. |
| PM10 | 207 | 2.686 | 2.619 | 2.624 | residual | Effectively a tie; the RMSE gap is 0.006 micrograms per cubic metre. |
| PM2.5 | 152 | 1.649 | 1.807 | 1.625 | residual | NL has lower RMSE; the paired RMSE-difference interval favors NL. |
| SO2 | 48 | 1.629 | 1.581 | 1.633 | CAMS | The residual candidate fails the NL improvement gate. CHROMAP-style is nominally lower, but evidence is not decisive and the station set is sparse. |

RMSE units are micrograms per cubic metre. The paired bootstrap is descriptive because it resamples stations without modelling remaining spatial dependence. SO2 has only 48 stations, with two folds containing five and four stations, so its ranking is especially uncertain. MLP training reached its iteration limit in all five folds for O3, PM10, and PM2.5; this is recorded rather than hidden.

## Products

The local output directory `outputs/comparison_2021_01_other_pollutants` contains paired NetCDF maps, individual PNG maps, three-panel comparison figures, all out-of-fold predictions, fold metrics, a validation chart, and the machine-readable summary. Each NetCDF uses the frozen 500 m EPSG:3035 Netherlands land grid and stores concentrations in `ug m-3`.

The result is suitable for comparison and method development with the stated caveats. It is not yet a claim that one modelling family is universally superior.
