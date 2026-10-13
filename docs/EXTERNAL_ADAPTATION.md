# Adapt the monthly CHROMAP workflow to another country

This guide is for reusing the **direct CHROMAP-style ensemble** on `main`.
The separate `nl-residual-comparison` branch is a different experiment and is
not needed. The public repository contains scripts, aggregate validation
metrics and figures, but **not** provider data, station-level records, fitted
models or NetCDF maps. It is a research adaptation, not a one-command Germany
product.

## 1. Prepare a new country workspace

Install Python 3.11+ with `python -m pip install -e ".[pipeline]"` and run
`python -m unittest discover -s tests -v`. Read the
[source register](../DATA_SOURCES.md), [runbook](RUNBOOK.md), and
[architecture](ARCHITECTURE.md). Accept each provider's data terms and keep
credentials outside Git.

For Germany, obtain a German land boundary and monitoring observations for the
target month, along with nearby-country stations if a training buffer is used.
Choose a documented output grid (the Netherlands example uses EPSG:3035 at
500 m) and rebuild the mask. Do not reuse the Dutch PDOK boundary or Dutch
station table. The current acquisition/preparation scripts contain Dutch
source paths and regional extents, so adapt them explicitly.

## 2. Build matched monthly targets and features

Prepare one station row per pollutant, station and month with `value` in
µg/m³, `x`/`y` in the grid CRS, `month` as `YYYY-MM`, and the feature columns.
Recompute completeness from the source observations. Align every gridded
feature to the same `y, x` grid, with a `land_mask` or a separate mask NetCDF.
Do not use a later-period predictor without documenting its epoch. For the
paper-defined NO₂ experiment, the [22 features and source choices](PAPER_FAITHFUL_NO2_2021_01.md)
are the reference. The other four January maps use a project-prepared monthly
stack. SO₂ has a **project-defined** preset, not one from the supplied CHROMAP
archive.

The Netherlands January input tables can be created with the existing
acquisition and predictor scripts after obtaining all provider files. For
NO₂, `scripts/build_paper_no2_features_2021_01.py` writes the station and
gridded paper-feature inputs. For O₃, PM₁₀, PM₂.₅ and SO₂,
`scripts/build_station_predictors_2021.py` (and
`scripts/build_so2_inputs_2021_01.py` for SO₂) prepares station predictors;
`scripts/build_monthly_pilot_feature_grid.py` combines aligned raster inputs.
These helpers still assume the 2021 Netherlands file layout. Replace their
paths and domain rules for Germany.

## 3. Train and validate

`scripts/train_monthly_chromap.py --help` lists required inputs. For the
Netherlands NO₂ example, after its prerequisites are built:

```powershell
python scripts/train_monthly_chromap.py --pollutant no2 --feature-set paper-no2 --month 2021-01 --stations data/processed/model/paper_no2_2021_01/paper_no2_station_features_2021_01.csv.gz --grid-features data/processed/model/paper_no2_2021_01/paper_no2_grid_features_2021_01.nc --grid-mask data/processed/grid/chromap_nl_500m_grid.nc --output-dir outputs/chromap_no2_2021_01
```

For the other four pollutants, first build each grid feature stack and then
call the same trainer with `--feature-set monthly-pilot`, its matching station
table and output directory. The runner uses fixed pollutant presets, median
imputation/scaling, the lower-bound rule and a four-member median. It saves
five-fold geographic out-of-fold station predictions, scores and a final map
trained on all eligible stations.

**Held-out** means a station is excluded when its fold is predicted. These
scores are cross-validation, not an untouched external test set. Overlaying
ground stations on the final map shows coverage, not independent accuracy.
Use matching station folds when comparing a CAMS baseline. Report Dutch-only
and all-training-station scores separately, as in the
[January example](VALIDATION_2021_01.md).

## 4. Review before sharing results

- Inspect coverage, units, missingness, predictor epochs and map seams.
- Record source versions, request parameters, licences, hashes, station counts,
  grid definition, folds, metrics, model warnings and any method departures.
- Keep raw station data and downloaded archives under `data/`, and generated
  NetCDFs under `outputs/`; these paths are excluded from Git.
- Reassess the monthly-versus-pooled training decision with German held-out
  stations. January Netherlands skill cannot be assumed to transfer.

Only January 2021 five-pollutant maps are documented on `main`. A complete
2013-onward series has not been produced or validated.
