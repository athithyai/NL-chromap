# NL CHROMAP: monthly adaptation

This repository is an independent research adaptation of [CHROMAP v1.0](ACKNOWLEDGEMENTS.md)
for 500 m monthly maps over the European Netherlands. **`main` contains only
the direct CHROMAP-style four-model median ensemble.** A distinct Netherlands
CAMS-plus-residual experiment is preserved on the
[`nl-residual-comparison` branch](https://github.com/athithyai/NL-chromap/tree/nl-residual-comparison);
its results are not mixed with those below. This is not an official product of
the CHROMAP authors.

## January 2021 results

The following is a **single-station example** for Hoogvliet–Leemkuil
(`NL01485`) in January 2021. All entries are monthly concentrations in µg/m³,
**not RMSE**. The map column samples the current final CHROMAP map at the
station's 500 m cell; click a pollutant name to view that map.

| Pollutant | Ground measurement | CHROMAP map | CHROMAP held-out |
| --- | ---: | ---: | ---: |
| [NO₂](docs/results/no2_2021_01_chromap_map.png) | 23.44 | 27.05 | 24.64 |
| [O₃](docs/results/o3_2021_01_chromap_map.png) | 30.02 | 29.94 | 31.16 |
| [PM₁₀](docs/results/pm10_2021_01_chromap_map.png) | 14.73 | 15.39 | 14.42 |
| [PM₂.₅](docs/results/pm2p5_2021_01_chromap_map.png) | 8.82 | 9.44 | 10.11 |
| [SO₂](docs/results/so2_2021_01_chromap_map.png) | 1.65 | 2.15 | 3.76 |

`CHROMAP held-out` is the station's out-of-fold prediction: its geographic
fold was excluded when that prediction was made. The final map was fitted
using all eligible stations, including this one, so its sampled value is not
an independent accuracy test. Aggregate spatial-validation scores and the
fold definition are in [validation details](docs/VALIDATION_2021_01.md).
The current map samples for O₃, PM₁₀, PM₂.₅ and SO₂ differ slightly from
earlier comparison exports because final-map preprocessing was corrected to
fit on training stations before transforming grid cells; held-out predictions
were unchanged. This correction is recorded in the [change log](CHANGELOG.md).

Dutch-only ground-measurement figures for each pollutant are available for
[NO₂](docs/results/no2_2021_01_dutch_measurements.png),
[O₃](docs/results/o3_2021_01_dutch_measurements.png),
[PM₁₀](docs/results/pm10_2021_01_dutch_measurements.png),
[PM₂.₅](docs/results/pm2p5_2021_01_dutch_measurements.png), and
[SO₂](docs/results/so2_2021_01_dutch_measurements.png).

## NO₂: CHROMAP versus CAMS

![January 2021 NO₂ CAMS baseline and CHROMAP adaptation on the same colour scale](docs/results/no2_2021_01_chromap_vs_cams.png)

Both panels use the same 0–40 µg/m³ scale and 500 m output grid. CAMS is the
regional chemical-transport reanalysis input; the CHROMAP adaptation uses it
among 22 predictors and adds spatial detail through the trained ensemble.
On Dutch held-out stations, RMSE was **4.120 µg/m³ for CAMS** and **3.570 µg/m³
for CHROMAP** in this January experiment. A visually finer map alone does not
prove better accuracy; the held-out comparison is the relevant check.

## What was adapted

The supplied CHROMAP v1.0 archive provided feature definitions, model code,
pollutant hyperparameters, preprocessing, lower-bound and median-ensemble
procedure—not fitted weights or the original processed training arrays. We
rebuilt predictors and trained new Ridge, Random Forest, XGBoost and MLP
models. The paper describes annual European mapping; this repository trains
independent January 2021 monthly models over a Netherlands-centred domain.
The NO₂ branch uses the documented 22-feature stack. O₃, PM₁₀ and PM₂.₅ use
a project-prepared monthly predictor stack and the supplied pollutant presets.
**SO₂ uses a project-defined preset**, because the supplied archive contained
no SO₂ preset. These four non-NO₂ results are CHROMAP-style adaptations, not
exact reproductions of the paper's feature tables. See the
[method audit](docs/PAPER_FAITHFUL_NO2_2021_01.md).

## Run or adapt

Install Python 3.11+ and the pipeline dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[pipeline]"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Raw observations, downloaded predictors and generated NetCDFs are not in Git.
Follow the [runbook](docs/RUNBOOK.md), [data-source register](DATA_SOURCES.md)
and [external adaptation guide](docs/EXTERNAL_ADAPTATION.md). The generic
`scripts/train_monthly_chromap.py` runner takes a station feature table and an
aligned gridded feature NetCDF; it trains the direct ensemble, saves held-out
station predictions and writes the final NetCDF map. Acquisition and feature
construction remain region-specific. Germany therefore requires German data,
a German grid and predictor preparation, retraining and fresh validation.

This is a January 2021 research result, **not** a complete 2013-onward monthly
product. The public figures use a 2026 PDOK boundary as the outline/land mask
for the 2021 experiment; this time mismatch is documented rather than hidden.

## Licence and credit

Original repository code and documentation are [Apache-2.0 licensed](LICENSE).
Source datasets and the original CHROMAP archive retain their own terms and
are not redistributed here. See [acknowledgements](ACKNOWLEDGEMENTS.md),
[source register](DATA_SOURCES.md), and [change log](CHANGELOG.md).
