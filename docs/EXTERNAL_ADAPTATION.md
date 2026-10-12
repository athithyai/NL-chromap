# Adapting this repository

This is a research pipeline with a completed 2021 pilot and a January 2021
five-pollutant comparison. It is not yet a one-command historical production
system. Start by inspecting the versioned compact validation results, then
acquire the inputs needed for a new run.

## 1. Install and inspect

Use Python 3.11 or newer. Install with `python -m pip install -e ".[pipeline]"`
and run the unit tests as shown in the [README](../README.md). Review the
[source register](../DATA_SOURCES.md), [architecture](ARCHITECTURE.md), and
[runbook](RUNBOOK.md). All commands in this guide assume the repository root as
the working directory.

The CSV metrics and figures in `docs/results/` can be inspected immediately.
They are documentation of the local research runs, not substitutes for the
excluded training tables or maps.

## 2. Obtain source data under provider terms

The repository does not redistribute the CHROMAP v1.0 reference archive,
Copernicus downloads, RIVM/EEA observations, satellite scenes, OpenStreetMap
extracts, or generated NetCDF files. The original CHROMAP archive supplied the
model-training code, feature definitions, and pollutant presets; it did not
include fitted weights or the original processed predictor arrays. Models in
this repository are trained afresh.

Acquire and checksum the sources listed in [DATA_SOURCES.md](../DATA_SOURCES.md).
Copernicus CDS access requires accepting the relevant dataset terms and
configuring `cdsapi`. Keep credentials outside the repository. The expected
workspace is:

```text
data/raw/          downloaded provider files and local manifests
data/processed/    station targets, grid, predictors, and baselines
outputs/           generated model comparisons and maps
output/            local station exports and review material
```

The source files can be large. Several scripts expect specific filenames and
relative paths under these directories; inspect each script's constants before
replacing a source or running it in another region. Only scripts that use
`argparse` provide `--help` options.

## 3. Reproduce the January 2021 comparison

Build the common 2021 observations, grid, CAMS, meteorology, and static
predictors using the [runbook](RUNBOOK.md). The paper-feature NO₂ branch then
uses these steps:

```powershell
python scripts/download_cerra_month.py --year 2021 --month 1
python scripts/build_tropomi_no2_month.py --year 2021 --month 1 --workers 4
python scripts/build_chromap_static_buffer.py
python scripts/build_paper_no2_features_2021_01.py
python scripts/compare_paper_chromap_no2_2021_01.py
python scripts/explain_paper_no2_2021_01_shap.py
```

These commands assume all their earlier common inputs are present. The
other-pollutant comparison needs the 2021 station predictor table and the
separately prepared January SO₂ inputs:

```powershell
python scripts/build_so2_inputs_2021_01.py
python scripts/run_other_pollutants_2021_01.py
```

The NO₂ paper-feature comparison and the other-pollutant comparison use
different feature stacks. SO₂ uses a project-defined CHROMAP-style preset, not
an author-supplied SO₂ preset. The [NO₂ method audit](PAPER_FAITHFUL_NO2_2021_01.md)
lists the remaining differences from the annual CHROMAP publication.

## 4. Validate at stations

The comparison scripts save out-of-fold station predictions. Each fold's
stations are excluded from fitting when their predictions are made. Final maps
are fitted using all eligible stations, so sampling a final map at a training
station measures fit and must not be presented as independent accuracy.

After the January maps and predictions exist, export the station locations and
both kinds of comparison:

```powershell
python scripts/export_january_2021_station_checks.py
python scripts/plot_january_2021_ground_stations.py
```

The exporter normally reads NetCDF with xarray. If the local Python NetCDF
library cannot load, pass a working GDAL executable explicitly:

```powershell
python scripts/export_january_2021_station_checks.py --gdal-translate "C:\path\to\gdal_translate.exe"
```

The local CSV and GeoPackage exports are written under
`output/station_checks_2021_01/` and remain untracked. The two compact station
figures in `docs/results/` are versioned. The figures use a 2026 PDOK boundary
as a static geographical outline; point coordinates represent the January
2021 monitoring records.

## 5. Adapt to a new month, year, or country

1. Choose the study boundary, output CRS, grid resolution, and training buffer.
   Rebuild and freeze the grid before comparing models.
2. Select period-matched monitoring targets and predictors. Document each
   source epoch; static layers such as 2018 land cover or 2026 boundaries are
   proxies when used for 2021 mapping.
3. Rework the scripts that contain `2021`, Netherlands-specific paths,
   source-specific column names, or hard-coded extents. The settings in
   `config/product.yml` document the intended product but are not yet read by
   every script.
4. Validate all competing methods using the same held-out station folds.
   Recompute completeness, RMSE, MAE, bias, coverage, and map QA before
   producing the final all-station raster.
5. Record source licences, software and data versions, grid definition,
   training stations, hashes, and any departure from the CHROMAP reference.
   Update [CHANGELOG.md](../CHANGELOG.md) for method changes.

The project target is NO₂, O₃, PM₁₀, PM₂.₅, and SO₂. The earlier 2021 pilot also
contains SOMO35, a distinct ozone exposure metric. A complete 2013-onward
five-pollutant monthly series has not yet been generated or validated.
