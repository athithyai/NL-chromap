# Reproduction runbook

Run commands from the repository root with Python 3.11 or newer. The acquisition
steps require internet access and acceptance of the relevant provider licences.
Copernicus downloads also require a configured CDS API account. The current
scripts constitute research pilots; many expect particular 2021 input names.

## Environment

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[pipeline]"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Research pipeline order

1. Review `DATA_SOURCES.md`, accept provider licences, and configure CDS access.
2. Acquire RIVM/EEA observations, the boundary, CAMS, ERA5, and static predictors
   using the corresponding `scripts/acquire_*` and `scripts/download_*` commands.
3. Build monthly targets and the frozen Netherlands grid.
4. Align raster predictors and build station-level modelling tables.
5. Train candidates with geographic cross-validation. Compare direct CHROMAP
   models, the CAMS baseline, and residual corrections on the same folds.
6. Generate pilot maps, QA plots, and NetCDF files for the selected experiment.
7. Compare held-out predictions with station measurements. Treat final-map
   values at training stations as descriptive fit only.
8. Verify completeness, metadata, spatial extent, units, checksums, and manifests.

Only scripts using `argparse` expose options through
`python scripts/<name>.py --help`. Other scripts contain fixed research-pilot
paths and should be inspected before rerunning. The January 2021 NO₂ and
other-pollutant commands are listed in the
[external adaptation guide](EXTERNAL_ADAPTATION.md).

## Expected local layout

```text
data/raw/          immutable provider downloads and source manifests
data/processed/    harmonized targets, grid, predictors, and baselines
outputs/qa/        visual quality-control plots
outputs/pilot_2021/   2021 pilot models and monthly/annual NetCDF products
outputs/paper_chromap_no2_2021_01/   paper-feature NO2 comparison
outputs/comparison_2021_01_other_pollutants/   O3, PM10, PM2.5, SO2 comparison
output/station_checks_2021_01/   local station exports
```

## Recovery and reruns

Never modify a downloaded source file in place. If an acquisition is incomplete,
remove only that provider's incomplete target after confirming its manifest, then
rerun the corresponding download command. Generated products can be recreated
from checked source manifests and the versioned configuration.

