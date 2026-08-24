# Reproduction runbook

Run commands from the repository root with Python 3.11 or newer. The acquisition
steps require internet access and acceptance of the relevant provider licences.
Copernicus downloads also require a configured CDS API account.

## Environment

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[pipeline]"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Pipeline order

1. Review `DATA_SOURCES.md`, accept provider licences, and configure CDS access.
2. Acquire RIVM/EEA observations, the boundary, CAMS, ERA5, and static predictors
   using the corresponding `scripts/acquire_*` and `scripts/download_*` commands.
3. Build monthly targets and the frozen Netherlands grid.
4. Align raster predictors and build station-level modelling tables.
5. Train candidates with spatial cross-validation and retain corrections only
   where they improve the CAMS baseline.
6. Generate monthly maps, QA plots, uncertainty proxies, and annual NetCDF files.
7. Verify completeness, metadata, spatial extent, units, checksums, and manifests.

Each script exposes its exact options through `python scripts/<name>.py --help`.
Start with a single month in a scratch workspace before running the full period.

## Expected local layout

```text
data/raw/          immutable provider downloads and source manifests
data/processed/    harmonized targets, grid, predictors, and baselines
outputs/qa/        visual quality-control plots
outputs/pilot_2021 pilot models and monthly/annual NetCDF products
```

## Recovery and reruns

Never modify a downloaded source file in place. If an acquisition is incomplete,
remove only that provider's incomplete target after confirming its manifest, then
rerun the corresponding download command. Generated products can be recreated
from checked source manifests and the versioned configuration.

