# Reproduction runbook

Run from the repository root with Python 3.11+. Install with
`python -m pip install -e ".[pipeline]"` and run
`python -m unittest discover -s tests -v`. Copernicus acquisition requires
accepted dataset terms and a configured CDS API account. Provider downloads,
monitoring records and generated NetCDFs are excluded from Git.

## January 2021 order

1. Acquire and checksum RIVM/EEA observations, CBS/PDOK boundary, CAMS,
   ERA5/CERRA, TROPOMI and static geographic products listed in
   [DATA_SOURCES.md](../DATA_SOURCES.md).
2. Build the monthly station targets and frozen 500 m EPSG:3035 Dutch grid.
3. Align raster predictors and build station feature tables. For the paper
   NO₂ stack, run `build_paper_no2_features_2021_01.py` after its prerequisites.
   For other pollutants, use the monthly station stack and
   `build_monthly_pilot_feature_grid.py`. SO₂ has a separate January input
   preparation script.
4. Run `train_monthly_chromap.py` separately per pollutant. It writes the
   five-fold geographic held-out predictions and final all-station NetCDF map.
5. Review station coverage, RMSE/MAE/bias, MLP convergence warnings, missing
   grid cells, units, CRS and source manifests before dissemination.

Exact trainer commands and the steps needed for another country are in the
[external adaptation guide](EXTERNAL_ADAPTATION.md). The static and acquisition
helpers still contain 2021 Netherlands paths and extents; inspect their
constants before running a new month or country. `--help` is available for
scripts that use `argparse`.
