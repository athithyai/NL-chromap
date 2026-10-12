# Contributing

Work on a feature branch and keep changes reproducible. Before opening a pull
request, run `python -m unittest discover -s tests -v` after installing the
project, and document any change to data sources, aggregation rules, grids,
features, validation, or output metadata. Update `CHANGELOG.md` for material
changes and cite the source product and version for new predictors.

Do not commit downloaded data, station-level records, generated NetCDFs,
credentials, private correspondence, machine-specific paths, or
provider-restricted material. Keep local exports under `output/`; generated
model products belong under `outputs/`. Compact aggregate metrics and figures
may be added to `docs/results/` after checking their attribution and metadata.

Changes to the frozen grid, pollutant definitions, completeness thresholds,
spatial validation, or model-selection rule are methodology changes. Include a
rationale, relevant tests, and an assessment of whether earlier products need
rebuilding. Contributions to this repository are submitted under Apache-2.0.

