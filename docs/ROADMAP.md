# Production roadmap

## Completed foundation

- Ratified RIVM target history for 2013–2025.
- Frozen 500 m EPSG:3035 European-Netherlands land grid.
- Complete 2021 cross-border pilot inputs and twelve-month baseline.
- Buffered spatial validation and pollutant-specific model selection.
- Monthly maps, uncertainty proxies, annual NetCDF, and NO2 palette packaging.

## Production sequence

1. Freeze the 2021 pilot configuration and output schema.
2. Acquire and checksum CAMS, ERA5, and year-appropriate predictors for 2013–2025.
3. Produce all 156 months for NO2, PM2.5, PM10, O3, and SOMO35.
4. Run automated completeness, units, range, seam, and metadata checks.
5. Perform annual and cross-year station validation and temporal-consistency QA.
6. Publish annual pollutant files with manifests, checksums, and provenance.
7. Mark 2025 provisional; validate through 2024 unless newer ratified observations
   support promoting the release.

## Release gate

No annual product is released until all twelve months, land cells, variables,
coordinate metadata, validation tables, and checksums pass. Model changes require
a new version and a documented backfill decision to protect time-series
comparability.

