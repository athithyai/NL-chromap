# Production roadmap

## Completed foundation

- Ratified RIVM target history for 2013–2025.
- Frozen 500 m EPSG:3035 European-Netherlands land grid.
- Complete 2021 cross-border pilot inputs and twelve-month baseline.
- Buffered spatial validation and pollutant-specific model selection.
- Monthly maps, uncertainty proxies, annual NetCDF, and NO2 palette packaging
  for the 2021 pilot, which includes SOMO35.
- January 2021 paper-feature NO2 and CHROMAP-style O3, PM10, PM2.5, and SO2
  comparisons using geographic validation.

## Production sequence

1. Freeze the five-pollutant target schema: NO2, O3, PM10, PM2.5, and SO2.
2. Acquire and checksum CAMS, ERA5, and year-appropriate predictors for 2013–2025.
3. Choose monthly versus pooled training using comparable geographic validation.
4. Produce the 156 months from January 2013 through December 2025 for NO2,
   PM2.5, PM10, O3, and SO2. Handle any unavailable predictor years explicitly.
5. Run automated completeness, units, range, seam, and metadata checks.
6. Perform annual and cross-year station validation and temporal-consistency QA.
7. Publish annual pollutant files with manifests, checksums, and provenance.
8. Mark each year validated or provisional according to the actual observations
   and predictor products available at release time.

## Release gate

No annual product is released until all twelve months, land cells, variables,
coordinate metadata, validation tables, and checksums pass. Model changes require
a new version and a documented backfill decision to protect time-series
comparability.

