# Authoritative data-source register

This register defines the preferred sources for the monthly CHROMAP-NL product. Dataset versions, request payloads, retrieval times, file hashes, and licence identifiers must also be recorded in the machine-readable manifest produced during acquisition.

## Chemical baseline

### CAMS European air-quality reanalyses

- Provider: Copernicus Atmosphere Monitoring Service / ECMWF and the regional modelling consortium.
- Dataset ID: `cams-europe-air-quality-reanalyses`.
- DOI: <https://doi.org/10.24381/7cc0465a>.
- Catalogue: <https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-reanalyses>.
- API catalogue record: <https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/cams-europe-air-quality-reanalyses>.
- Product: ensemble median, surface level, validated reanalysis when available and interim reanalysis only for explicitly provisional years.
- Variables: nitrogen dioxide, ozone, particulate matter below 2.5 micrometres, and particulate matter below 10 micrometres.
- Resolution: 0.1 degree, hourly, supplied as monthly files.
- Coverage: 2013 onward; the catalogue currently exposes 2013-2025.
- Licence: CC BY 4.0.
- Processing: calculate monthly means from hourly fields and monthly SOMO35 contributions from hourly O3 before reprojecting to the 500 m target grid.

The collection currently supplies complete European files rather than a Netherlands-only API subset. Acquisition is therefore staged one month at a time, cropped immediately to the buffered training domain, and recorded by checksum.

## Surface observations

### RIVM Luchtmeetnet

- Provider: Rijksinstituut voor Volksgezondheid en Milieu and participating Dutch monitoring networks.
- Archive: <https://data.rivm.nl/data/luchtmeetnet/>.
- Documentation: <https://data.rivm.nl/data/luchtmeetnet/readme.pdf>.
- Product: ratified hourly observations under `Vastgesteld-jaar`, with provisional data used only when explicitly labelled.
- Pollutants: NO2, O3, PM10, and PM2.5.
- Metadata: station coordinates, measurement-series classifications, area type, station type, measurement height, methods, and network.
- Time convention: source timestamps include fixed UTC+1 offsets; ingestion converts them to UTC while preserving source timestamps and metadata.
- Important quality rule: the archive omits rejected/missing values rather than inserting placeholder rows. Completeness must be reconstructed from expected hourly timestamps.

### EEA Air Quality Download Service

- Provider: European Environment Agency.
- Landing page: <https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d>.
- API: <https://eeadmz1-downloads-api-appservice.azurewebsites.net/>.
- Product: verified and up-to-date time-series data supplied as Parquet files.
- Coverage: 2013 onward.
- Use: cross-border training observations and an independent consistency check on Netherlands station data.
- Acquired pilot stack: 3,943 verified E1a station-series files for Belgium, Germany, France, and Luxembourg (4,525,719,535 bytes). The API URL-list method returns full series; the downstream preparation step selects 2021 and clips stations to the training buffer.
- Reproducibility: `data/raw/eea_air_quality/manifest.json` records every source URL, byte count, and SHA-256 digest.
- Station geometry: the official EEA ArcGIS monitoring-station service is archived as `data/raw/eea_air_quality/station_locations.geojson`; its EoI station codes are joined to the codes embedded in Parquet filenames. This avoids obsolete identifier mismatches in the older sampling-point metadata extract.

RIVM is the preferred Netherlands record because it supplies the authoritative ratified hourly source directly. EEA is required for the Belgium/Germany buffer and for harmonised European station classifications.

## Meteorological predictors

### CERRA

- Provider: Copernicus Climate Change Service / ECMWF.
- Dataset: CERRA sub-daily regional reanalysis on single levels.
- DOI: <https://doi.org/10.24381/cds.622a565a>.
- Preferred variables: 10 m wind components, 2 m temperature, surface pressure, and total cloud cover where available and definitionally compatible.
- Processing: calculate monthly statistics from sub-daily fields using variable-specific aggregation rules.

### ERA5

- Provider: Copernicus Climate Change Service / ECMWF.
- Dataset: ERA5 hourly data on single levels.
- DOI: <https://doi.org/10.24381/cds.adbb2d47>.
- Preferred variables: total precipitation, surface solar radiation downwards, and boundary-layer height.
- Processing: accumulated variables must be converted to physically correct monthly totals or means before interpolation.
- Acquired pilot layer: all 12 months of 2021 for the 0-11 E, 49-56 N buffered domain. The package contains monthly `u10`, `v10`, `t2m`, `blh`, `sp`, `tcc`, `ssrd`, and `tp` fields on a 29 x 45 grid. The archive and its two NetCDF members are checksummed in `data/raw/era5/era5_monthly_nl_buffer_2021.manifest.json`.
- CDS delivery detail: accumulated (`ssrd`, `tp`) and instantaneous variables are supplied in separate NetCDF streams and must retain their variable-specific aggregation semantics during predictor preparation.

## Geographic predictors

### ESA WorldCover 2021

- Provider: European Space Agency / WorldCover consortium.
- Product: WorldCover 2021 v200, 10 m categorical land cover, DOI <https://doi.org/10.5281/zenodo.7254221>.
- Access: <https://esa-worldcover.org/en/data-access>.
- Acquired: 10 GeoTIFF tiles covering the Netherlands and buffered domain (708,409,131 bytes), recorded in `data/raw/esa_worldcover/2021_v200/manifest.json`.
- Licence: CC BY 4.0.
- Use: crop, tree, grass, built-up, shrub, and bare-soil fractions.
- Temporal treatment: freeze the 2021 version and describe it as a static spatial proxy; do not present it as annual historical land cover.

### GHSL / Eurostat GISCO

- Providers: European Commission Joint Research Centre and Eurostat.
- GHSL: <https://human-settlement.emergency.copernicus.eu/>.
- GISCO grids: <https://ec.europa.eu/eurostat/web/gisco/geodata/grids>.
- Use: population density and degree of urbanisation.
- Processing: select documented reference-year snapshots and avoid pretending interpolated census values are direct annual observations.
- Acquired population layer: Eurostat Census Grid 2021 v1.0, 1 km, EPSG:3035 (186,301,606-byte archive; SHA-256 `da38dae9398f9cf1dbe059872f249eaeb34430eb35118b1d57dd530cb41f8b35`). Manifest: `data/raw/eurostat/Eurostat_Census-GRID_2021_V1-0.manifest.json`.

### Copernicus DEM

- Provider: Copernicus.
- Product/access: Copernicus DEM GLO-30, 2021 release, via the official public registry <https://registry.opendata.aws/copernicus-dem/>.
- Acquired: 57 one-degree tiles covering 0-11 E and 49-56 N where land data exist (1,598,408,761 bytes), with per-tile hashes in `data/raw/copernicus_dem/glo30/manifest.json`.
- Use: elevation aggregated to the 500 m target grid.

### Roads

- Candidate sources: Dutch National Road Database/PDOK for the Netherlands and OpenStreetMap for a consistent cross-border layer.
- Use: road length by class and total road density within each 500 m cell.
- Decision rule: prefer the authoritative Dutch source for the final Netherlands product if historical/versioned extracts are accessible; preserve a consistent cross-border alternative for training.
- Acquired consistent stack: Geofabrik OpenStreetMap PBF extracts for the Netherlands, Belgium, Luxembourg, Nordrhein-Westfalen, Niedersachsen, Rheinland-Pfalz, and Nord-Pas-de-Calais (4,056,073,103 bytes). Resolved dated snapshot URLs and hashes are in `data/raw/osm_geofabrik/manifest.json`; OpenStreetMap data are licensed under ODbL 1.0.

### Boundaries, coastline, and water mask

- Selected Netherlands source: CBS Wijk- en buurtkaart 2026 v0 via PDOK.
- Dataset page: <https://www.pdok.nl/ogc-apis/-/article/cbs-wijken-en-buurten>.
- ATOM file: <https://service.pdok.nl/cbs/wijken-en-buurten-2026/atom/downloads/WijkBuurtkaart_2026_v0.gpkg>.
- Selection: union all `buurten` polygons where `water = NEE`. CBS states that the land-water boundary is taken from Bestand Bodemgebruik 2022; the municipality boundary is derived from the Kadaster BRK.
- Retrieved file SHA-256: `408c75b18914c9ee23314ca41b56ab409d7b383f47f3a18171f1d9a102e91075`.
- Frozen output grid: EPSG:3035, globally aligned 500 m cells, cell-centre inclusion, 140,260 land cells representing 35,065 km2.
- European fallback and cross-border source: Eurostat GISCO.
- Use: immutable EPSG:3035 output grid, land mask, coast distance, and publication clipping.

## Optional satellite enhancement

### Sentinel-5P TROPOMI NO2

- Provider: Copernicus Sentinel-5P.
- Use: optional NO2 total-column feature for 2019 onward after quality filtering.
- Production status: excluded from the temporally consistent 2013-2025 core until a 2019+ sensitivity analysis shows spatial-validation improvement without a structural break.

## Source-selection principles

1. Prefer validated and versioned products over near-real-time values.
2. Preserve original units and timestamps before transformation.
3. Record licence, DOI, request payload, retrieval time, and SHA-256 hash.
4. Never replace missing observations with zero.
5. Treat CAMS as both a predictor and benchmark, and avoid describing the resulting maps as independent of CAMS.
6. Mark every output derived from interim inputs as provisional.
