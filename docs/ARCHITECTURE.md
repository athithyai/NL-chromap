# Architecture

NL CHROMAP is a file-oriented geospatial research pipeline. The January 2021
comparison shares a fixed grid, station targets, and geographic validation
folds. Historical production will require an explicit decision about monthly
versus pooled training and consistent predictor availability across years.

## Data flow

```mermaid
flowchart LR
    subgraph Sources
      R[RIVM and EEA observations]
      C[CAMS reanalysis]
      E[ERA5 and CERRA meteorology]
      S[Population, DEM, land cover, roads, TROPOMI]
      B[CBS/PDOK boundary]
    end
    A[Acquisition and checksums]
    H[Temporal and spatial harmonization]
    G[Frozen 500 m EPSG:3035 land grid]
    T[Station-month targets]
    V[Shared geographic validation folds]
    C[CHROMAP direct four-model ensemble]
    M[CAMS plus NL residual candidate]
    P[January maps and station comparisons]
    Q[QA, manifests, and release gates]
    N[Local NetCDF products]

    R --> A
    C --> A
    E --> A
    S --> A
    B --> G
    A --> H
    H --> G
    H --> T
    T --> V
    G --> V
    V --> C
    V --> M
    C --> P
    M --> P
    G --> P
    P --> Q
    Q --> N
```

## Software components

```mermaid
flowchart TB
    CFG[config/product.yml] --> CLI[scripts: acquisition, processing, modelling]
    LIB[src/chromap_nl: target and RIVM domain logic] --> CLI
    TEST[tests: aggregation and ingestion contracts] --> LIB
    CLI --> RAW[data/raw: immutable downloads]
    RAW --> PROC[data/processed: aligned layers and tables]
    PROC --> MODEL[geographic CV, direct ensemble, residual candidate]
    MODEL --> OUT[outputs: local NetCDF, station predictions, QA]
    DOC[docs and provenance] -. defines gates .-> CLI
```

## Modelling contract

The original 2021 Netherlands pilot used CAMS as a baseline and selected an
Extra Trees residual correction only when spatial validation improved RMSE.
It selected corrections for NO2 and PM10, and CAMS for O3, PM2.5, and SOMO35.

The January 2021 five-pollutant experiment also trains a CHROMAP-style direct
ensemble of Ridge, Random Forest, XGBoost, and MLP. The NO2 branch reconstructs
the documented 22-feature stack; the other pollutants use the pilot-oriented
feature stack. SO2 has a project-defined preset because the supplied CHROMAP
archive contains no SO2 preset. All comparisons report held-out predictions
from the same five geographic folds within each pollutant. Final maps are
trained using all eligible stations and are not independent validation.

## Release boundary

```mermaid
sequenceDiagram
    participant A as Acquire
    participant P as Process
    participant V as Validate
    participant R as Release
    A->>A: Download, license, checksum
    A->>P: Immutable source manifest
    P->>V: Grid-aligned predictors and targets
    V->>V: Spatial CV, map QA, completeness checks
    V-->>P: Reject or investigate failures
    V->>R: Approved pilot layers
    R->>R: NetCDF metadata, checksum, provenance
```

