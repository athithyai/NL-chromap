# Architecture

NL CHROMAP is a file-oriented, reproducible geospatial pipeline. The frozen grid,
target rules, model selection protocol, and output schema remain constant across
years so monthly products are comparable.

## Data flow

```mermaid
flowchart LR
    subgraph Sources
      R[RIVM and EEA observations]
      C[CAMS reanalysis]
      E[ERA5 meteorology]
      S[Population, DEM, land cover, roads]
      B[CBS/PDOK boundary]
    end
    A[Acquisition and checksums]
    H[Temporal and spatial harmonization]
    G[Frozen 500 m EPSG:3035 land grid]
    T[Station-month targets]
    V[Buffered spatial cross-validation]
    M[Pollutant-specific model selection]
    P[Monthly predictions and uncertainty]
    Q[QA, manifests, and release gates]
    N[CF-style NetCDF products]

    R --> A
    C --> A
    E --> A
    S --> A
    B --> G
    A --> H
    H --> G
    H --> T
    G --> V
    T --> V
    V --> M
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
    PROC --> MODEL[spatial CV and residual models]
    MODEL --> OUT[outputs: NetCDF, QA maps, manifests]
    DOC[docs and provenance] -. defines gates .-> CLI
```

## Modelling contract

For each pollutant, CAMS is the baseline. Candidate Extra Trees models learn the
station residual after static and meteorological predictors are aligned. Models
are selected only when buffered spatial cross-validation improves RMSE; otherwise
the CAMS baseline is retained. The 2021 pilot selected residual correction for
NO2 and PM10, and CAMS for O3, PM2.5, and SOMO35.

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
    V->>R: Approved monthly layers
    R->>R: NetCDF metadata, checksum, provenance
```

