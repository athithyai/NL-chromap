# Architecture

The `main` branch is a file-oriented monthly **direct CHROMAP ensemble**
adaptation. The January 2021 result is a research pilot, not a production
time series. A different Netherlands residual experiment is retained only on
the `nl-residual-comparison` branch.

```mermaid
flowchart LR
  A[RIVM + EEA monthly station targets] --> H[Quality and temporal harmonisation]
  B[CAMS, TROPOMI, ERA5, CERRA, land cover, roads, DEM] --> F[Aligned monthly predictors]
  C[PDOK land boundary] --> G[500 m EPSG:3035 grid and mask]
  H --> T[Station feature table]
  F --> T
  F --> R[Gridded feature NetCDF]
  G --> R
  T --> V[Five geographic held-out folds]
  V --> E[Ridge + RF + XGBoost + MLP]
  E --> S[Held-out RMSE, MAE, bias]
  T --> E2[Final fit on all eligible stations]
  R --> E2
  E2 --> M[Four-member median NetCDF map]
  S --> Q[QA and provenance review]
  M --> Q
```

`scripts/train_monthly_chromap.py` implements the direct ensemble and spatial
validation after input preparation. `scripts/chromap_models.py` records the
fixed pollutant presets; the SO₂ preset is a project extension. The NO₂ input
branch uses the 22 documented paper predictors. The four other pollutants
use a monthly pilot predictor stack. `src/chromap_nl/` contains reusable
monitoring-target logic; `scripts/` contains source acquisition, raster
preparation and modelling tasks.

The final map's station overlay is not validation because its fit used those
stations. Only the out-of-fold predictions support the reported held-out
scores. Raw inputs and generated NetCDFs remain local and are not committed.
