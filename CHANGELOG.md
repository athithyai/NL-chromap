# Change log

## 2026-10-13 — public CHROMAP-only main branch

- Kept the separate Netherlands CAMS-plus-residual experiment on
  `nl-residual-comparison`; `main` now documents and runs only the direct
  CHROMAP-style ensemble.
- Added a reusable monthly trainer, an aligned pilot-grid feature builder,
  Dutch-only measurement figures, five CHROMAP maps and a same-scale January
  NO₂ CAMS-versus-CHROMAP comparison.
- Published Dutch-only held-out RMSE beside the Dutch-only figures and
  documented the geographic cross-validation definition and limits.
- Reworked the external adaptation guide and architecture for reuse in other
  countries. Raw station records and provider NetCDFs remain outside Git.
- Corrected final-map preprocessing for O₃, PM₁₀, PM₂.₅ and SO₂: the scaler
  and imputer are now fitted on training stations before transforming grid
  cells. Held-out metrics were unchanged; the public map figures were rendered
  from the corrected final fits, so they differ from earlier branch figures.

## 2026-10-12 — external adaptation preparation

- Added Apache-2.0 for original repository code and documentation.
- Documented sources, CHROMAP attribution, research limitations and
  reproducibility steps.
