# Change log

The repository records changes to code and documented methods. Generated data
products are versioned separately by their manifests and checksums.

## Unreleased — external adaptation preparation (2026-10-12)

- Added Apache-2.0 for repository code and original documentation.
- Clarified the three different scopes: the 2021 SOMO35 pilot, the January 2021
  five-pollutant comparison with SO₂, and the planned historical series.
- Documented the supplied CHROMAP v1.0 archive as the methodological reference,
  without implying that pretrained weights or input arrays were available.
- Added a guide for external users, including data dependencies, validation,
  known hard-coded pilot settings, and station-level export instructions.
- Added January 2021 station-location figures and reproducible export/plot
  scripts. Kept raw station tables and generated NetCDFs outside Git.
- Excluded local review material and temporary files from Git.

## Earlier research milestones

- Rebuilt the paper-defined 22-feature NO₂ stack for January 2021 and compared
  a newly trained CHROMAP ensemble with the Netherlands residual model using
  identical geographic folds.
- Produced January 2021 comparisons for O₃, PM₁₀, PM₂.₅, and SO₂. The SO₂
  preset is a project extension because it is absent from the supplied archive.
- Built a twelve-month Netherlands pilot for 2021, including SOMO35 as a
  separate ozone exposure indicator.
