# Local data workspace

This directory is intentionally excluded from Git. Acquisition scripts populate
`raw/`; processing scripts write harmonized inputs to `processed/`. Source URLs,
licences, and acquisition notes are documented in [`../DATA_SOURCES.md`](../DATA_SOURCES.md).

Do not commit credentials, downloaded archives, station-level source files, or
generated rasters. The Apache-2.0 repository licence does not grant rights to
redistribute provider data. Recreate inputs using the
[`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) and the provider terms in
[`../DATA_SOURCES.md`](../DATA_SOURCES.md).

