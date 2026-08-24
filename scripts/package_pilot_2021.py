"""Package twelve validated pilot files as one annual NetCDF collection."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    paths = sorted(Path("outputs/pilot_2021").glob("chromap_nl_500m_2021_??.nc"))
    datasets = [xr.open_dataset(path).load() for path in paths]
    annual = xr.concat(datasets, dim="time")
    output = Path("outputs/pilot_2021/chromap_nl_500m_monthly_2021.nc")
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(-9999)}
        for name in annual.data_vars if np.issubdtype(annual[name].dtype, np.floating)
    }
    annual.to_netcdf(output, engine="netcdf4", encoding=encoding)
    for dataset in datasets:
        dataset.close()
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
        "months": len(paths), "bytes": output.stat().st_size, "sha256": digest.hexdigest(),
        "monthly_sources": [str(path) for path in paths],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
