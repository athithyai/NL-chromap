"""Package NO2 fields with an embedded reference-palette representation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from pyproj import CRS


COLORS = [
    "#4f83b5", "#3d9fc0", "#55babe", "#83cbb5", "#bed8ae",
    "#e8dfa5", "#f3bd7b", "#e87b49", "#c94835",
]
POSITIONS = [0.00, 0.14, 0.28, 0.43, 0.58, 0.70, 0.82, 0.92, 1.00]
VMIN = 0.0
VMAX = 40.0
NODATA_INDEX = 255


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def styled_dataset(source: xr.Dataset) -> xr.Dataset:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "reference_blue_green_orange", list(zip(POSITIONS, COLORS)), N=255
    )
    data = source.no2.values.astype("float32")
    valid = np.isfinite(data)
    scaled = np.clip((data - VMIN) / (VMAX - VMIN), 0, 1)
    palette_index = np.rint(np.where(valid, scaled, 0) * 254).astype("uint8")
    palette_index[~valid] = NODATA_INDEX

    lut = np.rint(cmap(np.linspace(0, 1, 255)) * 255).astype("uint8")
    # Index 255 represents no-data and uses the reference's deep-blue background.
    background = np.array([[79, 131, 181, 255]], dtype="uint8")
    palette_rgba = np.vstack((lut, background))
    rgba = palette_rgba[palette_index]
    palette_values = np.r_[np.linspace(VMIN, VMAX, 255), np.nan].astype("float32")

    crs = CRS.from_epsg(3035)
    output = xr.Dataset(
        data_vars={
            "no2": source.no2.copy(deep=True),
            "no2_uncertainty": source.no2_uncertainty.copy(deep=True),
            "no2_palette_index": (("time", "y", "x"), palette_index, {
                "long_name": "NO2 display palette index",
                "valid_range": np.array([0, 254], dtype="uint8"),
                "nodata_index": NODATA_INDEX,
                "palette_variable": "palette_rgba",
            }),
            "no2_rgba": (("time", "y", "x", "rgba_channel"), rgba, {
                "long_name": "Ready-rendered NO2 RGBA image",
                "display_scale": "0 to 40 ug m-3, clipped",
                "nodata_rendering": "deep blue background; consult no2 for the scientific mask",
            }),
            "palette_rgba": (("palette_index", "rgba_channel"), palette_rgba, {
                "long_name": "Reference palette red-green-blue-alpha lookup table",
            }),
            "palette_value_ug_m3": (("palette_index",), palette_values, {
                "long_name": "NO2 concentration represented by each palette index",
                "units": "ug m-3",
            }),
            "spatial_ref": ((), np.int32(0), {
                "spatial_ref": crs.to_wkt(), "crs_wkt": crs.to_wkt(),
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "epsg_code": "EPSG:3035",
            }),
        },
        coords={
            "time": source.time.values, "y": source.y.values, "x": source.x.values,
            "palette_index": np.arange(256, dtype="uint16"),
            "rgba_channel": np.arange(4, dtype="uint8"),
        },
        attrs={
            **source.attrs,
            "title": "CHROMAP-NL NO2 with embedded reference palette",
            "palette_name": "reference_blue_green_orange",
            "palette_colors_hex": json.dumps(COLORS),
            "palette_positions": json.dumps(POSITIONS),
            "display_min_ug_m3": VMIN,
            "display_max_ug_m3": VMAX,
            "palette_note": "Styling variables do not alter the scientific NO2 or uncertainty values.",
        },
    )
    output.no2.attrs.update({
        "grid_mapping": "spatial_ref", "ancillary_variables": "no2_uncertainty no2_palette_index",
        "display_min": VMIN, "display_max": VMAX,
        "recommended_colormap": "palette_rgba",
    })
    output.no2_uncertainty.attrs["grid_mapping"] = "spatial_ref"
    return output


def write(dataset: xr.Dataset, path: Path) -> None:
    encoding = {
        "no2": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0},
        "no2_uncertainty": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0},
        "no2_palette_index": {"zlib": True, "complevel": 4, "dtype": "uint8", "_FillValue": NODATA_INDEX},
        "no2_rgba": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "palette_rgba": {"zlib": True, "complevel": 4, "dtype": "uint8"},
        "palette_value_ug_m3": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan},
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def main() -> None:
    source_dir = Path("outputs/pilot_2021")
    output_dir = source_dir / "no2_reference_palette" / "netcdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for month in range(1, 13):
        source_path = source_dir / f"chromap_nl_500m_2021_{month:02d}.nc"
        with xr.open_dataset(source_path) as source:
            dataset = styled_dataset(source)
            path = output_dir / f"no2_nl_500m_2021_{month:02d}_palette.nc"
            write(dataset, path)
        outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        print(path)

    annual_source = source_dir / "chromap_nl_500m_monthly_2021.nc"
    with xr.open_dataset(annual_source) as source:
        annual = styled_dataset(source)
        annual_path = output_dir / "no2_nl_500m_monthly_2021_palette.nc"
        write(annual, annual_path)
    outputs.append({"path": str(annual_path), "bytes": annual_path.stat().st_size, "sha256": sha256(annual_path)})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "palette": {"colors": COLORS, "positions": POSITIONS, "vmin": VMIN, "vmax": VMAX},
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
