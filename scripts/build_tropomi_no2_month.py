"""Build a QA-filtered monthly TROPOMI total-column NO2 predictor.

The script reads the public Copernicus Sentinel-5P RPRO v2.4 archive through
HTTP byte ranges. Only orbits intersecting the Netherlands training buffer are
read beyond geolocation, so global ~600 MB orbit files are not downloaded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import h5py
import numpy as np
import requests
import xarray as xr
from scipy.ndimage import distance_transform_edt


BUCKET = "https://meeo-s5p.s3.amazonaws.com/"
NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TOTAL_COLUMN = "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_total_column"
BBOX = (0.0, 49.0, 11.0, 56.0)
RESOLUTION_DEGREES = 0.05


def list_month(year: int, month: int) -> list[dict[str, object]]:
    params = {
        "list-type": "2",
        "prefix": f"RPRO/L2__NO2___/{year}/{month:02d}/",
        "max-keys": "1000",
    }
    records: list[dict[str, object]] = []
    while True:
        response = requests.get(BUCKET, params=params, timeout=120)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("s3:Contents", NAMESPACE):
            records.append({
                "key": item.findtext("s3:Key", namespaces=NAMESPACE),
                "bytes": int(item.findtext("s3:Size", namespaces=NAMESPACE)),
                "etag": item.findtext("s3:ETag", namespaces=NAMESPACE).strip('"'),
                "last_modified": item.findtext("s3:LastModified", namespaces=NAMESPACE),
            })
        token = root.findtext("s3:NextContinuationToken", namespaces=NAMESPACE)
        if not token:
            return records
        params["continuation-token"] = token


def candidate(record: dict[str, object]) -> bool:
    match = re.search(r"____\d{8}T(\d{2})\d{4}_", str(record["key"]))
    return bool(match and 8 <= int(match.group(1)) <= 13)


def read_orbit(record: dict[str, object]) -> dict[str, object]:
    key = str(record["key"])
    url = BUCKET + key
    result = {**record, "url": url, "intersects": False, "valid_pixels": 0}
    with fsspec.open(url, "rb", block_size=16 * 1024 * 1024,
                     cache_type="blockcache").open() as stream:
        with h5py.File(stream, "r") as source:
            latitude = source["PRODUCT/latitude"][0]
            longitude = source["PRODUCT/longitude"][0]
            in_domain = (
                (latitude >= BBOX[1]) & (latitude <= BBOX[3])
                & (longitude >= BBOX[0]) & (longitude <= BBOX[2])
            )
            scanlines = np.where(in_domain)[0]
            if not len(scanlines):
                return result
            result["intersects"] = True
            first, last = int(scanlines.min()), int(scanlines.max()) + 1
            lat = latitude[first:last, :]
            lon = longitude[first:last, :]
            qa_dataset = source["PRODUCT/qa_value"]
            qa = qa_dataset[0, first:last, :].astype("float32")
            qa *= float(np.asarray(qa_dataset.attrs.get("scale_factor", 1)).squeeze())
            total = source[TOTAL_COLUMN][0, first:last, :].astype("float64")
            valid = (
                (lat >= BBOX[1]) & (lat <= BBOX[3])
                & (lon >= BBOX[0]) & (lon <= BBOX[2])
                & (qa > 0.75) & np.isfinite(total) & (np.abs(total) < 1e30)
            )
            result["valid_pixels"] = int(valid.sum())
            result["longitude"] = lon[valid].astype("float32")
            result["latitude"] = lat[valid].astype("float32")
            result["total_column"] = total[valid].astype("float32")
            result["scanline_range"] = [first, last - 1]
            return result


def read_orbit_with_retry(record: dict[str, object]) -> dict[str, object]:
    last_error = None
    for attempt in range(1, 4):
        try:
            return read_orbit(record)
        except (OSError, requests.RequestException) as error:
            last_error = error
            time.sleep(attempt)
    return {
        **record, "url": BUCKET + str(record["key"]), "intersects": False,
        "valid_pixels": 0, "read_error": f"{type(last_error).__name__}: {last_error}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--month", type=int, default=1, choices=range(1, 13))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tropomi"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalogue = list_month(args.year, args.month)
    candidates = [record for record in catalogue if candidate(record)]
    orbit_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for number, result in enumerate(executor.map(read_orbit_with_retry, candidates), start=1):
            orbit_results.append(result)
            if result["intersects"]:
                print(f"selected {Path(str(result['key'])).name}: "
                      f"{result['valid_pixels']:,} QA-valid pixels", flush=True)
            elif number % 10 == 0:
                print(f"screened {number}/{len(candidates)} candidate orbits", flush=True)

    selected = [item for item in orbit_results if item["intersects"]]
    if not selected or not sum(int(item["valid_pixels"]) for item in selected):
        raise RuntimeError("No QA-valid TROPOMI pixels intersect the training domain")

    lon_edges = np.arange(BBOX[0], BBOX[2] + RESOLUTION_DEGREES, RESOLUTION_DEGREES)
    lat_edges = np.arange(BBOX[1], BBOX[3] + RESOLUTION_DEGREES, RESOLUTION_DEGREES)
    total_sum = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype="float64")
    count = np.zeros_like(total_sum, dtype="uint16")
    for result in selected:
        lon_index = np.clip(np.searchsorted(lon_edges, result.pop("longitude"), side="right") - 1,
                            0, len(lon_edges) - 2)
        lat_index = np.clip(np.searchsorted(lat_edges, result.pop("latitude"), side="right") - 1,
                            0, len(lat_edges) - 2)
        values = result.pop("total_column")
        np.add.at(total_sum, (lat_index, lon_index), values)
        np.add.at(count, (lat_index, lon_index), 1)
    observed = np.divide(total_sum, count, out=np.full_like(total_sum, np.nan), where=count > 0)
    nearest = distance_transform_edt(~np.isfinite(observed), return_distances=False,
                                     return_indices=True)
    filled = observed[tuple(nearest)].astype("float32")
    dataset = xr.Dataset(
        {
            "CHEM_Sat_no2": (("latitude", "longitude"), filled, {
                "units": "mol m-2",
                "long_name": "Monthly mean TROPOMI total vertical NO2 column",
                "qa_filter": "qa_value > 0.75",
            }),
            "CHEM_Sat_no2_observed": (("latitude", "longitude"), observed.astype("float32"), {
                "units": "mol m-2", "long_name": "Unfilled monthly binned mean",
            }),
            "observation_count": (("latitude", "longitude"), count),
        },
        coords={
            "longitude": (lon_edges[:-1] + lon_edges[1:]) / 2,
            "latitude": (lat_edges[:-1] + lat_edges[1:]) / 2,
        },
        attrs={
            "source": "Copernicus Sentinel-5P TROPOMI RPRO L2 NO2 public MEEO archive",
            "processor_version": "02.04.00",
            "temporal_scope": f"{args.year}-{args.month:02d}",
            "spatial_binning": f"{RESOLUTION_DEGREES} degree mean; nearest fill for empty cells",
            "paper_match": "CHEM_Sat_no2 total column, RPRO v2.4, qa_value > 0.75",
        },
    )
    output = args.output_dir / f"tropomi_total_no2_{args.year}_{args.month:02d}.nc"
    dataset.to_netcdf(output, engine="netcdf4", encoding={
        "CHEM_Sat_no2": {"zlib": True, "complevel": 4, "_FillValue": -9999.0},
        "CHEM_Sat_no2_observed": {"zlib": True, "complevel": 4, "_FillValue": -9999.0},
        "observation_count": {"zlib": True, "complevel": 4},
    })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalogue_objects": len(catalogue), "screened_candidates": len(candidates),
        "selected_orbits": len(selected),
        "read_errors": sum("read_error" in item for item in orbit_results),
        "valid_pixels": sum(int(item["valid_pixels"]) for item in selected),
        "observed_grid_cells": int((count > 0).sum()), "grid_cells": int(count.size),
        "bbox": BBOX, "resolution_degrees": RESOLUTION_DEGREES,
        "output": str(output), "output_bytes": output.stat().st_size,
        "records": [{key: value for key, value in item.items()
                     if key not in {"longitude", "latitude", "total_column"}}
                    for item in orbit_results],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
