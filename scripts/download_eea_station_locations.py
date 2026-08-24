"""Acquire EEA station locations for the cross-border pilot countries."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

QUERY = "https://air.discomap.eea.europa.eu/arcgis/rest/services/AirQuality/AirQualityDownloadServiceEUMonitoringStations/MapServer/0/query"


def main() -> None:
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "CountryCode IN ('BE','DE','FR','LU')",
            "outFields": "AirQualityStation,CountryCode,AirQualityStationEoICode,AQStationName,stationClass",
            "returnGeometry": "true", "outSR": "4326", "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 2000,
        }
        response = requests.get(QUERY, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json().get("features", [])
        features.extend(batch)
        if len(batch) < 2000:
            break
        offset += len(batch)
    collection = {"type": "FeatureCollection", "features": features}
    target = Path("data/raw/eea_air_quality/station_locations.geojson")
    target.write_text(json.dumps(collection), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "source": QUERY, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "countries": ["BE", "DE", "FR", "LU"], "feature_count": len(features),
        "bytes": target.stat().st_size, "sha256": digest,
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
