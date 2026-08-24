"""Readers and target builders for the ratified RIVM Luchtmeetnet archive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .targets import (
    AggregationRules,
    monthly_mean,
    monthly_mean_from_daily,
    monthly_somo35,
)


RIVM_COMPONENT_TO_INDICATOR = {
    "NO2": "no2",
    "PM2.5": "pm2p5",
    "PM10": "pm10",
    "O3": "o3",
    "SO2": "so2",
}


def read_measurements(path: str | Path) -> pd.DataFrame:
    """Read one RIVM annual component file and retain hourly air measurements."""

    frame = pd.read_csv(
        path,
        sep=";",
        comment="#",
        encoding="utf-8",
        low_memory=False,
        usecols=[
            "meetreeks_id",
            "meetlocatie_id",
            "component",
            "matrix",
            "meetduur",
            "eenheid",
            "begindatumtijd",
            "waarde",
        ],
    )
    frame = frame.loc[
        (frame["matrix"] == "lucht") & frame["meetduur"].isin(["uur", "dag"])
    ].copy()
    frame["source_date"] = pd.to_datetime(
        frame["begindatumtijd"].astype(str).str.slice(0, 10), errors="coerce"
    )
    frame["timestamp"] = pd.to_datetime(
        frame.pop("begindatumtijd"), utc=True, errors="coerce"
    )
    frame["waarde"] = pd.to_numeric(frame["waarde"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "waarde", "meetlocatie_id"])

    components = frame["component"].dropna().unique()
    if len(components) != 1 or components[0] not in RIVM_COMPONENT_TO_INDICATOR:
        raise ValueError(f"unsupported or mixed components in {path}: {components}")
    return frame


def read_metadata(
    locations_path: str | Path,
    series_path: str | Path,
) -> pd.DataFrame:
    """Read station coordinates and series classifications."""

    locations = pd.read_csv(
        locations_path,
        sep=";",
        comment="#",
        encoding="utf-8",
        low_memory=False,
    )
    locations = locations.rename(
        columns={
            "breedtegraad": "latitude",
            "lengtegraad": "longitude",
            "hoogte": "altitude_m",
        }
    )
    series = pd.read_csv(
        series_path,
        sep=";",
        comment="#",
        encoding="utf-8",
        low_memory=False,
        usecols=[
            "meetreeks_id",
            "meetlocatie_id",
            "component",
            "matrix",
            "gebiedstype",
            "stationstype",
            "meethoogte",
        ],
    )
    return series.merge(
        locations[
            [
                "meetlocatie_id",
                "meetlocatie_naam",
                "meetlocatie_plaatsnaam",
                "latitude",
                "longitude",
                "altitude_m",
            ]
        ],
        on="meetlocatie_id",
        how="left",
        validate="many_to_one",
    )


def build_monthly_targets(
    measurements: pd.DataFrame,
    metadata: pd.DataFrame,
    rules: AggregationRules = AggregationRules(),
) -> pd.DataFrame:
    """Aggregate one component's hourly measurements for every series and month."""

    component = measurements["component"].iloc[0]
    indicator = RIVM_COMPONENT_TO_INDICATOR[component]
    rows: list[pd.DataFrame] = []

    for series_id, group in measurements.groupby("meetreeks_id", sort=False):
        hourly_group = group.loc[group["meetduur"] == "uur"]
        if not hourly_group.empty:
            source_values = hourly_group.set_index("timestamp")["waarde"]
            monthly = monthly_mean(source_values, rules).reset_index()
            source_resolution = "hourly"
        else:
            daily_group = group.loc[group["meetduur"] == "dag"]
            source_values = daily_group.set_index("source_date")["waarde"]
            monthly = monthly_mean_from_daily(source_values, rules).reset_index()
            source_resolution = "daily"
        monthly.insert(0, "indicator", indicator)
        monthly.insert(0, "meetreeks_id", series_id)
        monthly["source_resolution"] = source_resolution
        rows.append(monthly)

        if component == "O3" and not hourly_group.empty:
            somo35 = monthly_somo35(source_values, rules).reset_index()
            somo35.insert(0, "indicator", "somo35")
            somo35.insert(0, "meetreeks_id", series_id)
            somo35["source_resolution"] = "hourly"
            rows.append(somo35)

    targets = pd.concat(rows, ignore_index=True)
    target_metadata = metadata.loc[
        (metadata["component"] == component) & (metadata["matrix"] == "lucht")
    ].drop_duplicates("meetreeks_id")
    targets = targets.merge(
        target_metadata.drop(columns=["component", "matrix"]),
        on="meetreeks_id",
        how="left",
        validate="many_to_one",
    )
    targets["month"] = targets["month"].astype(str)
    return targets


def coverage_summary(targets: pd.DataFrame) -> pd.DataFrame:
    """Summarize valid station counts and coverage by indicator and month."""

    valid = targets.loc[targets["is_valid"]].copy()
    summary = (
        valid.groupby(["indicator", "month"], as_index=False)
        .agg(
            valid_series=("meetreeks_id", "nunique"),
            valid_stations=("meetlocatie_id", "nunique"),
            median_coverage=("coverage", "median"),
            minimum_coverage=("coverage", "min"),
        )
        .sort_values(["indicator", "month"])
    )
    return summary
