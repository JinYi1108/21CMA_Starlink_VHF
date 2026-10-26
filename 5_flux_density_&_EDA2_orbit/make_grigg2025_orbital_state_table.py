#!/usr/bin/env python3
"""Create a compact orbital-state table for the 137.05-MHz emitters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PHASE_LABELS = {
    "orbit raising": "升轨",
    "stable orbital phase": "稳定轨道阶段",
    "orbit lowering": "降轨",
    "temporary plateau": "暂时高度平台",
    "transitional/uncertain": "过渡或不确定",
    "uncertain": "不确定",
}

COVERAGE_LABELS = {
    "two-sided": "事件前后双侧覆盖",
    "limited: post-only": "仅事件后覆盖",
    "limited: pre-only": "仅事件前覆盖",
    "limited: reference-day only": "仅事件当日",
    "limited: no data": "无可用数据",
}


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_dir = (
        project_root / "results" / "5_flux_density" / "chapter5_flux_density"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Extract satellite name, 14-day mean-altitude trend, and orbital "
            "state from the full-history classification table."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "grigg2025_137p05_orbital_phase_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "grigg2025_137p05_orbital_state_table.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    source = pd.read_csv(args.input)
    required = {
        "norad_id",
        "object_name",
        "observing_date",
        "event_mean_altitude_km",
        "trend_14d_km_per_day",
        "preliminary_orbital_phase",
        "classification_coverage",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Input table lacks columns: {sorted(missing)}")

    table = source[
        [
            "object_name",
            "norad_id",
            "observing_date",
            "event_mean_altitude_km",
            "trend_14d_km_per_day",
            "preliminary_orbital_phase",
            "classification_coverage",
        ]
    ].copy()
    table = table.rename(
        columns={
            "object_name": "satellite_name",
            "observing_date": "detection_date_utc",
            "event_mean_altitude_km": "mean_altitude_at_detection_km",
            "trend_14d_km_per_day": "altitude_change_rate_km_per_day",
            "preliminary_orbital_phase": "orbital_state",
            "classification_coverage": "tle_coverage",
        }
    )
    table["orbital_state_zh"] = table["orbital_state"].map(PHASE_LABELS)
    table["tle_coverage_zh"] = table["tle_coverage"].map(COVERAGE_LABELS)
    table["mean_altitude_at_detection_km"] = table[
        "mean_altitude_at_detection_km"
    ].round(3)
    table["altitude_change_rate_km_per_day"] = table[
        "altitude_change_rate_km_per_day"
    ].round(4)
    table = table[
        [
            "satellite_name",
            "norad_id",
            "detection_date_utc",
            "mean_altitude_at_detection_km",
            "altitude_change_rate_km_per_day",
            "orbital_state",
            "orbital_state_zh",
            "tle_coverage",
            "tle_coverage_zh",
        ]
    ].sort_values(["detection_date_utc", "norad_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(
        table[
            [
                "satellite_name",
                "norad_id",
                "detection_date_utc",
                "mean_altitude_at_detection_km",
                "altitude_change_rate_km_per_day",
                "orbital_state",
            ]
        ].to_string(index=False)
    )
    print("\nThe daily rate is the Theil--Sen slope over the +/-14-day window.")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
