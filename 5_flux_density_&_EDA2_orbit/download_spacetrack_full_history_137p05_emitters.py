#!/usr/bin/env python3
"""Download long-term Space-Track GP histories for all 137.05-MHz emitters.

The input is the emission-segment table produced by Section 10.1 of
``5_flux_density_&_EDA2_orbit.ipynb``. For each unique NORAD object, the query starts
90 days before its first detected 137.05-MHz emission and ends on the current
UTC date (or on a user-supplied end date).  One query is made per satellite so
that downloads can be cached and resumed independently.

Space-Track credentials are read from ``SPACETRACK_USER`` and
``SPACETRACK_PASSWORD`` or requested interactively.  They are never saved.
Existing raw responses are reused unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BASE_URL = "https://www.space-track.org"


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_input = (
        project_root
        / "results"
        / "5_flux_density"
        / "chapter5_flux_density"
        / "grigg2025_137p05_emission_segments.csv"
    )
    default_output = (
        project_root
        / "data"
        / "external"
        / "spacetrack"
        / "full_137p05_emitters"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Download the GP_HISTORY of each 137.05-MHz emitter from a "
            "fixed interval before its first detection through the present."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--pre-detection-days",
        type=int,
        default=90,
        help="Days included before each satellite's first detection (default: 90)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "Last UTC calendar date to include, in YYYY-MM-DD form "
            "(default: current UTC date)"
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help="Pause in seconds between Space-Track queries (default: 3)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum attempts for a failed query (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload responses even when matching cached files exist",
    )
    return parser.parse_args()


def read_emission_segments(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing emission-segment table: {path}\n"
            "Run Section 10.1 of 5_flux_density_&_EDA2_orbit.ipynb first."
        )

    segments = pd.read_csv(path)
    required = {"norad_id", "segment_start_utc"}
    missing = required.difference(segments.columns)
    if missing:
        raise ValueError(f"Input table lacks columns: {sorted(missing)}")

    segments = segments.copy()
    segments["norad_id"] = pd.to_numeric(
        segments["norad_id"], errors="raise"
    ).astype(int)
    segments["segment_start_utc"] = pd.to_datetime(
        segments["segment_start_utc"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    return segments.sort_values(["norad_id", "segment_start_utc"]).reset_index(
        drop=True
    )


def parse_end_date(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").normalize()
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.normalize()


def make_query_plan(
    segments: pd.DataFrame,
    pre_detection_days: int,
    last_included_date: pd.Timestamp,
) -> pd.DataFrame:
    if pre_detection_days < 0:
        raise ValueError("--pre-detection-days must be non-negative")

    first_detections = (
        segments.groupby("norad_id", as_index=False)
        .agg(
            first_detection_utc=("segment_start_utc", "min"),
            last_detection_utc=("segment_start_utc", "max"),
            number_of_emission_segments=("segment_start_utc", "size"),
        )
        .sort_values("norad_id")
        .reset_index(drop=True)
    )
    first_detections["query_start_utc"] = (
        first_detections["first_detection_utc"].dt.normalize()
        - pd.Timedelta(days=pre_detection_days)
    )
    # Space-Track range filters include calendar dates.  Use the following
    # midnight as the upper bound so the requested final UTC day is complete.
    first_detections["query_end_utc_exclusive"] = (
        last_included_date + pd.Timedelta(days=1)
    )
    invalid = first_detections["query_start_utc"].ge(
        first_detections["query_end_utc_exclusive"]
    )
    if invalid.any():
        raise ValueError(
            "The requested end date precedes at least one query start:\n"
            + first_detections.loc[
                invalid, ["norad_id", "query_start_utc"]
            ].to_string(index=False)
        )
    return first_detections


def get_credentials() -> tuple[str, str]:
    username = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not username:
        username = input("Space-Track username: ").strip()
    if not password:
        password = getpass.getpass("Space-Track password: ")
    if not username or not password:
        raise RuntimeError("Space-Track username and password are required")
    return username, password


def build_query_url(row: object) -> str:
    epoch_range = (
        f"{row.query_start_utc:%Y-%m-%d}--"
        f"{row.query_end_utc_exclusive:%Y-%m-%d}"
    )
    return (
        f"{BASE_URL}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{int(row.norad_id)}/"
        f"EPOCH/{quote(epoch_range, safe='-')}/"
        "orderby/EPOCH%20asc,CREATION_DATE%20asc/format/json"
    )


def raw_output_path(output_dir: Path, row: object) -> Path:
    return output_dir / (
        f"norad_{int(row.norad_id)}_"
        f"{row.query_start_utc:%Y%m%d}_"
        f"{row.query_end_utc_exclusive:%Y%m%d}.json"
    )


def load_cached_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Cached response is not a JSON list: {path}")
    return records


def download_records(
    session: requests.Session,
    row: object,
    max_attempts: int,
) -> list[dict]:
    if max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")

    url = build_query_url(row)
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=300)
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list):
                raise RuntimeError(
                    f"Unexpected Space-Track response for NORAD {row.norad_id}: "
                    f"{records!r}"
                )
            return records
        except (requests.RequestException, requests.JSONDecodeError) as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Space-Track query failed for NORAD {row.norad_id} "
                    f"after {max_attempts} attempts"
                ) from error
            wait_seconds = min(60.0, 2.0 ** (attempt - 1))
            print(
                f"  Attempt {attempt}/{max_attempts} failed: {error}; "
                f"retrying in {wait_seconds:.0f} s"
            )
            time.sleep(wait_seconds)

    raise AssertionError("unreachable")


def normalise_history(records: list[dict]) -> pd.DataFrame:
    history = pd.DataFrame.from_records(records)
    if history.empty:
        return history
    history["NORAD_CAT_ID"] = pd.to_numeric(
        history["NORAD_CAT_ID"], errors="raise"
    ).astype(int)
    for column in ("EPOCH", "CREATION_DATE"):
        if column in history.columns:
            history[column] = pd.to_datetime(
                history[column], utc=True, errors="coerce"
            )
    sort_columns = [
        column
        for column in ("NORAD_CAT_ID", "EPOCH", "CREATION_DATE")
        if column in history.columns
    ]
    return history.sort_values(sort_columns).reset_index(drop=True)


def make_unique_epoch_table(history: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest publication when an object epoch is repeated."""

    if history.empty:
        return history.copy()
    required = {"NORAD_CAT_ID", "EPOCH"}
    if not required.issubset(history.columns):
        raise ValueError(f"Downloaded history lacks columns: {sorted(required)}")
    sort_columns = ["NORAD_CAT_ID", "EPOCH"]
    if "CREATION_DATE" in history.columns:
        sort_columns.append("CREATION_DATE")
    return (
        history.sort_values(sort_columns)
        .drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="last")
        .reset_index(drop=True)
    )


def make_coverage_table(plan: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in plan.itertuples(index=False):
        selected = history[history["NORAD_CAT_ID"].eq(row.norad_id)]
        first_epoch = selected["EPOCH"].min() if not selected.empty else pd.NaT
        last_epoch = selected["EPOCH"].max() if not selected.empty else pd.NaT
        rows.append(
            {
                "norad_id": row.norad_id,
                "first_detection_utc": row.first_detection_utc,
                "last_detection_utc": row.last_detection_utc,
                "number_of_emission_segments": row.number_of_emission_segments,
                "query_start_utc": row.query_start_utc,
                "query_end_utc_exclusive": row.query_end_utc_exclusive,
                "number_of_unique_gp_epochs": len(selected),
                "first_gp_epoch": first_epoch,
                "last_gp_epoch": last_epoch,
                "has_pre_detection_gp": bool(
                    not selected.empty
                    and (selected["EPOCH"] < row.first_detection_utc).any()
                ),
                "has_post_detection_gp": bool(
                    not selected.empty
                    and (selected["EPOCH"] > row.last_detection_utc).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    segments = read_emission_segments(args.input)
    last_included_date = parse_end_date(args.end_date)
    plan = make_query_plan(
        segments, args.pre_detection_days, last_included_date
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan_output = args.output_dir / "spacetrack_full_history_query_plan.csv"
    plan.to_csv(plan_output, index=False)

    print(f"Emission segments: {len(segments)}")
    print(f"Unique NORAD objects: {len(plan)}")
    print(f"Last included UTC date: {last_included_date.date()}")
    print("Query plan:")
    print(
        plan[
            [
                "norad_id",
                "first_detection_utc",
                "query_start_utc",
                "query_end_utc_exclusive",
            ]
        ].to_string(index=False)
    )

    cached_records: list[dict] = []
    pending: list[tuple[object, Path]] = []
    for row in plan.itertuples(index=False):
        output = raw_output_path(args.output_dir, row)
        if output.exists() and not args.force:
            records = load_cached_records(output)
            cached_records.extend(records)
            print(
                f"Using cached NORAD {row.norad_id}: "
                f"{output.name} ({len(records)} records)"
            )
        else:
            pending.append((row, output))

    downloaded_records: list[dict] = []
    if pending:
        username, password = get_credentials()
        with requests.Session() as session:
            login = session.post(
                f"{BASE_URL}/ajaxauth/login",
                data={"identity": username, "password": password},
                timeout=60,
            )
            login.raise_for_status()

            for index, (row, output) in enumerate(pending, start=1):
                print(
                    f"Downloading {index}/{len(pending)}: NORAD "
                    f"{row.norad_id}, {row.query_start_utc:%Y-%m-%d} to "
                    f"{row.query_end_utc_exclusive:%Y-%m-%d}"
                )
                records = download_records(session, row, args.max_attempts)
                output.write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                downloaded_records.extend(records)
                print(f"  Saved {len(records)} records: {output}")
                if index < len(pending):
                    time.sleep(max(args.request_delay, 0.0))

    history = normalise_history(cached_records + downloaded_records)
    if history.empty:
        raise RuntimeError("No GP_HISTORY records were returned")

    combined_output = args.output_dir / "grigg2025_full_137p05_gp_history.csv"
    history.to_csv(combined_output, index=False)

    unique_history = make_unique_epoch_table(history)
    unique_output = (
        args.output_dir / "grigg2025_full_137p05_unique_gp_epochs.csv"
    )
    unique_history.to_csv(unique_output, index=False)

    coverage = make_coverage_table(plan, unique_history)
    coverage_output = args.output_dir / "spacetrack_full_history_coverage.csv"
    coverage.to_csv(coverage_output, index=False)

    missing = coverage["number_of_unique_gp_epochs"].eq(0)
    print(f"\nDownloaded/cached GP_HISTORY rows: {len(history)}")
    print(f"Unique NORAD+EPOCH rows: {len(unique_history)}")
    print(f"Satellites without any returned GP records: {int(missing.sum())}")
    print("Saved query plan:", plan_output)
    print("Saved combined history:", combined_output)
    print("Saved analysis-ready unique epochs:", unique_output)
    print("Saved coverage table:", coverage_output)


if __name__ == "__main__":
    main()
