#!/usr/bin/env python3
"""Download historical Space-Track GP records for all 137.05-MHz emitters.

The input is the satellite-date table produced by Section 10.2 of
``5_flux_density_&_EDA2_orbit.ipynb``. Each observing date is expanded by a configurable
margin (14 days by default), overlapping date ranges are merged, and one
GP_HISTORY request is issued for each merged range.  This gives enough orbital
history to estimate whether each satellite was raising, stable, or lowering
around the EDA2 detection date without making one request per satellite.

Space-Track credentials are read from ``SPACETRACK_USER`` and
``SPACETRACK_PASSWORD`` or requested interactively.  They are never saved.
Downloaded responses are cached, so rerunning the script does not contact
Space-Track unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BASE_URL = "https://www.space-track.org"


@dataclass
class QueryGroup:
    """One merged date interval and the satellites required within it."""

    start: pd.Timestamp
    end: pd.Timestamp
    norad_ids: set[int] = field(default_factory=set)
    combination_ids: set[int] = field(default_factory=set)


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_input = (
        project_root
        / "results"
        / "5_flux_density"
        / "chapter5_flux_density"
        / "grigg2025_137p05_satellite_date_combinations.csv"
    )
    default_output = (
        project_root / "data" / "external" / "spacetrack" / "all_137p05_emitters"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Download historical GP/TLE-like element sets around every "
            "137.05-MHz satellite-date combination."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--padding-days",
        type=int,
        default=14,
        help="Days retained before and after each observing date (default: 14)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload queries even when matching cached responses exist",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help="Pause in seconds between Space-Track queries (default: 3)",
    )
    return parser.parse_args()


def read_satellite_dates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input table: {path}\n"
            "Run Sections 10.1 and 10.2 of 5_flux_density_&_EDA2_orbit.ipynb first."
        )

    combinations = pd.read_csv(path)
    required = {"norad_id", "observing_date"}
    missing = required.difference(combinations.columns)
    if missing:
        raise ValueError(f"Input table lacks columns: {sorted(missing)}")

    combinations = combinations.copy()
    combinations["norad_id"] = combinations["norad_id"].astype(int)
    combinations["observing_date"] = pd.to_datetime(
        combinations["observing_date"], utc=True
    ).dt.normalize()
    if "combination_id" not in combinations.columns:
        combinations.insert(0, "combination_id", range(1, len(combinations) + 1))
    combinations["combination_id"] = combinations["combination_id"].astype(int)
    return combinations.sort_values(["observing_date", "norad_id"]).reset_index(
        drop=True
    )


def make_query_groups(
    combinations: pd.DataFrame, padding_days: int
) -> list[QueryGroup]:
    """Merge overlapping padded observing-date intervals."""

    if padding_days < 0:
        raise ValueError("--padding-days must be non-negative")

    margin = pd.Timedelta(days=padding_days)
    one_day = pd.Timedelta(days=1)
    intervals: list[QueryGroup] = []
    for row in combinations.itertuples(index=False):
        # The upper bound is the start of the day after the final padded date,
        # ensuring that the complete final UTC day is included.
        intervals.append(
            QueryGroup(
                start=row.observing_date - margin,
                end=row.observing_date + margin + one_day,
                norad_ids={int(row.norad_id)},
                combination_ids={int(row.combination_id)},
            )
        )

    intervals.sort(key=lambda item: (item.start, item.end))
    merged: list[QueryGroup] = []
    for interval in intervals:
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
            continue
        current = merged[-1]
        current.end = max(current.end, interval.end)
        current.norad_ids.update(interval.norad_ids)
        current.combination_ids.update(interval.combination_ids)
    return merged


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


def build_query_url(group: QueryGroup) -> str:
    ids = ",".join(str(value) for value in sorted(group.norad_ids))
    epoch_range = f"{group.start:%Y-%m-%d}--{group.end:%Y-%m-%d}"
    return (
        f"{BASE_URL}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{ids}/EPOCH/{quote(epoch_range, safe='-')}/"
        "orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/json"
    )


def group_stem(group_number: int, group: QueryGroup) -> str:
    ids = ",".join(str(value) for value in sorted(group.norad_ids))
    identity = hashlib.sha256(ids.encode("ascii")).hexdigest()[:10]
    return (
        f"group_{group_number:02d}_{group.start:%Y%m%d}_{group.end:%Y%m%d}_"
        f"n{len(group.norad_ids)}_{identity}"
    )


def load_cached_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Cached response is not a JSON list: {path}")
    return records


def download_records(session: requests.Session, group: QueryGroup) -> list[dict]:
    response = session.get(build_query_url(group), timeout=300)
    response.raise_for_status()
    try:
        records = response.json()
    except requests.JSONDecodeError as error:
        preview = response.text[:300].replace("\n", " ")
        raise RuntimeError(
            "Space-Track did not return JSON; login may have failed. "
            f"Response begins: {preview!r}"
        ) from error
    if not isinstance(records, list):
        raise RuntimeError(f"Unexpected Space-Track response: {records!r}")
    return records


def normalise_history(records: list[dict]) -> pd.DataFrame:
    history = pd.DataFrame.from_records(records)
    if history.empty:
        return history
    if "NORAD_CAT_ID" in history.columns:
        history["NORAD_CAT_ID"] = pd.to_numeric(
            history["NORAD_CAT_ID"], errors="raise"
        ).astype(int)
    for column in ("EPOCH", "CREATION_DATE"):
        if column in history.columns:
            history[column] = pd.to_datetime(history[column], utc=True, errors="coerce")
    sort_columns = [
        column
        for column in ("NORAD_CAT_ID", "EPOCH", "CREATION_DATE")
        if column in history.columns
    ]
    return history.sort_values(sort_columns).reset_index(drop=True)


def make_unique_epoch_table(history: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest release when GP_HISTORY repeats an object epoch."""

    required = {"NORAD_CAT_ID", "EPOCH"}
    if history.empty or not required.issubset(history.columns):
        return history.copy()
    sort_columns = ["NORAD_CAT_ID", "EPOCH"]
    if "CREATION_DATE" in history.columns:
        sort_columns.append("CREATION_DATE")
    return (
        history.sort_values(sort_columns)
        .drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="last")
        .reset_index(drop=True)
    )


def make_coverage_table(
    combinations: pd.DataFrame, history: pd.DataFrame, padding_days: int
) -> pd.DataFrame:
    rows = []
    for row in combinations.itertuples(index=False):
        start = row.observing_date - pd.Timedelta(days=padding_days)
        end = row.observing_date + pd.Timedelta(days=padding_days + 1)
        if history.empty or not {"NORAD_CAT_ID", "EPOCH"}.issubset(history.columns):
            selected = history.iloc[0:0]
        else:
            selected = history[
                history["NORAD_CAT_ID"].eq(row.norad_id)
                & history["EPOCH"].ge(start)
                & history["EPOCH"].lt(end)
            ]
        rows.append(
            {
                "combination_id": row.combination_id,
                "norad_id": row.norad_id,
                "observing_date": row.observing_date.date().isoformat(),
                "query_start_utc": start,
                "query_end_utc_exclusive": end,
                "number_of_unique_gp_epochs": len(selected),
                "first_gp_epoch": selected["EPOCH"].min() if not selected.empty else pd.NaT,
                "last_gp_epoch": selected["EPOCH"].max() if not selected.empty else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    combinations = read_satellite_dates(args.input)
    groups = make_query_groups(combinations, args.padding_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    for number, group in enumerate(groups, start=1):
        plan_rows.append(
            {
                "query_group": number,
                "query_start_utc": group.start,
                "query_end_utc": group.end,
                "number_of_norad_ids": len(group.norad_ids),
                "norad_ids": ",".join(map(str, sorted(group.norad_ids))),
                "combination_ids": ",".join(map(str, sorted(group.combination_ids))),
            }
        )
    plan = pd.DataFrame(plan_rows)
    plan_output = args.output_dir / "spacetrack_query_plan.csv"
    plan.to_csv(plan_output, index=False)

    print(f"Input satellite-date combinations: {len(combinations)}")
    print(f"Unique NORAD objects: {combinations['norad_id'].nunique()}")
    print(f"Merged Space-Track queries: {len(groups)}")
    print(plan.to_string(index=False))

    pending = []
    cached_records: list[dict] = []
    for number, group in enumerate(groups, start=1):
        raw_output = args.output_dir / f"{group_stem(number, group)}.json"
        if raw_output.exists() and not args.force:
            records = load_cached_records(raw_output)
            cached_records.extend(records)
            print(f"Using cached group {number}: {raw_output} ({len(records)} records)")
        else:
            pending.append((number, group, raw_output))

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

            for index, (number, group, raw_output) in enumerate(pending):
                print(
                    f"Downloading group {number}/{len(groups)}: "
                    f"{group.start:%Y-%m-%d} to {group.end:%Y-%m-%d}, "
                    f"{len(group.norad_ids)} NORAD objects"
                )
                records = download_records(session, group)
                raw_output.write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                downloaded_records.extend(records)
                print(f"  Saved {len(records)} records to {raw_output}")
                if index + 1 < len(pending):
                    time.sleep(max(args.request_delay, 0.0))

    history = normalise_history(cached_records + downloaded_records)
    if history.empty:
        raise RuntimeError("No GP_HISTORY records were returned for any query group")

    all_output = args.output_dir / "grigg2025_all_137p05_gp_history.csv"
    history.to_csv(all_output, index=False)

    unique_history = make_unique_epoch_table(history)
    unique_output = args.output_dir / "grigg2025_all_137p05_unique_gp_epochs.csv"
    unique_history.to_csv(unique_output, index=False)

    coverage = make_coverage_table(
        combinations, unique_history, args.padding_days
    )
    coverage_output = args.output_dir / "grigg2025_all_137p05_gp_coverage.csv"
    coverage.to_csv(coverage_output, index=False)

    missing = coverage["number_of_unique_gp_epochs"].eq(0)
    print(f"\nDownloaded/cached GP_HISTORY rows: {len(history)}")
    print(f"Unique NORAD+EPOCH rows: {len(unique_history)}")
    print(
        "Satellite-date combinations with no matching GP epoch: "
        f"{int(missing.sum())}"
    )
    if missing.any():
        print(coverage.loc[missing, ["norad_id", "observing_date"]].to_string(index=False))
    print("Saved query plan:", plan_output)
    print("Saved all returned records:", all_output)
    print("Saved analysis-ready unique epochs:", unique_output)
    print("Saved coverage check:", coverage_output)


if __name__ == "__main__":
    main()
