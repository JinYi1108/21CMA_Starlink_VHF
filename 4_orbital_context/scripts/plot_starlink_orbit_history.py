"""Plot the complete TLE-derived altitude history of STARLINK-32838.

Functions and outputs
---------------------
``parse_tle_history(input_path)``
    Reads a 2LE/3LE history file, validates NORAD 62827 records, removes
    duplicate epochs, and returns chronological perigee/apogee/mean heights.
``write_history_csv(rows, output_path)``
    Writes one row per unique TLE epoch with orbital heights and TLE fields.
``write_summary_json(rows, output_path)``
    Writes record counts, extrema, the April VHF-event context, and major
    altitude milestones.
``save_history_plot(rows, output_path)``
    Writes a two-panel PNG: the complete history and a launch-to-June zoom.

Default outputs are written below
``results/orbit_history/`` next to this script.  Heights are derived from the
TLE mean elements with the WGS-72 Earth radius used by SGP4.  They describe
the mean orbit's perigee/apogee, not a precise instantaneous geodetic height.
The April VHF payload point is plotted separately as an instantaneous WGS-84
height and therefore should not be compared kilometre-for-kilometre with the
mean-element curve.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec


NORAD_ID = 62827
VHF_EVENT_UTC = datetime(2025, 4, 15, 14, 0, 0, 250000, tzinfo=timezone.utc)
VHF_PAYLOAD_GEODETIC_ALTITUDE_KM = 323.58690316224056
TLE_GAP_START_UTC = datetime(2025, 4, 13, 18, 0, 2, 160, tzinfo=timezone.utc)
TLE_GAP_STOP_UTC = datetime(2025, 4, 17, 21, 38, 9, 404736, tzinfo=timezone.utc)
CATALOGUED_DECAY_UTC = datetime(2026, 1, 5, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OrbitRow:
    """One unique TLE epoch and its mean-element orbital heights."""

    epoch_utc: datetime
    perigee_km: float
    apogee_km: float
    mean_altitude_km: float
    eccentricity: float
    mean_motion_rev_per_day: float
    bstar: float
    name: str
    tle_line1: str
    tle_line2: str


def _tle_epoch_utc(line1: str) -> datetime:
    """Convert the fixed-width TLE epoch field to a timezone-aware UTC time."""

    year_two_digits = int(line1[18:20])
    year = 1900 + year_two_digits if year_two_digits >= 57 else 2000 + year_two_digits
    day_of_year = float(line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def parse_tle_history(input_path: str | Path) -> list[OrbitRow]:
    """Parse and de-duplicate NORAD 62827 records from a 2LE or 3LE file."""

    lines = Path(input_path).read_text(encoding="ascii").splitlines()
    parsed: list[OrbitRow] = []
    current_name = "UNKNOWN"

    index = 0
    while index < len(lines):
        line = lines[index].strip("\r")
        if line.startswith("0 "):
            current_name = line[2:].strip()
            index += 1
            continue
        if not line.startswith("1 "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("2 "):
            raise ValueError(f"TLE line 1 at file line {index + 1} has no following line 2")

        line1 = line
        line2 = lines[index + 1].strip("\r")
        satellite = Satrec.twoline2rv(line1, line2)
        if satellite.satnum != NORAD_ID:
            raise ValueError(
                f"Unexpected NORAD ID {satellite.satnum}; expected only {NORAD_ID}"
            )

        earth_radius_km = satellite.radiusearthkm
        perigee_km = satellite.altp * earth_radius_km
        apogee_km = satellite.alta * earth_radius_km
        mean_motion_rev_per_day = satellite.no_kozai * 1440.0 / (2.0 * np.pi)
        parsed.append(
            OrbitRow(
                epoch_utc=_tle_epoch_utc(line1),
                perigee_km=perigee_km,
                apogee_km=apogee_km,
                mean_altitude_km=(perigee_km + apogee_km) / 2.0,
                eccentricity=satellite.ecco,
                mean_motion_rev_per_day=mean_motion_rev_per_day,
                bstar=satellite.bstar,
                name=current_name,
                tle_line1=line1,
                tle_line2=line2,
            )
        )
        index += 2

    if not parsed:
        raise ValueError(f"No NORAD {NORAD_ID} TLE records found in {input_path}")

    # Space-Track GP_HISTORY can repeat a physical solution with a different
    # name line or revolution-count suffix.  Collapse equal epochs; the few
    # differing same-epoch solutions in this file agree at sub-metre scale.
    by_epoch: dict[datetime, list[OrbitRow]] = defaultdict(list)
    for row in parsed:
        by_epoch[row.epoch_utc].append(row)

    unique_rows: list[OrbitRow] = []
    for epoch, candidates in by_epoch.items():
        candidates.sort(key=lambda item: item.mean_altitude_km)
        unique_rows.append(candidates[len(candidates) // 2])
    return sorted(unique_rows, key=lambda item: item.epoch_utc)


def write_history_csv(rows: list[OrbitRow], output_path: str | Path) -> None:
    """Write the de-duplicated orbit-height history as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch_utc",
        "perigee_km",
        "apogee_km",
        "mean_altitude_km",
        "eccentricity",
        "mean_motion_rev_per_day",
        "bstar",
        "name",
        "tle_line1",
        "tle_line2",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["epoch_utc"] = row.epoch_utc.isoformat().replace("+00:00", "Z")
            writer.writerow(values)


def _first_at_or_above(rows: list[OrbitRow], threshold_km: float) -> OrbitRow | None:
    """Return the first post-event row at or above an altitude threshold."""

    return next(
        (
            row
            for row in rows
            if row.epoch_utc > VHF_EVENT_UTC and row.mean_altitude_km >= threshold_km
        ),
        None,
    )


def write_summary_json(rows: list[OrbitRow], output_path: str | Path) -> None:
    """Write extrema and event-related orbit milestones as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minimum = min(rows, key=lambda item: item.mean_altitude_km)
    maximum = max(rows, key=lambda item: item.mean_altitude_km)
    before_event = max(
        (row for row in rows if row.epoch_utc < VHF_EVENT_UTC),
        key=lambda item: item.epoch_utc,
    )
    after_event = min(
        (row for row in rows if row.epoch_utc > VHF_EVENT_UTC),
        key=lambda item: item.epoch_utc,
    )

    milestones: dict[str, object] = {}
    for threshold in (330.0, 440.0, 470.0):
        row = _first_at_or_above(rows, threshold)
        milestones[f"first_post_event_at_or_above_{int(threshold)}_km"] = (
            None
            if row is None
            else {
                "epoch_utc": row.epoch_utc.isoformat().replace("+00:00", "Z"),
                "mean_altitude_km": row.mean_altitude_km,
            }
        )

    summary = {
        "norad_id": NORAD_ID,
        "unique_tle_epoch_count": len(rows),
        "coverage_start_utc": rows[0].epoch_utc.isoformat().replace("+00:00", "Z"),
        "coverage_stop_utc": rows[-1].epoch_utc.isoformat().replace("+00:00", "Z"),
        "minimum_mean_altitude": {
            "epoch_utc": minimum.epoch_utc.isoformat().replace("+00:00", "Z"),
            "altitude_km": minimum.mean_altitude_km,
        },
        "maximum_mean_altitude": {
            "epoch_utc": maximum.epoch_utc.isoformat().replace("+00:00", "Z"),
            "altitude_km": maximum.mean_altitude_km,
        },
        "vhf_event": {
            "epoch_utc": VHF_EVENT_UTC.isoformat().replace("+00:00", "Z"),
            "payload_instantaneous_geodetic_altitude_km": VHF_PAYLOAD_GEODETIC_ALTITUDE_KM,
            "last_tle_before": {
                "epoch_utc": before_event.epoch_utc.isoformat().replace("+00:00", "Z"),
                "mean_altitude_km": before_event.mean_altitude_km,
            },
            "first_tle_after": {
                "epoch_utc": after_event.epoch_utc.isoformat().replace("+00:00", "Z"),
                "mean_altitude_km": after_event.mean_altitude_km,
            },
            "public_tle_gap_hours": (
                after_event.epoch_utc - before_event.epoch_utc
            ).total_seconds()
            / 3600.0,
        },
        "milestones": milestones,
        "interpretation_caution": (
            "TLE mean-element heights show orbital evolution but do not by themselves "
            "identify maneuver purpose, spacecraft health, or controlled versus passive decay."
        ),
    }
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _plot_altitude_panel(
    axis: plt.Axes,
    rows: list[OrbitRow],
    *,
    x_start: datetime,
    x_stop: datetime,
    title: str,
    shade_tle_gap: bool,
) -> None:
    """Draw one perigee/mean/apogee altitude panel."""

    times = np.array([row.epoch_utc for row in rows], dtype=object)
    perigees = np.array([row.perigee_km for row in rows])
    apogees = np.array([row.apogee_km for row in rows])
    means = np.array([row.mean_altitude_km for row in rows])

    axis.fill_between(
        times,
        perigees,
        apogees,
        color="#56B4E9",
        alpha=0.28,
        label="Perigee-apogee envelope",
    )
    axis.plot(
        times,
        means,
        color="#0072B2",
        linewidth=1.35,
        marker=".",
        markersize=2.2,
        label="Mean altitude / TLE epochs",
    )
    axis.axvline(
        VHF_EVENT_UTC,
        color="#D55E00",
        linestyle="--",
        linewidth=1.25,
        label="21CMA VHF event",
    )
    axis.scatter(
        [VHF_EVENT_UTC],
        [VHF_PAYLOAD_GEODETIC_ALTITUDE_KM],
        marker="*",
        s=100,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.45,
        zorder=5,
        label="Payload instantaneous altitude",
    )
    if shade_tle_gap:
        axis.axvspan(
            TLE_GAP_START_UTC,
            TLE_GAP_STOP_UTC,
            color="#E69F00",
            alpha=0.16,
            label="Public TLE gap",
        )
    if x_start <= CATALOGUED_DECAY_UTC <= x_stop:
        axis.axvline(
            CATALOGUED_DECAY_UTC,
            color="#CC79A7",
            linestyle=":",
            linewidth=1.4,
            label="Catalogued decay date",
        )

    axis.set_xlim(x_start, x_stop)
    axis.set_ylabel("Altitude (km)")
    axis.set_title(title, loc="left", fontsize=11)
    axis.grid(True, alpha=0.25, linewidth=0.7)


def save_history_plot(rows: list[OrbitRow], output_path: str | Path) -> None:
    """Write the complete and early-orbit altitude history PNG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    _plot_altitude_panel(
        axes[0],
        rows,
        x_start=rows[0].epoch_utc - timedelta(days=3),
        x_stop=CATALOGUED_DECAY_UTC + timedelta(days=4),
        title="Complete TLE-derived orbit history",
        shade_tle_gap=True,
    )
    axes[0].set_ylim(210, 500)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    _plot_altitude_panel(
        axes[1],
        rows,
        x_start=datetime(2025, 1, 24, tzinfo=timezone.utc),
        x_stop=datetime(2025, 6, 10, tzinfo=timezone.utc),
        title="Launch and orbit-raising phase (zoom)",
        shade_tle_gap=True,
    )
    axes[1].set_ylim(260, 465)
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_xlabel("UTC date")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "STARLINK-32838 (NORAD 62827): altitude evolution and 21CMA VHF event",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.07, right=0.99, hspace=0.18)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    analysis_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot STARLINK-32838 altitude history from a Space-Track 2LE/3LE file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=analysis_dir / "STARLINK-32838_full_history.tle",
        help="Input Space-Track GP_HISTORY file in 2LE or 3LE format.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=analysis_dir / "results" / "orbit_history",
        help="Directory for the PNG, CSV, and JSON outputs.",
    )
    return parser


def main() -> None:
    """Parse the TLE history and write all standard orbit-history products."""

    args = build_parser().parse_args()
    rows = parse_tle_history(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "starlink_32838_orbit_history.csv"
    json_path = args.output_dir / "starlink_32838_orbit_history_summary.json"
    png_path = args.output_dir / "starlink_32838_orbit_history.png"
    write_history_csv(rows, csv_path)
    write_summary_json(rows, json_path)
    save_history_plot(rows, png_path)
    print(f"Unique TLE epochs: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
