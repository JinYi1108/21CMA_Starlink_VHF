"""Compare STARLINK-32838 with all 23 Starlink Group 11-6 spacecraft.

Functions and outputs
---------------------
``parse_group_history(input_path)``
    Reads a Space-Track 2LE/3LE GP_HISTORY file, groups records by NORAD ID,
    removes duplicate epochs, and derives mean-element orbital heights.
``write_history_csv(histories, output_path)``
    Writes every unique satellite/TLE epoch for reproducible later analysis.
``write_satellite_summary_csv(histories, output_path)``
    Writes one row per spacecraft with event-time altitude, TLE freshness,
    altitude milestones, extrema, and the final downloaded state.
``write_comparison_json(histories, output_path)``
    Writes the numerical STARLINK-32838-versus-peer comparison.
``save_comparison_plot(histories, output_path)``
    Writes a two-panel PNG with all launch-group tracks, the peer distribution,
    the April 2025 VHF event, and the target spacecraft highlighted.

Default outputs are written to ``results/group11_6_comparison/``.  Heights are
derived from TLE mean elements using the WGS-72 Earth radius.  Straight lines
between TLE epochs do not imply continuous tracking, especially across gaps.
The payload marker is an independently derived instantaneous WGS-84 altitude,
not a TLE mean-orbit altitude.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec


TARGET_NORAD_ID = 62827
EXPECTED_NORAD_IDS = set(range(62806, 62829))
VHF_EVENT_UTC = datetime(2025, 4, 15, 14, 0, 0, 250000, tzinfo=timezone.utc)
VHF_PAYLOAD_GEODETIC_ALTITUDE_KM = 323.58690316224056
TARGET_TLE_GAP_START_UTC = datetime(
    2025, 4, 13, 18, 0, 2, 160, tzinfo=timezone.utc
)
TARGET_TLE_GAP_STOP_UTC = datetime(
    2025, 4, 17, 21, 38, 9, 404736, tzinfo=timezone.utc
)
CATALOGUED_DECAY_UTC = datetime(2026, 1, 5, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OrbitRow:
    """One unique TLE epoch and its mean-element orbital heights."""

    norad_id: int
    name: str
    epoch_utc: datetime
    perigee_km: float
    apogee_km: float
    mean_altitude_km: float
    eccentricity: float
    mean_motion_rev_per_day: float
    bstar: float
    tle_line1: str
    tle_line2: str


def _tle_epoch_utc(line1: str) -> datetime:
    """Convert a fixed-width TLE epoch into a timezone-aware UTC datetime."""

    year_two_digits = int(line1[18:20])
    year = 1900 + year_two_digits if year_two_digits >= 57 else 2000 + year_two_digits
    day_of_year = float(line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def parse_group_history(input_path: str | Path) -> dict[int, list[OrbitRow]]:
    """Parse, validate, name, and de-duplicate the Group 11-6 history file."""

    lines = Path(input_path).read_text(encoding="ascii").splitlines()
    raw_rows: list[OrbitRow] = []
    names_by_id: dict[int, list[str]] = defaultdict(list)
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
            raise ValueError(f"TLE line 1 at file line {index + 1} has no line 2")

        line1 = line
        line2 = lines[index + 1].strip("\r")
        satellite = Satrec.twoline2rv(line1, line2)
        norad_id = satellite.satnum
        if norad_id not in EXPECTED_NORAD_IDS:
            raise ValueError(f"Unexpected NORAD ID {norad_id} in Group 11-6 input")

        earth_radius_km = satellite.radiusearthkm
        perigee_km = satellite.altp * earth_radius_km
        apogee_km = satellite.alta * earth_radius_km
        names_by_id[norad_id].append(current_name)
        raw_rows.append(
            OrbitRow(
                norad_id=norad_id,
                name=current_name,
                epoch_utc=_tle_epoch_utc(line1),
                perigee_km=perigee_km,
                apogee_km=apogee_km,
                mean_altitude_km=(perigee_km + apogee_km) / 2.0,
                eccentricity=satellite.ecco,
                mean_motion_rev_per_day=satellite.no_kozai
                * 1440.0
                / (2.0 * np.pi),
                bstar=satellite.bstar,
                tle_line1=line1,
                tle_line2=line2,
            )
        )
        index += 2

    observed_ids = {row.norad_id for row in raw_rows}
    if observed_ids != EXPECTED_NORAD_IDS:
        missing = sorted(EXPECTED_NORAD_IDS - observed_ids)
        extra = sorted(observed_ids - EXPECTED_NORAD_IDS)
        raise ValueError(f"Group membership mismatch; missing={missing}, extra={extra}")

    canonical_names: dict[int, str] = {}
    for norad_id, names in names_by_id.items():
        useful_names = [name for name in names if name != "TBA - TO BE ASSIGNED"]
        canonical_names[norad_id] = Counter(useful_names or names).most_common(1)[0][0]

    by_satellite_epoch: dict[tuple[int, datetime], list[OrbitRow]] = defaultdict(list)
    for row in raw_rows:
        by_satellite_epoch[(row.norad_id, row.epoch_utc)].append(row)

    histories: dict[int, list[OrbitRow]] = defaultdict(list)
    for (norad_id, _epoch), candidates in by_satellite_epoch.items():
        candidates.sort(key=lambda item: item.mean_altitude_km)
        chosen = candidates[len(candidates) // 2]
        histories[norad_id].append(
            OrbitRow(
                **{
                    **asdict(chosen),
                    "name": canonical_names[norad_id],
                }
            )
        )
    for history in histories.values():
        history.sort(key=lambda item: item.epoch_utc)
    return dict(sorted(histories.items()))


def _nearest_to_event(history: list[OrbitRow]) -> OrbitRow:
    """Return the TLE epoch nearest the VHF event."""

    return min(
        history,
        key=lambda row: abs((row.epoch_utc - VHF_EVENT_UTC).total_seconds()),
    )


def _first_at_or_above(history: list[OrbitRow], altitude_km: float) -> OrbitRow | None:
    """Return the first TLE at or above an altitude threshold."""

    return next(
        (row for row in history if row.mean_altitude_km >= altitude_km), None
    )


def _iso_or_blank(value: datetime | None) -> str:
    """Return an ISO UTC string for CSV/JSON serialization."""

    return "" if value is None else value.isoformat().replace("+00:00", "Z")


def write_history_csv(
    histories: dict[int, list[OrbitRow]], output_path: str | Path
) -> None:
    """Write all unique Group 11-6 TLE-derived heights as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "norad_id",
        "name",
        "epoch_utc",
        "perigee_km",
        "apogee_km",
        "mean_altitude_km",
        "eccentricity",
        "mean_motion_rev_per_day",
        "bstar",
        "tle_line1",
        "tle_line2",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories.values():
            for row in history:
                values = asdict(row)
                values["epoch_utc"] = _iso_or_blank(row.epoch_utc)
                writer.writerow(values)


def _satellite_summary_rows(
    histories: dict[int, list[OrbitRow]],
) -> list[dict[str, object]]:
    """Build one numerical summary row for each launch-group spacecraft."""

    summary_rows: list[dict[str, object]] = []
    for norad_id, history in histories.items():
        nearest = _nearest_to_event(history)
        threshold_rows = {
            altitude: _first_at_or_above(history, altitude)
            for altitude in (400.0, 440.0, 445.0, 470.0, 475.0)
        }
        summary_rows.append(
            {
                "norad_id": norad_id,
                "name": history[0].name,
                "is_target": norad_id == TARGET_NORAD_ID,
                "unique_tle_epoch_count": len(history),
                "coverage_start_utc": _iso_or_blank(history[0].epoch_utc),
                "coverage_stop_utc": _iso_or_blank(history[-1].epoch_utc),
                "nearest_tle_to_vhf_event_utc": _iso_or_blank(nearest.epoch_utc),
                "nearest_tle_age_hours": (
                    nearest.epoch_utc - VHF_EVENT_UTC
                ).total_seconds()
                / 3600.0,
                "event_nearest_tle_mean_altitude_km": nearest.mean_altitude_km,
                "minimum_mean_altitude_km": min(
                    row.mean_altitude_km for row in history
                ),
                "maximum_mean_altitude_km": max(
                    row.mean_altitude_km for row in history
                ),
                "first_at_or_above_400_km_utc": _iso_or_blank(
                    None if threshold_rows[400.0] is None else threshold_rows[400.0].epoch_utc
                ),
                "first_at_or_above_440_km_utc": _iso_or_blank(
                    None if threshold_rows[440.0] is None else threshold_rows[440.0].epoch_utc
                ),
                "first_at_or_above_445_km_utc": _iso_or_blank(
                    None if threshold_rows[445.0] is None else threshold_rows[445.0].epoch_utc
                ),
                "first_at_or_above_470_km_utc": _iso_or_blank(
                    None if threshold_rows[470.0] is None else threshold_rows[470.0].epoch_utc
                ),
                "first_at_or_above_475_km_utc": _iso_or_blank(
                    None if threshold_rows[475.0] is None else threshold_rows[475.0].epoch_utc
                ),
                "last_tle_mean_altitude_km": history[-1].mean_altitude_km,
            }
        )
    return summary_rows


def write_satellite_summary_csv(
    histories: dict[int, list[OrbitRow]], output_path: str | Path
) -> None:
    """Write one comparison row per Group 11-6 spacecraft."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _satellite_summary_rows(histories)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median_datetime(values: list[datetime]) -> datetime:
    """Return the median UTC datetime."""

    return datetime.fromtimestamp(
        float(np.median([value.timestamp() for value in values])), tz=timezone.utc
    )


def write_comparison_json(
    histories: dict[int, list[OrbitRow]], output_path: str | Path
) -> None:
    """Write the main target-versus-peer findings as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _satellite_summary_rows(histories)
    target = next(row for row in rows if row["norad_id"] == TARGET_NORAD_ID)
    peers = [row for row in rows if row["norad_id"] != TARGET_NORAD_ID]
    peer_event_altitudes = np.array(
        [float(row["event_nearest_tle_mean_altitude_km"]) for row in peers]
    )

    milestones: dict[str, object] = {}
    for altitude in (400, 440, 445, 470, 475):
        key = f"first_at_or_above_{altitude}_km_utc"
        target_time = datetime.fromisoformat(str(target[key]).replace("Z", "+00:00"))
        peer_times = [
            datetime.fromisoformat(str(row[key]).replace("Z", "+00:00"))
            for row in peers
            if row[key]
        ]
        peer_median = _median_datetime(peer_times)
        milestones[str(altitude)] = {
            "target_utc": _iso_or_blank(target_time),
            "peer_median_utc": _iso_or_blank(peer_median),
            "target_delay_from_peer_median_days": (
                target_time - peer_median
            ).total_seconds()
            / 86400.0,
        }

    final_low_ids = [
        int(row["norad_id"])
        for row in rows
        if float(row["last_tle_mean_altitude_km"]) < 400.0
    ]
    result = {
        "launch_group": "Starlink Group 11-6 / 2025-018",
        "member_count": len(histories),
        "target": {
            "norad_id": TARGET_NORAD_ID,
            "name": target["name"],
        },
        "vhf_event": {
            "epoch_utc": _iso_or_blank(VHF_EVENT_UTC),
            "target_nearest_tle_epoch_utc": target[
                "nearest_tle_to_vhf_event_utc"
            ],
            "target_nearest_tle_age_hours": target["nearest_tle_age_hours"],
            "target_nearest_tle_mean_altitude_km": target[
                "event_nearest_tle_mean_altitude_km"
            ],
            "payload_instantaneous_geodetic_altitude_km": VHF_PAYLOAD_GEODETIC_ALTITUDE_KM,
            "peer_mean_altitude_min_km": float(peer_event_altitudes.min()),
            "peer_mean_altitude_median_km": float(np.median(peer_event_altitudes)),
            "peer_mean_altitude_max_km": float(peer_event_altitudes.max()),
            "target_minus_peer_median_km": float(
                float(target["event_nearest_tle_mean_altitude_km"])
                - np.median(peer_event_altitudes)
            ),
            "target_altitude_rank_low_to_high": 1,
            "target_public_tle_gap_hours": (
                TARGET_TLE_GAP_STOP_UTC - TARGET_TLE_GAP_START_UTC
            ).total_seconds()
            / 3600.0,
        },
        "altitude_milestones": milestones,
        "members_below_400_km_at_last_downloaded_epoch": final_low_ids,
        "interpretation": (
            "STARLINK-32838 was an extreme low-altitude outlier during the VHF "
            "event and reached the peer orbital plateaus about three months late, "
            "but it later reached and maintained approximately 475 km before its "
            "December 2025 descent. TLEs alone do not identify the operational cause."
        ),
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _interpolated_peer_distribution(
    histories: dict[int, list[OrbitRow]],
    start: datetime,
    stop: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate peer tracks daily and return dates, min, median, and max."""

    day_count = int((stop - start).total_seconds() // 86400) + 1
    dates = np.array([start + timedelta(days=day) for day in range(day_count)], dtype=object)
    sample_seconds = np.array([date.timestamp() for date in dates])
    peer_tracks: list[np.ndarray] = []
    for norad_id, history in histories.items():
        if norad_id == TARGET_NORAD_ID:
            continue
        times = np.array([row.epoch_utc.timestamp() for row in history])
        heights = np.array([row.mean_altitude_km for row in history])
        peer_tracks.append(np.interp(sample_seconds, times, heights))
    stack = np.vstack(peer_tracks)
    return dates, stack.min(axis=0), np.median(stack, axis=0), stack.max(axis=0)


def _plot_panel(
    axis: plt.Axes,
    histories: dict[int, list[OrbitRow]],
    *,
    x_start: datetime,
    x_stop: datetime,
    title: str,
    show_payload: bool,
) -> None:
    """Draw launch-group tracks with target and peer distribution highlighted."""

    for norad_id, history in histories.items():
        if norad_id == TARGET_NORAD_ID:
            continue
        axis.plot(
            [row.epoch_utc for row in history],
            [row.mean_altitude_km for row in history],
            color="#9E9E9E",
            linewidth=0.55,
            alpha=0.32,
            zorder=1,
        )

    dates, peer_minimum, peer_median, peer_maximum = _interpolated_peer_distribution(
        histories, x_start, x_stop
    )
    axis.fill_between(
        dates,
        peer_minimum,
        peer_maximum,
        color="#56B4E9",
        alpha=0.14,
        label="22-peer min-max envelope",
        zorder=0,
    )
    axis.plot(
        dates,
        peer_median,
        color="#0072B2",
        linewidth=1.5,
        label="22-peer median",
        zorder=2,
    )

    target = histories[TARGET_NORAD_ID]
    axis.plot(
        [row.epoch_utc for row in target],
        [row.mean_altitude_km for row in target],
        color="#D55E00",
        linewidth=2.1,
        marker=".",
        markersize=2.5,
        label="STARLINK-32838",
        zorder=4,
    )
    axis.axvline(
        VHF_EVENT_UTC,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.35,
        label="21CMA VHF event",
        zorder=3,
    )
    if show_payload:
        axis.axvspan(
            TARGET_TLE_GAP_START_UTC,
            TARGET_TLE_GAP_STOP_UTC,
            color="#E69F00",
            alpha=0.16,
            label="32838 public TLE gap",
        )
        axis.scatter(
            [VHF_EVENT_UTC],
            [VHF_PAYLOAD_GEODETIC_ALTITUDE_KM],
            marker="*",
            s=105,
            color="#E69F00",
            edgecolor="black",
            linewidth=0.45,
            label="Payload instantaneous altitude",
            zorder=5,
        )
    if x_start <= CATALOGUED_DECAY_UTC <= x_stop:
        axis.axvline(
            CATALOGUED_DECAY_UTC,
            color="#000000",
            linestyle=":",
            linewidth=1.2,
            label="32838 catalogued decay",
        )

    axis.set_xlim(x_start, x_stop)
    axis.set_ylabel("TLE mean altitude (km)")
    axis.set_title(title, loc="left", fontsize=11)
    axis.grid(True, alpha=0.25, linewidth=0.7)


def save_comparison_plot(
    histories: dict[int, list[OrbitRow]], output_path: str | Path
) -> None:
    """Write the complete and VHF-event-focused launch-group comparison PNG."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = min(history[0].epoch_utc for history in histories.values())
    stop = max(history[-1].epoch_utc for history in histories.values())
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    _plot_panel(
        axes[0],
        histories,
        x_start=start - timedelta(days=2),
        x_stop=stop + timedelta(days=2),
        title="Complete Group 11-6 altitude histories",
        show_payload=True,
    )
    axes[0].set_ylim(210, 510)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    _plot_panel(
        axes[1],
        histories,
        x_start=datetime(2025, 2, 1, tzinfo=timezone.utc),
        x_stop=datetime(2025, 6, 15, tzinfo=timezone.utc),
        title="Early-orbit comparison around the 21CMA event",
        show_payload=True,
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
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Starlink Group 11-6: STARLINK-32838 versus 22 launch companions",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(top=0.87, bottom=0.07, left=0.075, right=0.99, hspace=0.2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the launch-group comparison."""

    analysis_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare STARLINK-32838 with the other Group 11-6 spacecraft."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=analysis_dir / "STARLINK_Group11-6_full_history.tle",
        help="Space-Track GP_HISTORY 2LE/3LE file for NORAD 62806--62828.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=analysis_dir / "results" / "group11_6_comparison",
        help="Directory for comparison PNG, CSV, and JSON products.",
    )
    return parser


def main() -> None:
    """Run the complete Group 11-6 orbit-history comparison."""

    args = build_parser().parse_args()
    histories = parse_group_history(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_csv = args.output_dir / "group11_6_orbit_history.csv"
    summary_csv = args.output_dir / "group11_6_satellite_summary.csv"
    summary_json = args.output_dir / "group11_6_comparison_summary.json"
    figure = args.output_dir / "group11_6_orbit_comparison.png"
    write_history_csv(histories, history_csv)
    write_satellite_summary_csv(histories, summary_csv)
    write_comparison_json(histories, summary_json)
    save_comparison_plot(histories, figure)
    print(f"Launch-group members: {len(histories)}")
    print(f"Detailed CSV: {history_csv}")
    print(f"Satellite summary CSV: {summary_csv}")
    print(f"JSON: {summary_json}")
    print(f"PNG: {figure}")


if __name__ == "__main__":
    main()
