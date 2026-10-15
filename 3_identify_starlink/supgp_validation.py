"""Independent SpaceX-E SupGP validation for the STARLINK-32838 result.

Functions and outputs
---------------------
``run_validation(supgp_csv, results_dir, station_center_csv)``
    Reads the CelesTrak SupGP CSV, propagates each event-day solution with
    SGP4, compares it with the Event-1 inferred ECEF state, evaluates the
    two-event differential Doppler, summarizes the orbit evolution, and
    returns a JSON-serializable dictionary.
``load_station_center(path)``
    Reads the one-row E08 WGS84 station-center CSV and returns latitude,
    longitude, and ellipsoidal height for station-geometry calculations.

Saved files in ``results_dir``
------------------------------
``supgp_state_match.csv``
    Exact-time and optimized ECEF position/velocity residuals for every
    event-day SupGP record.
``supgp_vs_standard_tle.csv``
    Compact comparison between the original all-catalogue winner TLE and the
    nearest pre-event SpaceX-E SupGP result.
``supgp_differential_doppler.csv``
    Event-1/Event-2 station geometry and predicted Doppler from the selected
    pre-event SupGP solution.
``supgp_orbit_evolution.csv``
    Epoch, mean-motion-derived altitude, fit RMS, and fit parameters for all
    SupGP records in the requested file.
``figure14_1_supgp_orbit_evolution.png/.pdf``
    Mean altitude and SupGP-fit RMS across the supplied history, with the
    Event-1 state time marked.
``supgp_validation_summary.json``
    Input provenance, selected solution, comparison metrics, Doppler result,
    and interpretation limits.

The CelesTrak ``RMS`` field is treated only as the RMS residual of the SGP4
fit to source ephemeris.  It is not used as a position covariance or a 1-sigma
uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sgp4 import omm
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, load
from skyfield.framelib import itrs


DEFAULT_ROOT = Path("/home/yxp/FM/starlink_vhf_137055")
DEFAULT_SUPGP = DEFAULT_ROOT / "目标确定/sat000062827.csv"
DEFAULT_RESULTS = DEFAULT_ROOT / "目标确定/results"
DEFAULT_STATION_CENTER = (
    DEFAULT_ROOT / "瀑布图/results/e08_station_center_wgs84.csv"
)

EVENT1_STATE_UTC = datetime(2025, 4, 15, 14, 0, 0, 250000, tzinfo=timezone.utc)
EVENT_SPACING_S = 99.9991
VELOCITY_WEIGHT_S = 100.0
TIME_BOUND_S = 180.0
RF_FREQUENCY_HZ = 137_055_000.0
LIGHT_SPEED_KM_S = 299_792.458

EARTH_MU_KM3_S2 = 398600.4418
WGS84_EQUATORIAL_RADIUS_KM = 6378.137


def _plain(value: Any) -> Any:
    """Convert NumPy, Path, and datetime values to JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha256(path: Path, block_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one small input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_epoch(text: str) -> datetime:
    """Parse a timezone-naive OMM UTC epoch as an aware datetime."""
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def load_supgp_rows(path: str | Path) -> list[dict[str, str]]:
    """Read and validate the requested STARLINK-32838 SupGP CSV rows."""
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No SupGP rows found in {path}")
    required = {
        "OBJECT_NAME",
        "EPOCH",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "EPHEMERIS_TYPE",
        "CLASSIFICATION_TYPE",
        "NORAD_CAT_ID",
        "ELEMENT_SET_NO",
        "REV_AT_EPOCH",
        "BSTAR",
        "MEAN_MOTION_DOT",
        "MEAN_MOTION_DDOT",
        "RMS",
        "DATA_SOURCE",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing SupGP columns: {sorted(missing)}")
    if {row["NORAD_CAT_ID"] for row in rows} != {"62827"}:
        raise ValueError("Expected only NORAD 62827 in the SupGP file")
    if {row["DATA_SOURCE"] for row in rows} != {"SpaceX-E"}:
        raise ValueError("Expected CelesTrak SpaceX-E SupGP data")
    return sorted(rows, key=lambda row: _parse_epoch(row["EPOCH"]))


def load_station_center(path: str | Path) -> dict[str, float | str]:
    """Read and validate the E08 WGS84 station center saved by the waterfall notebook."""
    path = Path(path)
    table = pd.read_csv(path)
    required = {
        "station",
        "longitude_deg_east",
        "latitude_deg_north",
        "ellipsoidal_height_m",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing station-center columns: {sorted(missing)}")
    if len(table) != 1 or str(table.iloc[0]["station"]).upper() != "E08":
        raise ValueError("Expected exactly one E08 row in the station-center CSV")
    row = table.iloc[0]
    return {
        "source_file": str(path),
        "station": "E08",
        "longitude_deg_east": float(row["longitude_deg_east"]),
        "latitude_deg_north": float(row["latitude_deg_north"]),
        "ellipsoidal_height_m": float(row["ellipsoidal_height_m"]),
    }


def earth_satellite_from_omm_row(row: dict[str, str], timescale) -> EarthSatellite:
    """Create a Skyfield satellite from one CelesTrak OMM/SupGP row."""
    satrec = Satrec()
    omm.initialize(satrec, row)
    return EarthSatellite.from_satrec(satrec, timescale)


def satellite_state_ecef(
    satellite: EarthSatellite,
    utc: datetime,
    timescale,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ECEF/ITRS position in km and velocity in km/s."""
    state = satellite.at(timescale.from_datetime(utc))
    position, velocity = state.frame_xyz_and_velocity(itrs)
    return np.asarray(position.km), np.asarray(velocity.km_per_s)


def optimize_state_match(
    satellite: EarthSatellite,
    reference_utc: datetime,
    packet_position_km: np.ndarray,
    packet_velocity_km_s: np.ndarray,
    timescale,
    *,
    velocity_weight_s: float = VELOCITY_WEIGHT_S,
    time_bound_s: float = TIME_BOUND_S,
) -> dict[str, Any]:
    """Compare one SupGP solution with the packet state at and near its UTC."""

    def errors(offset_s: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        position, velocity = satellite_state_ecef(
            satellite,
            reference_utc + timedelta(seconds=float(offset_s)),
            timescale,
        )
        position_error = float(np.linalg.norm(position - packet_position_km))
        velocity_error = float(np.linalg.norm(velocity - packet_velocity_km_s))
        return position, velocity, position_error, velocity_error

    exact_position, exact_velocity, exact_position_error, exact_velocity_error = errors(0.0)

    def objective(offset_s: float) -> float:
        _, _, position_error, velocity_error = errors(offset_s)
        return position_error**2 + (velocity_weight_s * velocity_error) ** 2

    optimum = minimize_scalar(
        objective,
        bounds=(-time_bound_s, time_bound_s),
        method="bounded",
        options={"xatol": 1e-6},
    )
    best_offset_s = float(optimum.x)
    best_position, best_velocity, best_position_error, best_velocity_error = errors(
        best_offset_s
    )
    return {
        "exact_position_error_km": exact_position_error,
        "exact_velocity_error_m_s": exact_velocity_error * 1000.0,
        "exact_x_ecef_km": exact_position[0],
        "exact_y_ecef_km": exact_position[1],
        "exact_z_ecef_km": exact_position[2],
        "exact_vx_ecef_km_s": exact_velocity[0],
        "exact_vy_ecef_km_s": exact_velocity[1],
        "exact_vz_ecef_km_s": exact_velocity[2],
        "best_offset_s": best_offset_s,
        "best_time_utc": reference_utc + timedelta(seconds=best_offset_s),
        "best_position_error_km": best_position_error,
        "best_velocity_error_m_s": best_velocity_error * 1000.0,
        "best_x_ecef_km": best_position[0],
        "best_y_ecef_km": best_position[1],
        "best_z_ecef_km": best_position[2],
        "best_vx_ecef_km_s": best_velocity[0],
        "best_vy_ecef_km_s": best_velocity[1],
        "best_vz_ecef_km_s": best_velocity[2],
        "joint_score_km2": float(optimum.fun),
    }


def station_ecef_km(
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
) -> np.ndarray:
    """Return the observatory WGS84 station center in ECEF km."""
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    height_km = elevation_m / 1000.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    prime_vertical_km = WGS84_EQUATORIAL_RADIUS_KM / np.sqrt(
        1.0 - eccentricity_sq * np.sin(latitude) ** 2
    )
    return np.asarray(
        [
            (prime_vertical_km + height_km)
            * np.cos(latitude)
            * np.cos(longitude),
            (prime_vertical_km + height_km)
            * np.cos(latitude)
            * np.sin(longitude),
            (prime_vertical_km * (1.0 - eccentricity_sq) + height_km)
            * np.sin(latitude),
        ]
    )


def event_geometry(
    satellite: EarthSatellite,
    event_utc: datetime,
    model_offset_s: float,
    site_ecef_km: np.ndarray,
    timescale,
) -> dict[str, Any]:
    """Return station range, range rate, and predicted Doppler for one event."""
    model_utc = event_utc + timedelta(seconds=float(model_offset_s))
    position, velocity = satellite_state_ecef(satellite, model_utc, timescale)
    relative = position - site_ecef_km
    slant_range_km = float(np.linalg.norm(relative))
    range_rate_km_s = float(np.dot(relative, velocity) / slant_range_km)
    predicted_doppler_hz = (
        -RF_FREQUENCY_HZ * range_rate_km_s / LIGHT_SPEED_KM_S
    )
    return {
        "reference_utc": event_utc,
        "model_utc": model_utc,
        "slant_range_km": slant_range_km,
        "range_rate_km_s": range_rate_km_s,
        "predicted_doppler_hz": predicted_doppler_hz,
    }


def mean_altitude_from_mean_motion_km(mean_motion_rev_day: float) -> float:
    """Convert mean motion to mean semi-major-axis altitude above WGS84 a."""
    angular_rate_rad_s = mean_motion_rev_day * 2.0 * math.pi / 86400.0
    semi_major_axis_km = (
        EARTH_MU_KM3_S2 / angular_rate_rad_s**2
    ) ** (1.0 / 3.0)
    return semi_major_axis_km - WGS84_EQUATORIAL_RADIUS_KM


def save_orbit_evolution_figure(
    orbit_table: pd.DataFrame,
    output_stem: Path,
    event_utc: datetime,
) -> None:
    """Save the SupGP mean-altitude and fit-RMS history as PNG and PDF."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Source Han Sans CN",
                "Noto Sans CJK TC",
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    epochs = pd.to_datetime(orbit_table["epoch_utc"], utc=True)
    event_timestamp = pd.Timestamp(event_utc)
    pre_mask = epochs <= event_timestamp

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        constrained_layout=True,
    )
    axes[0].plot(
        epochs,
        orbit_table["mean_altitude_km"],
        color="#315A9B",
        marker="o",
        markersize=3.8,
        linewidth=1.45,
        label="SpaceX-E SupGP",
    )
    axes[0].scatter(
        epochs[pre_mask].iloc[-1:],
        orbit_table.loc[pre_mask, "mean_altitude_km"].iloc[-1:],
        marker="*",
        s=135,
        color="#D94F3D",
        edgecolor="black",
        linewidth=0.65,
        zorder=5,
        label="Selected pre-event SupGP",
    )
    axes[0].axvline(event_timestamp, color="#D94F3D", linestyle="--", linewidth=1.2)
    axes[0].text(
        event_timestamp,
        axes[0].get_ylim()[1],
        "  Event 1",
        color="#B13C2E",
        ha="left",
        va="top",
        fontsize=10,
    )
    axes[0].set_ylabel("Mean altitude from mean motion (km)")
    axes[0].legend(loc="best", frameon=False)
    axes[0].grid(True, which="major", color="#D7DCE2", linewidth=0.7)

    axes[1].plot(
        epochs,
        orbit_table["rms_fit_km"],
        color="#6A4C93",
        marker="o",
        markersize=3.5,
        linewidth=1.25,
    )
    axes[1].axvline(event_timestamp, color="#D94F3D", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("SGP4 fit RMS (km)")
    axes[1].set_xlabel("SupGP epoch (UTC)")
    axes[1].grid(True, which="major", color="#D7DCE2", linewidth=0.7)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[1].get_xticklabels(), rotation=22, ha="right")

    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_validation(
    supgp_csv: str | Path = DEFAULT_SUPGP,
    results_dir: str | Path = DEFAULT_RESULTS,
    station_center_csv: str | Path = DEFAULT_STATION_CENTER,
) -> dict[str, Any]:
    """Run the independent SupGP validation and save all structured outputs."""
    supgp_csv = Path(supgp_csv)
    results_dir = Path(results_dir)
    station_center_csv = Path(station_center_csv)
    results_dir.mkdir(parents=True, exist_ok=True)

    packet_state_path = results_dir / "event1_packet_state.csv"
    target_summary_path = results_dir / "target_identification_summary.json"
    differential_cfo_path = results_dir / "differential_cfo_summary.csv"
    packet_pair_path = results_dir / "packet_pair_summary.csv"
    for path in (
        supgp_csv,
        packet_state_path,
        target_summary_path,
        differential_cfo_path,
        packet_pair_path,
        station_center_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = load_supgp_rows(supgp_csv)
    station_center = load_station_center(station_center_csv)
    timescale = load.timescale()

    packet_state_series = pd.read_csv(packet_state_path, index_col=0)["value"]
    packet_position_km = np.asarray(
        [
            float(packet_state_series["x_ecef_km"]),
            float(packet_state_series["y_ecef_km"]),
            float(packet_state_series["z_ecef_km"]),
        ]
    )
    packet_velocity_km_s = np.asarray(
        [
            float(packet_state_series["vx_ecef_km_s"]),
            float(packet_state_series["vy_ecef_km_s"]),
            float(packet_state_series["vz_ecef_km_s"]),
        ]
    )
    state_utc = datetime.fromisoformat(str(packet_state_series["gps_state_utc"]))

    match_rows = []
    satellite_by_epoch: dict[str, EarthSatellite] = {}
    for row in rows:
        epoch = _parse_epoch(row["EPOCH"])
        if epoch.date() != state_utc.date():
            continue
        satellite = earth_satellite_from_omm_row(row, timescale)
        satellite_by_epoch[row["EPOCH"]] = satellite
        match = optimize_state_match(
            satellite,
            state_utc,
            packet_position_km,
            packet_velocity_km_s,
            timescale,
        )
        match_rows.append(
            {
                "object_name": row["OBJECT_NAME"],
                "norad_id": int(row["NORAD_CAT_ID"]),
                "data_source": row["DATA_SOURCE"],
                "epoch_utc": epoch,
                "epoch_age_at_state_h": (state_utc - epoch).total_seconds() / 3600.0,
                "rms_fit_km": float(row["RMS"]),
                **match,
            }
        )
    state_match = pd.DataFrame(match_rows).sort_values("epoch_utc")
    state_match.to_csv(results_dir / "supgp_state_match.csv", index=False)

    pre_event_match = state_match[state_match["epoch_utc"] <= state_utc]
    if pre_event_match.empty:
        raise ValueError("No pre-event SupGP solution is present")
    selected_match = pre_event_match.iloc[-1]
    selected_epoch_key = selected_match["epoch_utc"].strftime("%Y-%m-%dT%H:%M:%S.%f")
    selected_epoch_key = min(
        satellite_by_epoch,
        key=lambda key: abs(
            (_parse_epoch(key) - selected_match["epoch_utc"]).total_seconds()
        ),
    )
    selected_satellite = satellite_by_epoch[selected_epoch_key]

    target_summary = json.loads(target_summary_path.read_text())
    original_winner = target_summary["winner"]
    comparison = pd.DataFrame(
        [
            {
                "solution": "Original full-catalogue winner TLE",
                "source": "Active_0415.tle",
                "epoch_utc": original_winner["tle_epoch_utc"],
                "epoch_age_at_state_h": original_winner["tle_age_hours"],
                "best_offset_s": original_winner["best_offset_s"],
                "best_position_error_km": original_winner["best_position_error_km"],
                "best_velocity_error_m_s": original_winner[
                    "best_velocity_error_km_s"
                ]
                * 1000.0,
                "rms_fit_km": np.nan,
            },
            {
                "solution": "Nearest pre-event SpaceX-E SupGP",
                "source": "CelesTrak SpaceX-E",
                "epoch_utc": selected_match["epoch_utc"],
                "epoch_age_at_state_h": selected_match["epoch_age_at_state_h"],
                "best_offset_s": selected_match["best_offset_s"],
                "best_position_error_km": selected_match[
                    "best_position_error_km"
                ],
                "best_velocity_error_m_s": selected_match[
                    "best_velocity_error_m_s"
                ],
                "rms_fit_km": selected_match["rms_fit_km"],
            },
        ]
    )
    comparison.to_csv(results_dir / "supgp_vs_standard_tle.csv", index=False)

    pair_values = pd.read_csv(packet_pair_path, index_col=0)["value"]
    measured_spacing_s = float(
        pair_values.get("measured_preamble_spacing_s", EVENT_SPACING_S)
    )
    cfo_values = pd.read_csv(differential_cfo_path, index_col=0)["value"]
    observed_delta_cfo_hz = float(cfo_values["event2_minus_event1_cfo_hz"])
    observed_sigma_hz = float(cfo_values["conservative_empirical_uncertainty_hz"])

    site = station_ecef_km(
        station_center["latitude_deg_north"],
        station_center["longitude_deg_east"],
        station_center["ellipsoidal_height_m"],
    )
    event_times = {
        "Event 1": state_utc,
        "Event 2": state_utc + timedelta(seconds=measured_spacing_s),
    }
    geometry_rows = []
    for label, event_utc in event_times.items():
        geometry_rows.append(
            {
                "event": label,
                **event_geometry(
                    selected_satellite,
                    event_utc,
                    float(selected_match["best_offset_s"]),
                    site,
                    timescale,
                ),
            }
        )
    geometry = pd.DataFrame(geometry_rows).set_index("event")
    predicted_delta_cfo_hz = float(
        geometry.loc["Event 2", "predicted_doppler_hz"]
        - geometry.loc["Event 1", "predicted_doppler_hz"]
    )
    geometry["observed_delta_cfo_hz"] = np.nan
    geometry["predicted_delta_from_event1_hz"] = (
        geometry["predicted_doppler_hz"]
        - geometry.loc["Event 1", "predicted_doppler_hz"]
    )
    geometry.loc["Event 2", "observed_delta_cfo_hz"] = observed_delta_cfo_hz
    geometry.to_csv(results_dir / "supgp_differential_doppler.csv")

    orbit_rows = []
    for row in rows:
        mean_motion = float(row["MEAN_MOTION"])
        orbit_rows.append(
            {
                "epoch_utc": _parse_epoch(row["EPOCH"]),
                "mean_motion_rev_day": mean_motion,
                "mean_altitude_km": mean_altitude_from_mean_motion_km(mean_motion),
                "eccentricity": float(row["ECCENTRICITY"]),
                "inclination_deg": float(row["INCLINATION"]),
                "raan_deg": float(row["RA_OF_ASC_NODE"]),
                "mean_motion_dot_rev_day2": float(row["MEAN_MOTION_DOT"]),
                "bstar": float(row["BSTAR"]),
                "rms_fit_km": float(row["RMS"]),
                "element_set_no": int(row["ELEMENT_SET_NO"]),
                "data_source": row["DATA_SOURCE"],
            }
        )
    orbit_table = pd.DataFrame(orbit_rows).sort_values("epoch_utc")
    orbit_table.to_csv(results_dir / "supgp_orbit_evolution.csv", index=False)
    save_orbit_evolution_figure(
        orbit_table,
        results_dir / "figure14_1_supgp_orbit_evolution",
        state_utc,
    )

    event_day_orbit = orbit_table[
        orbit_table["epoch_utc"].dt.date == state_utc.date()
    ]
    summary = {
        "scope": (
            "Independent post-identification refinement of STARLINK-32838; "
            "not a replacement for the original all-catalogue blind search."
        ),
        "input": {
            "supgp_csv": str(supgp_csv),
            "sha256": _sha256(supgp_csv),
            "records": len(rows),
            "epoch_start_utc": _parse_epoch(rows[0]["EPOCH"]),
            "epoch_stop_utc": _parse_epoch(rows[-1]["EPOCH"]),
            "data_source": "SpaceX-E",
            "norad_id": 62827,
        },
        "station_center": station_center,
        "event1_packet_state": {
            "gps_state_utc": state_utc,
            "position_ecef_km": packet_position_km.tolist(),
            "velocity_ecef_km_s": packet_velocity_km_s.tolist(),
        },
        "selected_pre_event_supgp": {
            "epoch_utc": selected_match["epoch_utc"],
            "epoch_age_at_state_h": selected_match["epoch_age_at_state_h"],
            "rms_fit_km": selected_match["rms_fit_km"],
            "exact_position_error_km": selected_match[
                "exact_position_error_km"
            ],
            "exact_velocity_error_m_s": selected_match[
                "exact_velocity_error_m_s"
            ],
            "best_offset_s": selected_match["best_offset_s"],
            "best_position_error_km": selected_match[
                "best_position_error_km"
            ],
            "best_velocity_error_m_s": selected_match[
                "best_velocity_error_m_s"
            ],
        },
        "original_winner_tle": {
            "epoch_utc": original_winner["tle_epoch_utc"],
            "epoch_age_at_state_h": original_winner["tle_age_hours"],
            "best_offset_s": original_winner["best_offset_s"],
            "best_position_error_km": original_winner[
                "best_position_error_km"
            ],
            "best_velocity_error_m_s": original_winner[
                "best_velocity_error_km_s"
            ]
            * 1000.0,
        },
        "differential_doppler": {
            "event_spacing_s": measured_spacing_s,
            "observed_delta_cfo_hz": observed_delta_cfo_hz,
            "observed_empirical_uncertainty_hz": observed_sigma_hz,
            "supgp_predicted_delta_hz": predicted_delta_cfo_hz,
            "observed_minus_predicted_hz": (
                observed_delta_cfo_hz - predicted_delta_cfo_hz
            ),
        },
        "event_day_orbit": {
            "first_epoch_utc": event_day_orbit.iloc[0]["epoch_utc"],
            "last_epoch_utc": event_day_orbit.iloc[-1]["epoch_utc"],
            "first_mean_altitude_km": event_day_orbit.iloc[0][
                "mean_altitude_km"
            ],
            "last_mean_altitude_km": event_day_orbit.iloc[-1][
                "mean_altitude_km"
            ],
            "altitude_change_km": (
                event_day_orbit.iloc[-1]["mean_altitude_km"]
                - event_day_orbit.iloc[0]["mean_altitude_km"]
            ),
        },
        "interpretation": {
            "identity": (
                "The operator-ephemeris-derived SupGP independently tightens "
                "the ECEF state match after STARLINK-32838 was selected by the "
                "full-catalogue blind search."
            ),
            "maneuver": (
                "The mean-motion history is consistent with active orbit "
                "raising across the event date."
            ),
            "uncertainty_limit": (
                "RMS is the SGP4 fit residual to source ephemeris, not a state "
                "covariance or formal confidence interval."
            ),
        },
    }
    (results_dir / "supgp_validation_summary.json").write_text(
        json.dumps(_plain(summary), ensure_ascii=False, indent=2) + "\n"
    )
    return _plain(summary)


def main() -> int:
    """Command-line entry point; prints the saved validation summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the STARLINK-32838 identification with a historical "
            "CelesTrak SpaceX-E SupGP CSV."
        )
    )
    parser.add_argument("--supgp", type=Path, default=DEFAULT_SUPGP)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--station-center",
        type=Path,
        default=DEFAULT_STATION_CENTER,
        help="One-row E08 WGS84 station-center CSV.",
    )
    args = parser.parse_args()
    summary = run_validation(args.supgp, args.results_dir, args.station_center)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
