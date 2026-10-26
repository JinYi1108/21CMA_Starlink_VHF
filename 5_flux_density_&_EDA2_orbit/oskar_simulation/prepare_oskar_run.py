"""Prepare the E08 OSKAR telescope model and beam-pattern settings.

Functions and outputs
---------------------
``load_unique_e08_positions()``
    Loads ``ant_pos.py`` and returns the 127 unique E/N/U element positions.
``write_telescope_model()``
    Writes the OSKAR telescope model: station position, one-station layout,
    and the 127-element station layout, all in metres.
``write_event_targets()``
    Converts the existing four-event azimuth/elevation time series into an
    OSKAR horizon-coordinate target list and a row-aligned metadata CSV.
``write_settings()``
    Writes one full-horizon image configuration and one exact target-sampling
    configuration at 137.055 MHz.

The model intentionally contains only the ideal station array factor:
identical isotropic unpolarised elements, equal weights, no coupling, no
cable/gain/phase errors, and NCP tracking. OSKAR power is normalised to unity
at the NCP phase centre. Results therefore quantify the ideal geometric beam,
not a calibrated in-situ E08 sidelobe.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


FREQUENCY_HZ = 137_055_000.0
E08_LONGITUDE_DEG = 86.723982044547
E08_LATITUDE_DEG = 42.924240425222
E08_HEIGHT_M = 2607.040745328180
REFERENCE_UTC = "2025-04-15 14:00:00.000"


def load_unique_e08_positions(path: Path) -> np.ndarray:
    """Return the unique E08 element E/N/U coordinates in metres."""
    namespace = runpy.run_path(str(path))
    positions = np.asarray(namespace["data"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 coordinate array, got {positions.shape}")
    unique = np.unique(positions, axis=0)
    if unique.shape != (127, 3):
        raise ValueError(
            f"Expected 127 unique E08 positions; found {unique.shape[0]} in {path}"
        )
    return unique


def write_telescope_model(model_dir: Path, positions: np.ndarray) -> None:
    """Write the minimal one-station OSKAR telescope model."""
    station_dir = model_dir / "station000"
    station_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        model_dir / "position.txt",
        [[E08_LONGITUDE_DEG, E08_LATITUDE_DEG, E08_HEIGHT_M]],
        fmt="%.12f",
        header="longitude_deg latitude_deg WGS84_ellipsoidal_height_m",
    )
    np.savetxt(
        model_dir / "layout.txt",
        [[0.0, 0.0, 0.0]],
        fmt="%.9f",
        header="station_east_m station_north_m station_up_m",
    )
    np.savetxt(
        station_dir / "layout.txt",
        positions,
        fmt="%.9f",
        header="element_east_m element_north_m element_up_m",
    )


def write_event_targets(direction_csv: Path, target_file: Path, catalog_csv: Path) -> int:
    """Write OSKAR az/el targets and matching metadata; return row count."""
    frame = pd.read_csv(direction_csv)
    required = {
        "event_id",
        "event_status",
        "sample_index",
        "offset_from_window_start_s",
        "sample_utc",
        "azimuth_deg_north_through_east",
        "elevation_deg",
        "ncp_separation_deg",
        "slant_range_km",
        "enu_east",
        "enu_north",
        "enu_up",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Direction CSV is missing columns: {sorted(missing)}")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        target_file,
        frame[
            ["azimuth_deg_north_through_east", "elevation_deg"]
        ].to_numpy(float),
        fmt="%.12f",
        header="azimuth_deg_north_through_east elevation_deg",
    )
    frame.insert(0, "oskar_target_row", np.arange(len(frame), dtype=int))
    frame.to_csv(catalog_csv, index=False)
    return len(frame)


def common_settings(model_dir: Path) -> str:
    """Return settings shared by image and exact-target OSKAR runs."""
    return f"""[General]
app=oskar_sim_beam_pattern
version=2.13.0

[simulator]
double_precision=true
use_gpus=false

[observation]
mode=Tracking
phase_centre_ra_deg=0.0
phase_centre_dec_deg=90.0
start_frequency_hz={FREQUENCY_HZ:.1f}
num_channels=1
frequency_inc_hz=0.0
start_time_utc={REFERENCE_UTC}
length=0.001
num_time_steps=1

[telescope]
input_directory={model_dir}
normalise_beams_at_phase_centre=true
allow_station_beam_duplication=false
pol_mode=Scalar
station_type=Aperture array
aperture_array\\array_pattern\\enable=true
aperture_array\\array_pattern\\normalise=true
aperture_array\\element_pattern\\enable_numerical=false
aperture_array\\element_pattern\\functional_type=Isotropic (unpolarised)
aperture_array\\element_pattern\\normalise=false
aperture_array\\element_pattern\\taper\\type=None
"""


def write_settings(
    config_dir: Path,
    model_dir: Path,
    raw_dir: Path,
    target_file: Path,
    image_size: int,
) -> tuple[Path, Path]:
    """Write full-horizon and exact-target OSKAR setting files."""
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_cfg = config_dir / "e08_horizon_beam_137055.ini"
    sample_cfg = config_dir / "e08_four_event_samples_137055.ini"
    image_cfg.write_text(
        common_settings(model_dir)
        + f"""
[beam_pattern]
all_stations=false
station_ids=0
coordinate_frame=Horizon
coordinate_type=Beam image
root_path={raw_dir / 'e08_horizon_137055'}
beam_image\\size={image_size}
output\\separate_time_and_channel=true
output\\average_time_and_channel=false
output\\average_single_axis=None
station_outputs\\fits_image\\auto_power=true
""",
        encoding="utf-8",
    )
    sample_cfg.write_text(
        common_settings(model_dir)
        + f"""
[beam_pattern]
all_stations=false
station_ids=0
coordinate_frame=Horizon
coordinate_type=Sky model
sky_model\\file={target_file}
root_path={raw_dir / 'e08_event_samples_137055'}
output\\separate_time_and_channel=true
output\\average_time_and_channel=false
output\\average_single_axis=None
station_outputs\\text_file\\auto_power=true
""",
        encoding="utf-8",
    )
    return image_cfg, sample_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64 pixels")

    root = Path(__file__).resolve().parent
    flux_dir = root.parent
    project_root = flux_dir.parent
    model_dir = root / "model" / "e08_station"
    config_dir = root / "config"
    raw_dir = root / "results" / "oskar_raw"
    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    ant_file = flux_dir / "ant_pos.py"
    direction_csv = (
        project_root
        / "results"
        / "5_flux_density"
        / "oskar_input"
        / "four_event_direction_timeseries.csv"
    )
    target_file = config_dir / "four_event_az_el_targets.txt"
    catalog_csv = table_dir / "four_event_target_catalog.csv"

    positions = load_unique_e08_positions(ant_file)
    write_telescope_model(model_dir, positions)
    target_count = write_event_targets(direction_csv, target_file, catalog_csv)
    image_cfg, sample_cfg = write_settings(
        config_dir, model_dir, raw_dir, target_file, args.image_size
    )

    metadata = {
        "frequency_hz": FREQUENCY_HZ,
        "image_size_pixels": args.image_size,
        "num_unique_elements": int(len(positions)),
        "num_event_direction_samples": target_count,
        "station": {
            "name": "21CMA E08",
            "longitude_deg_east": E08_LONGITUDE_DEG,
            "latitude_deg_north": E08_LATITUDE_DEG,
            "wgs84_ellipsoidal_height_m": E08_HEIGHT_M,
        },
        "phase_centre": {"ra_deg": 0.0, "dec_deg": 90.0, "name": "NCP"},
        "model_assumptions": [
            "127 measured E/N/U element positions",
            "identical isotropic unpolarised elements",
            "equal element weights",
            "array factor enabled and normalised",
            "power normalised to unity at NCP phase centre",
            "no measured embedded element pattern",
            "no mutual coupling or terrain",
            "no cable, gain, or phase errors",
        ],
        "inputs": {
            "antenna_coordinates": str(ant_file),
            "event_directions": str(direction_csv),
        },
        "settings": {"image": str(image_cfg), "exact_samples": str(sample_cfg)},
    }
    (root / "results" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared 127-element model and {target_count} exact event directions.")
    print(f"Image settings:  {image_cfg}")
    print(f"Sample settings: {sample_cfg}")


if __name__ == "__main__":
    main()
