"""Post-process OSKAR beam outputs into figures and event gain tables.

Functions and outputs
---------------------
``find_single_output()``
    Resolves the OSKAR FITS/text file produced by each simulation.
``summarise_exact_samples()``
    Joins the exact OSKAR power samples to event metadata, writes the full
    time-series CSV, and writes one mean/median/min/max row per event.
``plot_horizon_beam()``
    Writes a full visible-hemisphere ideal beam PNG with all four event tracks.
``plot_event_gains()``
    Writes OSKAR gain versus time for Event 1, Event 2, Missing 1, and Missing 2.

All reported gains are relative power gains in dB, normalised to 0 dB at the
NCP phase centre. They are ideal array-factor values rather than measured E08
beam calibration values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


FREQUENCY_HZ = 137_055_000.0
C_M_S = 299_792_458.0
EVENT_ORDER = ["Event 1", "Event 2", "Missing 1", "Missing 2"]
EVENT_COLOURS = {
    "Event 1": "#d62728",
    "Event 2": "#ff7f0e",
    "Missing 1": "#1f77b4",
    "Missing 2": "#17becf",
}


def find_single_output(raw_dir: Path, pattern: str) -> Path:
    """Return exactly one matching OSKAR output path."""
    matches = sorted(raw_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {raw_dir}; "
            f"found {[p.name for p in matches]}"
        )
    return matches[0]


def ideal_array_power(positions: np.ndarray, directions: np.ndarray, latitude_deg: float) -> np.ndarray:
    """Evaluate the independently coded ideal NCP-pointed array factor."""
    lat = np.deg2rad(latitude_deg)
    ncp = np.array([0.0, np.cos(lat), np.sin(lat)])
    phase = (2.0 * np.pi * FREQUENCY_HZ / C_M_S) * (
        positions @ (directions - ncp).T
    )
    voltage = np.mean(np.exp(1j * phase), axis=0)
    return np.abs(voltage) ** 2


def summarise_exact_samples(root: Path, sample_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write full and per-event OSKAR gain tables."""
    catalog = pd.read_csv(root / "results" / "tables" / "four_event_target_catalog.csv")
    values = np.loadtxt(sample_path, comments="#")
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(catalog):
        raise ValueError(
            f"OSKAR returned {len(values)} samples but catalog has {len(catalog)} rows"
        )
    catalog["oskar_relative_power_linear"] = values
    catalog["oskar_relative_gain_db"] = 10.0 * np.log10(
        np.maximum(values, np.finfo(float).tiny)
    )

    positions = np.loadtxt(root / "model" / "e08_station" / "station000" / "layout.txt")
    directions = catalog[["enu_east", "enu_north", "enu_up"]].to_numpy(float)
    ideal_power = ideal_array_power(positions, directions, latitude_deg=42.924240425222)
    catalog["direct_formula_relative_power_linear"] = ideal_power
    catalog["direct_formula_relative_gain_db"] = 10.0 * np.log10(
        np.maximum(ideal_power, np.finfo(float).tiny)
    )
    catalog["oskar_minus_direct_formula_db"] = (
        catalog["oskar_relative_gain_db"] - catalog["direct_formula_relative_gain_db"]
    )

    full_csv = root / "results" / "tables" / "four_event_oskar_gain_timeseries.csv"
    catalog.to_csv(full_csv, index=False)

    rows: list[dict[str, float | int | str]] = []
    for event_id in EVENT_ORDER:
        group = catalog[catalog["event_id"] == event_id]
        linear_mean = float(group["oskar_relative_power_linear"].mean())
        rows.append(
            {
                "event_id": event_id,
                "event_status": group["event_status"].iloc[0],
                "num_direction_samples": int(len(group)),
                "mean_relative_power_linear": linear_mean,
                "mean_relative_gain_db": 10.0 * np.log10(max(linear_mean, np.finfo(float).tiny)),
                "median_relative_gain_db": float(group["oskar_relative_gain_db"].median()),
                "minimum_relative_gain_db": float(group["oskar_relative_gain_db"].min()),
                "maximum_relative_gain_db": float(group["oskar_relative_gain_db"].max()),
                "mean_azimuth_deg": float(group["azimuth_deg_north_through_east"].mean()),
                "mean_elevation_deg": float(group["elevation_deg"].mean()),
                "mean_ncp_separation_deg": float(group["ncp_separation_deg"].mean()),
                "mean_slant_range_km": float(group["slant_range_km"].mean()),
                "max_abs_oskar_minus_direct_formula_db": float(
                    group["oskar_minus_direct_formula_db"].abs().max()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary_csv = root / "results" / "tables" / "four_event_oskar_gain_summary.csv"
    summary.to_csv(summary_csv, index=False)
    return catalog, summary


def plot_horizon_beam(root: Path, fits_path: Path, samples: pd.DataFrame) -> Path:
    """Plot full ideal horizon beam and four event tracks."""
    with fits.open(fits_path) as hdul:
        image = np.asarray(hdul[0].data, dtype=float).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image after squeeze, got {image.shape}")
    gain_db = 10.0 * np.log10(np.maximum(image, np.finfo(float).tiny))
    gain_db = np.where(np.isfinite(image), gain_db, np.nan)

    # OSKAR's horizon image columns run from east direction cosine +1 to -1.
    # Flip columns so the displayed x coordinate increases from west to east.
    gain_db = np.fliplr(gain_db)
    fig, ax = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
    mesh = ax.imshow(
        gain_db,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        cmap="viridis",
        vmin=-60.0,
        vmax=0.0,
        interpolation="nearest",
    )
    horizon = plt.Circle((0.0, 0.0), 1.0, fill=False, color="white", linewidth=1.0)
    ax.add_patch(horizon)
    for event_id in EVENT_ORDER:
        group = samples[samples["event_id"] == event_id]
        ax.plot(
            group["enu_east"],
            group["enu_north"],
            color=EVENT_COLOURS[event_id],
            linewidth=2.3,
            label=event_id,
        )
        midpoint = group.iloc[len(group) // 2]
        ax.scatter(
            midpoint["enu_east"], midpoint["enu_north"],
            s=38, color=EVENT_COLOURS[event_id], edgecolor="white", linewidth=0.6,
        )
    lat = np.deg2rad(42.924240425222)
    ax.scatter(0.0, np.cos(lat), marker="*", s=150, color="white", edgecolor="black", label="NCP")
    ax.text(0.02, np.cos(lat) + 0.025, "NCP", color="white", fontsize=10)
    ax.set(
        xlim=(-1.02, 1.02),
        ylim=(-1.02, 1.02),
        aspect="equal",
        xlabel="East direction cosine",
        ylabel="North direction cosine",
        title="Ideal 21CMA E08 array-factor beam at 137.055 MHz",
    )
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.025)
    colorbar.set_label("Relative power gain (dB; NCP = 0 dB)")
    output = root / "results" / "figures" / "e08_oskar_horizon_beam_137055MHz.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_event_gains(root: Path, samples: pd.DataFrame) -> Path:
    """Plot exact OSKAR gain through each event-duration direction track."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for event_id in EVENT_ORDER:
        group = samples[samples["event_id"] == event_id]
        ax.plot(
            group["offset_from_window_start_s"],
            group["oskar_relative_gain_db"],
            color=EVENT_COLOURS[event_id],
            linewidth=2.0,
            label=event_id,
        )
    ax.set(
        xlabel="Time from predicted/observed preamble start (s)",
        ylabel="Relative power gain (dB; NCP = 0 dB)",
        title="Exact OSKAR gain along four Starlink-event direction tracks",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output = root / "results" / "figures" / "four_event_oskar_gain_timeseries.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_dir = root / "results" / "oskar_raw"
    image_path = find_single_output(raw_dir, "e08_horizon_137055*AUTO_POWER*.fits")
    sample_path = find_single_output(raw_dir, "e08_event_samples_137055*AUTO_POWER*.txt")
    samples, summary = summarise_exact_samples(root, sample_path)
    beam_png = plot_horizon_beam(root, image_path, samples)
    gain_png = plot_event_gains(root, samples)
    report = {
        "oskar_fits": str(image_path),
        "oskar_exact_sample_text": str(sample_path),
        "beam_figure": str(beam_png),
        "event_gain_figure": str(gain_png),
        "event_summary": summary.to_dict(orient="records"),
    }
    report_path = root / "results" / "oskar_result_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\nOSKAR four-event gain summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nBeam figure:  {beam_png}")
    print(f"Gain figure:  {gain_png}")
    print(f"Summary CSV:  {root / 'results' / 'tables' / 'four_event_oskar_gain_summary.csv'}")
    print(f"Full samples: {root / 'results' / 'tables' / 'four_event_oskar_gain_timeseries.csv'}")


if __name__ == "__main__":
    main()
