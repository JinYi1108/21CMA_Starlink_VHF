#!/usr/bin/env python3
"""Classify and plot the long-term orbits of the 137.05-MHz emitters.

This script combines the emission segments extracted from the Grigg et al.
(2025) catalogue with the long-term Space-Track GP history downloaded by
``download_spacetrack_full_history_137p05_emitters.py``.  The primary unit of
analysis is a satellite--UTC-date combination, rather than an individual image
or polarisation record.

The phase labels are deliberately preliminary and are accompanied by all
continuous trend metrics needed for manual verification.  Four satellite
histories are plotted per page, with the 137.05-MHz detection epochs marked.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import theilslopes


EARTH_EQUATORIAL_RADIUS_KM = 6378.137
TREND_HALF_WINDOWS_DAYS = (14,)


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    history_dir = (
        project_root
        / "data"
        / "external"
        / "spacetrack"
        / "full_137p05_emitters"
    )
    result_dir = (
        project_root / "results" / "5_flux_density" / "chapter5_flux_density"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Measure orbital trends at the 137.05-MHz detections and plot "
            "the complete downloaded history of all 26 emitters."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=history_dir / "grigg2025_full_137p05_unique_gp_epochs.csv",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=result_dir / "grigg2025_137p05_emission_segments.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=result_dir)
    parser.add_argument(
        "--stable-rate-threshold",
        type=float,
        default=0.2,
        help=(
            "Absolute 14-day slope threshold separating a preliminary "
            "stable classification from orbit raising or lowering, in "
            "km/day (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--plots-per-page",
        type=int,
        choices=(2, 4),
        default=4,
        help="Number of satellite panels on each figure page (default: 4)",
    )
    return parser.parse_args()


def read_inputs(
    history_path: Path, segments_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (history_path, segments_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    history = pd.read_csv(history_path, low_memory=False)
    required_history = {
        "NORAD_CAT_ID",
        "EPOCH",
        "SEMIMAJOR_AXIS",
        "OBJECT_NAME",
        "LAUNCH_DATE",
    }
    missing = required_history.difference(history.columns)
    if missing:
        raise ValueError(f"History table lacks columns: {sorted(missing)}")
    history["NORAD_CAT_ID"] = pd.to_numeric(
        history["NORAD_CAT_ID"], errors="raise"
    ).astype(int)
    history["EPOCH"] = pd.to_datetime(
        history["EPOCH"], utc=True, format="mixed", errors="raise"
    )
    history["SEMIMAJOR_AXIS"] = pd.to_numeric(
        history["SEMIMAJOR_AXIS"], errors="coerce"
    )
    history["mean_altitude_km"] = (
        history["SEMIMAJOR_AXIS"] - EARTH_EQUATORIAL_RADIUS_KM
    )
    history["epoch_day_utc"] = history["EPOCH"].dt.normalize()
    history = history.dropna(subset=["mean_altitude_km"]).sort_values(
        ["NORAD_CAT_ID", "EPOCH"]
    )

    segments = pd.read_csv(segments_path)
    required_segments = {
        "norad_id",
        "segment_start_utc",
        "segment_end_utc",
    }
    missing = required_segments.difference(segments.columns)
    if missing:
        raise ValueError(f"Segment table lacks columns: {sorted(missing)}")
    segments["norad_id"] = pd.to_numeric(
        segments["norad_id"], errors="raise"
    ).astype(int)
    for column in ("segment_start_utc", "segment_end_utc"):
        segments[column] = pd.to_datetime(
            segments[column], utc=True, format="mixed", errors="raise"
        )
    segments["observing_date"] = segments["segment_start_utc"].dt.normalize()
    return history, segments


def make_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.groupby(["NORAD_CAT_ID", "epoch_day_utc"], as_index=False)
        .agg(
            object_name=("OBJECT_NAME", "last"),
            number_of_gp_epochs=("EPOCH", "size"),
            first_gp_epoch=("EPOCH", "min"),
            last_gp_epoch=("EPOCH", "max"),
            median_mean_altitude_km=("mean_altitude_km", "median"),
            minimum_mean_altitude_km=("mean_altitude_km", "min"),
            maximum_mean_altitude_km=("mean_altitude_km", "max"),
        )
        .rename(columns={"NORAD_CAT_ID": "norad_id"})
        .sort_values(["norad_id", "epoch_day_utc"])
        .reset_index(drop=True)
    )


def make_detection_combinations(segments: pd.DataFrame) -> pd.DataFrame:
    combinations = (
        segments.groupby(["norad_id", "observing_date"], as_index=False)
        .agg(
            number_of_emission_segments=("segment_start_utc", "size"),
            first_segment_utc=("segment_start_utc", "min"),
            last_segment_utc=("segment_end_utc", "max"),
        )
        .sort_values(["observing_date", "norad_id"])
        .reset_index(drop=True)
    )
    combinations.insert(0, "combination_id", np.arange(1, len(combinations) + 1))
    combinations["reference_utc"] = combinations["first_segment_utc"] + (
        combinations["last_segment_utc"] - combinations["first_segment_utc"]
    ) / 2
    return combinations


def robust_slope(
    satellite_daily: pd.DataFrame,
    reference_utc: pd.Timestamp,
    half_window_days: int,
) -> dict[str, object]:
    reference_day = reference_utc.normalize()
    selected = satellite_daily[
        satellite_daily["epoch_day_utc"].between(
            reference_day - pd.Timedelta(days=half_window_days),
            reference_day + pd.Timedelta(days=half_window_days),
        )
    ].dropna(subset=["median_mean_altitude_km"])
    before = int((selected["epoch_day_utc"] < reference_day).sum())
    after = int((selected["epoch_day_utc"] > reference_day).sum())
    if before and after:
        coverage = "TWO-SIDED"
    elif before:
        coverage = "PRE-ONLY"
    elif after:
        coverage = "POST-ONLY"
    elif len(selected):
        coverage = "REFERENCE-DAY ONLY"
    else:
        coverage = "NO DATA"

    slope = np.nan
    if len(selected) >= 3:
        x_days = (
            selected["epoch_day_utc"] - reference_day
        ).dt.total_seconds().to_numpy() / 86400.0
        slope = float(
            theilslopes(
                selected["median_mean_altitude_km"].to_numpy(float), x_days
            ).slope
        )
    return {
        "slope": slope,
        "number_of_days": len(selected),
        "pre_days": before,
        "post_days": after,
        "coverage": coverage,
    }


def interpolate_event_altitude(
    satellite_history: pd.DataFrame, reference_utc: pd.Timestamp
) -> dict[str, object]:
    preceding = satellite_history[
        satellite_history["EPOCH"].le(reference_utc)
    ].tail(1)
    following = satellite_history[
        satellite_history["EPOCH"].ge(reference_utc)
    ].head(1)
    if preceding.empty:
        return {
            "event_mean_altitude_km": np.nan,
            "gp_bracket_status": "NO PRE-DETECTION GP",
            "preceding_gp_epoch": pd.NaT,
            "following_gp_epoch": (
                following["EPOCH"].iloc[0] if not following.empty else pd.NaT
            ),
            "bracket_duration_hours": np.nan,
            "bracket_rate_km_per_day": np.nan,
        }
    if following.empty:
        return {
            "event_mean_altitude_km": np.nan,
            "gp_bracket_status": "NO POST-DETECTION GP",
            "preceding_gp_epoch": preceding["EPOCH"].iloc[0],
            "following_gp_epoch": pd.NaT,
            "bracket_duration_hours": np.nan,
            "bracket_rate_km_per_day": np.nan,
        }

    t0 = preceding["EPOCH"].iloc[0]
    t1 = following["EPOCH"].iloc[0]
    h0 = float(preceding["mean_altitude_km"].iloc[0])
    h1 = float(following["mean_altitude_km"].iloc[0])
    duration_seconds = (t1 - t0).total_seconds()
    if duration_seconds <= 0:
        return {
            "event_mean_altitude_km": np.nan,
            "gp_bracket_status": "INVALID BRACKET",
            "preceding_gp_epoch": t0,
            "following_gp_epoch": t1,
            "bracket_duration_hours": duration_seconds / 3600.0,
            "bracket_rate_km_per_day": np.nan,
        }
    fraction = (reference_utc - t0).total_seconds() / duration_seconds
    return {
        "event_mean_altitude_km": h0 + fraction * (h1 - h0),
        "gp_bracket_status": "BRACKETED",
        "preceding_gp_epoch": t0,
        "following_gp_epoch": t1,
        "bracket_duration_hours": duration_seconds / 3600.0,
        "bracket_rate_km_per_day": (h1 - h0) / (duration_seconds / 86400.0),
    }


def median_altitude_in_relative_window(
    satellite_daily: pd.DataFrame,
    reference_utc: pd.Timestamp,
    start_day: int,
    end_day: int,
) -> tuple[float, int]:
    reference_day = reference_utc.normalize()
    selected = satellite_daily[
        satellite_daily["epoch_day_utc"].between(
            reference_day + pd.Timedelta(days=start_day),
            reference_day + pd.Timedelta(days=end_day),
        )
    ]["median_mean_altitude_km"].dropna()
    return (
        float(selected.median()) if len(selected) else np.nan,
        int(len(selected)),
    )


def preliminary_phase(
    row: dict[str, object],
    stable_threshold: float,
) -> str:
    slope_14 = row["trend_14d_km_per_day"]
    if not np.isfinite(slope_14):
        return "uncertain"

    if slope_14 > stable_threshold:
        return "orbit raising"
    if abs(slope_14) <= stable_threshold:
        return "stable orbital phase"
    if slope_14 < -stable_threshold:
        return "orbit lowering"
    raise RuntimeError("Unreachable orbital-phase classification")


def build_phase_table(
    history: pd.DataFrame,
    daily: pd.DataFrame,
    combinations: pd.DataFrame,
    stable_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for combination in combinations.itertuples(index=False):
        satellite_history = history[
            history["NORAD_CAT_ID"].eq(combination.norad_id)
        ].sort_values("EPOCH")
        satellite_daily = daily[
            daily["norad_id"].eq(combination.norad_id)
        ].sort_values("epoch_day_utc")
        if satellite_history.empty:
            continue

        launch_date = pd.to_datetime(
            satellite_history["LAUNCH_DATE"].dropna().iloc[-1], utc=True
        )
        decay_values = satellite_history.get(
            "DECAY_DATE", pd.Series(dtype=object)
        ).dropna()
        decay_date = decay_values.iloc[-1] if len(decay_values) else None
        object_name = satellite_history["OBJECT_NAME"].dropna().iloc[-1]
        interpolated = interpolate_event_altitude(
            satellite_history, combination.reference_utc
        )
        event_altitude = interpolated["event_mean_altitude_km"]
        future_median, future_days = median_altitude_in_relative_window(
            satellite_daily, combination.reference_utc, 30, 90
        )
        preceding_median, preceding_days = median_altitude_in_relative_window(
            satellite_daily, combination.reference_utc, -60, -15
        )

        row: dict[str, object] = {
            "combination_id": combination.combination_id,
            "norad_id": combination.norad_id,
            "object_name": object_name,
            "launch_date": launch_date.date().isoformat(),
            "decay_date": decay_date,
            "observing_date": combination.observing_date.date().isoformat(),
            "reference_utc": combination.reference_utc,
            "number_of_emission_segments": combination.number_of_emission_segments,
            "age_since_launch_days": (
                combination.reference_utc - launch_date
            ).total_seconds()
            / 86400.0,
            **interpolated,
            "preceding_60_15d_median_altitude_km": preceding_median,
            "preceding_60_15d_number_of_days": preceding_days,
            "future_30_90d_median_altitude_km": future_median,
            "future_30_90d_number_of_days": future_days,
            "future_30_90d_shift_km": (
                future_median - event_altitude
                if np.isfinite(future_median) and np.isfinite(event_altitude)
                else np.nan
            ),
        }
        for half_window in TREND_HALF_WINDOWS_DAYS:
            result = robust_slope(
                satellite_daily, combination.reference_utc, half_window
            )
            prefix = f"trend_{half_window}d"
            row[f"{prefix}_km_per_day"] = result["slope"]
            row[f"{prefix}_number_of_days"] = result["number_of_days"]
            row[f"{prefix}_pre_days"] = result["pre_days"]
            row[f"{prefix}_post_days"] = result["post_days"]
            row[f"{prefix}_coverage"] = result["coverage"]

        row["preliminary_orbital_phase"] = preliminary_phase(
            row,
            stable_threshold,
        )
        row["classification_coverage"] = (
            "two-sided"
            if row["trend_14d_coverage"] == "TWO-SIDED"
            else "limited: " + str(row["trend_14d_coverage"]).lower()
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["observing_date", "norad_id"]
    ).reset_index(drop=True)


def plot_lifecycle_pages(
    history: pd.DataFrame,
    daily: pd.DataFrame,
    phase_table: pd.DataFrame,
    output_dir: Path,
    plots_per_page: int,
) -> tuple[Path, list[Path]]:
    satellites = sorted(phase_table["norad_id"].unique())
    if plots_per_page == 4:
        nrows, ncols, figsize = 2, 2, (14, 9)
    else:
        nrows, ncols, figsize = 2, 1, (12, 9)

    pdf_path = output_dir / "grigg2025_137p05_orbit_lifecycle_panels.pdf"
    png_paths: list[Path] = []
    phase_colours = {
        "stable orbital phase": "#2ca02c",
        "orbit raising": "#d95f02",
        "orbit lowering": "#7b3294",
        "uncertain": "#666666",
    }

    with PdfPages(pdf_path) as pdf:
        number_of_pages = math.ceil(len(satellites) / plots_per_page)
        for page_number in range(number_of_pages):
            page_ids = satellites[
                page_number * plots_per_page : (page_number + 1) * plots_per_page
            ]
            fig, axes = plt.subplots(
                nrows, ncols, figsize=figsize, constrained_layout=True
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, norad_id in zip(axes, page_ids):
                satellite_daily = daily[daily["norad_id"].eq(norad_id)]
                satellite_history = history[
                    history["NORAD_CAT_ID"].eq(norad_id)
                ]
                detections = phase_table[phase_table["norad_id"].eq(norad_id)]
                object_name = satellite_daily["object_name"].dropna().iloc[-1]
                phases = ", ".join(detections["preliminary_orbital_phase"].unique())
                phase_colour = phase_colours.get(
                    detections["preliminary_orbital_phase"].iloc[0], "#666666"
                )

                x = satellite_daily["epoch_day_utc"]
                median = satellite_daily["median_mean_altitude_km"].to_numpy(float)
                lower = satellite_daily["minimum_mean_altitude_km"].to_numpy(float)
                upper = satellite_daily["maximum_mean_altitude_km"].to_numpy(float)
                ax.fill_between(
                    x, lower, upper, color="#bdbdbd", alpha=0.35, linewidth=0
                )
                ax.plot(x, median, color="#2166ac", linewidth=1.25, label="Daily median")

                for detection_number, detection in enumerate(
                    detections.itertuples(index=False), start=1
                ):
                    ax.axvline(
                        detection.reference_utc,
                        color="#d73027",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.9,
                    )
                    if np.isfinite(detection.event_mean_altitude_km):
                        ax.scatter(
                            detection.reference_utc,
                            detection.event_mean_altitude_km,
                            marker="*",
                            s=75,
                            color="#d73027",
                            edgecolor="black",
                            linewidth=0.4,
                            zorder=5,
                        )
                    ax.annotate(
                        detection.observing_date,
                        xy=(detection.reference_utc, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, -4 - 13 * (detection_number - 1)),
                        textcoords="offset points",
                        rotation=90,
                        va="top",
                        ha="left",
                        color="#b2182b",
                        fontsize=7,
                    )

                decay_values = satellite_history.get(
                    "DECAY_DATE", pd.Series(dtype=object)
                ).dropna()
                if len(decay_values):
                    decay_date = pd.to_datetime(decay_values.iloc[-1], utc=True)
                    ax.axvline(
                        decay_date,
                        color="black",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.75,
                    )
                    ax.annotate(
                        "Decay",
                        xy=(decay_date, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(-3, 2),
                        textcoords="offset points",
                        rotation=90,
                        va="bottom",
                        ha="right",
                        fontsize=7,
                    )

                ax.set_title(
                    f"{object_name} (NORAD {norad_id})\n{phases}",
                    fontsize=10,
                    color=phase_colour,
                )
                ax.set_ylabel("Mean orbital altitude (km)")
                ax.grid(True, alpha=0.25, linewidth=0.6)
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                ax.margins(x=0.01)

            for ax in axes[len(page_ids) :]:
                ax.set_visible(False)
            for ax in axes[-ncols:]:
                if ax.get_visible():
                    ax.set_xlabel("UTC date")

            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            png_path = output_dir / (
                f"grigg2025_137p05_orbit_lifecycle_page_{page_number + 1:02d}.png"
            )
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            png_paths.append(png_path)
            plt.close(fig)

    return pdf_path, png_paths


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, segments = read_inputs(args.history, args.segments)
    daily = make_daily_history(history)
    combinations = make_detection_combinations(segments)
    phase_table = build_phase_table(
        history,
        daily,
        combinations,
        args.stable_rate_threshold,
    )

    daily_output = args.output_dir / "grigg2025_137p05_full_daily_orbit_history.csv"
    phase_output = args.output_dir / "grigg2025_137p05_orbital_phase_summary.csv"
    daily.to_csv(daily_output, index=False)
    phase_table.to_csv(phase_output, index=False)
    pdf_output, png_outputs = plot_lifecycle_pages(
        history, daily, phase_table, args.output_dir, args.plots_per_page
    )

    print(f"Unique GP epochs: {len(history)}")
    print(f"Unique satellites: {history['NORAD_CAT_ID'].nunique()}")
    print(f"Satellite-date combinations: {len(phase_table)}")
    print("\nPreliminary orbital-phase counts:")
    print(phase_table["preliminary_orbital_phase"].value_counts().to_string())
    print("\nClassification table:")
    display_columns = [
        "norad_id",
        "object_name",
        "observing_date",
        "age_since_launch_days",
        "event_mean_altitude_km",
        "trend_14d_km_per_day",
        "future_30_90d_shift_km",
        "preliminary_orbital_phase",
        "classification_coverage",
    ]
    print(phase_table[display_columns].to_string(index=False))
    print("\nSaved daily history:", daily_output)
    print("Saved phase summary:", phase_output)
    print("Saved multi-page PDF:", pdf_output)
    print(f"Saved {len(png_outputs)} PNG pages:")
    for path in png_outputs:
        print(" ", path)


if __name__ == "__main__":
    main()
