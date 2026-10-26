"""Estimate the currently measurable Starlink VHF apparent flux density.

Functions and outputs
---------------------
``load_event(summary_path)``
    Reads one LFdemod Starlink summary and returns the raw-file path, decoded
    packet boundaries, measured carrier offset, and event label.
``downconvert_analysis_span(event)``
    Runs LFdemod's existing Starlink VHF DDC over the packet plus the two
    background windows; returns 240 kS/s complex IQ and its file-relative
    start time.  This analysis does not modify LFdemod.
``band_power_timeseries(iq, ...)``
    Forms a linear two-sided PSD and integrates each time bin over the
    41.6667 kHz LoRa channel; returns time and channel-power arrays.
``measure_event(event)``
    Computes on/off power, excess ratio, equivalent antenna-temperature
    increase, main-beam-equivalent apparent spectral flux density, equivalent
    receiver-input signal power, and basic ADC-clipping diagnostics.
``main()``
    Processes both decoded events and writes ``apparent_flux_summary.csv``,
    ``apparent_flux_summary.json``, per-event power time-series CSV files, and
    ``channel_power_timeseries.png`` below ``results/5_flux_density``.

Interpretation
--------------
The reported flux is *not* corrected for the unknown E8 sidelobe response or
for unknown spacecraft/receiver polarisation coupling.  It is the flux an
unpolarised source would have in the nominal main beam to produce the measured
power ratio under the adopted single-pod SEFD.  The direction-corrected flux
is ``S_true = S_app / (B_rel * eta_pol)``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from lfdemod.starlink_vhf import StarlinkVHFDDCConfig, downconvert_starlink_vhf_voltage


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "5_flux_density" / "apparent_flux"
SUMMARY_PATHS = (
    ROOT / "results" / "2_demodulation" / "file0" / "summary.json",
    ROOT / "results" / "2_demodulation" / "file12" / "summary.json",
)

RF_FREQUENCY_HZ = 137_055_000.0
LORA_BANDWIDTH_HZ = 41_666.666666666664
BACKGROUND_GUARD_SECONDS = 0.30
BACKGROUND_SECONDS_PER_SIDE = 1.00
DDC_PADDING_SECONDS = 0.10
STFT_NPERSEG = 4096

# 21CMA single-pod values used in the existing sensitivity estimate.
EFFECTIVE_AREA_M2 = 218.0
REFERENCE_FREQUENCY_HZ = 150e6
REFERENCE_SKY_TEMPERATURE_K = 253.2
REFERENCE_SYSTEM_TEMPERATURE_K = 350.0
SKY_SPECTRAL_INDEX = -2.55
BOLTZMANN_J_PER_K = 1.380649e-23


@dataclass(frozen=True)
class Event:
    """Decoded event metadata needed by the power measurement."""

    label: str
    input_path: Path
    preamble_start_s: float
    packet_stop_s: float
    carrier_offset_hz: float


def load_event(summary_path: Path) -> Event:
    """Return the first decoded packet described by one LFdemod summary."""

    summary = json.loads(summary_path.read_text())
    packet = summary["packets"][0]
    return Event(
        label=summary_path.parent.name,
        input_path=Path(summary["input_path"]),
        preamble_start_s=float(packet["preamble_start_in_file_seconds"]),
        packet_stop_s=float(packet["packet_stop_in_file_seconds"]),
        carrier_offset_hz=float(packet["carrier_offset_hz"]),
    )


def analysis_windows(event: Event) -> dict[str, tuple[float, float]]:
    """Return packet, pre-background, and post-background time intervals."""

    pre_stop = event.preamble_start_s - BACKGROUND_GUARD_SECONDS
    post_start = event.packet_stop_s + BACKGROUND_GUARD_SECONDS
    return {
        "pre": (pre_stop - BACKGROUND_SECONDS_PER_SIDE, pre_stop),
        "on": (event.preamble_start_s, event.packet_stop_s),
        "post": (post_start, post_start + BACKGROUND_SECONDS_PER_SIDE),
    }


def downconvert_analysis_span(event: Event) -> tuple[np.ndarray, float, StarlinkVHFDDCConfig]:
    """Return 240 kS/s IQ covering all measurement windows and its start time."""

    windows = analysis_windows(event)
    span_start = windows["pre"][0]
    span_stop = windows["post"][1]
    padded_start = max(0.0, span_start - DDC_PADDING_SECONDS)
    padded_stop = span_stop + DDC_PADDING_SECONDS

    config = StarlinkVHFDDCConfig()
    dtype = np.dtype("<i2")
    raw = np.memmap(event.input_path, dtype=dtype, mode="r")
    file_duration = raw.size / config.fs_in
    if padded_stop > file_duration:
        raise ValueError(f"Analysis span exceeds {event.input_path}: {padded_stop} > {file_duration}")

    first = int(round(padded_start * config.fs_in))
    last = int(round(padded_stop * config.fs_in))
    initial_phase = float(
        np.remainder(2.0 * np.pi * RF_FREQUENCY_HZ * first / config.fs_in, 2.0 * np.pi)
    )
    padded_iq, _ = downconvert_starlink_vhf_voltage(
        raw[first:last], RF_FREQUENCY_HZ, config, initial_phase
    )
    trim = int(round((span_start - padded_start) * config.fs_out))
    count = int(round((span_stop - span_start) * config.fs_out))
    iq = np.asarray(padded_iq[trim : trim + count], dtype=np.complex64)
    if iq.size != count:
        raise RuntimeError("DDC output is shorter than the requested analysis span")
    return iq, span_start, config


def band_power_timeseries(
    iq: np.ndarray,
    sample_rate_hz: float,
    span_start_s: float,
    carrier_offset_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return file-relative time and linear power integrated over the LoRa band."""

    frequencies, times, psd = signal.spectrogram(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=0,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    psd = np.asarray(psd[order], dtype=np.float64)
    half_bandwidth = LORA_BANDWIDTH_HZ / 2.0
    selected = (
        (frequencies >= carrier_offset_hz - half_bandwidth)
        & (frequencies <= carrier_offset_hz + half_bandwidth)
    )
    if np.count_nonzero(selected) < 2:
        raise RuntimeError("No FFT bins fall inside the requested LoRa bandwidth")
    # NumPy 1.26 provides ``trapz``; ``trapezoid`` was added in NumPy 2.x.
    band_power = np.trapz(psd[selected], frequencies[selected], axis=0)
    return span_start_s + times, np.asarray(band_power, dtype=np.float64)


def adc_stats(event: Event) -> dict[str, float | int]:
    """Return ADC clipping indicators for the exact decoded packet interval."""

    fs = StarlinkVHFDDCConfig().fs_in
    raw = np.memmap(event.input_path, dtype="<i2", mode="r")
    first = int(round(event.preamble_start_s * fs))
    last = int(round(event.packet_stop_s * fs))
    chunk_samples = 10_000_000
    count = 0
    clipped = 0
    near_full_scale = 0
    minimum = 32767
    maximum = -32768
    sum_squares = 0.0
    threshold = 0.99 * 32768.0
    for start in range(first, last, chunk_samples):
        values = np.asarray(raw[start : min(start + chunk_samples, last)], dtype=np.int16)
        count += values.size
        clipped += int(np.count_nonzero((values == 32767) | (values == -32768)))
        near_full_scale += int(np.count_nonzero(np.abs(values.astype(np.int32)) >= threshold))
        minimum = min(minimum, int(values.min()))
        maximum = max(maximum, int(values.max()))
        floating = values.astype(np.float64)
        sum_squares += float(np.dot(floating, floating))
    return {
        "adc_sample_count": count,
        "adc_min": minimum,
        "adc_max": maximum,
        "adc_rms_counts": float(np.sqrt(sum_squares / count)),
        "adc_clip_fraction": clipped / count,
        "adc_near_full_scale_fraction": near_full_scale / count,
    }


def calibration_values() -> dict[str, float]:
    """Return the adopted 137.055 MHz Tsys and nominal-main-beam SEFD."""

    non_sky_temperature = REFERENCE_SYSTEM_TEMPERATURE_K - REFERENCE_SKY_TEMPERATURE_K
    sky_temperature = REFERENCE_SKY_TEMPERATURE_K * (
        RF_FREQUENCY_HZ / REFERENCE_FREQUENCY_HZ
    ) ** SKY_SPECTRAL_INDEX
    system_temperature = sky_temperature + non_sky_temperature
    sefd_jy = (
        2.0
        * BOLTZMANN_J_PER_K
        * system_temperature
        / EFFECTIVE_AREA_M2
        / 1e-26
    )
    reference_sefd_jy = (
        2.0
        * BOLTZMANN_J_PER_K
        * REFERENCE_SYSTEM_TEMPERATURE_K
        / EFFECTIVE_AREA_M2
        / 1e-26
    )
    return {
        "sky_temperature_137mhz_k": sky_temperature,
        "system_temperature_137mhz_k": system_temperature,
        "nominal_main_beam_sefd_137mhz_jy": sefd_jy,
        "published_assumption_sefd_150mhz_jy": reference_sefd_jy,
    }


def measure_event(event: Event) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return one event's calibrated apparent-flux result and power series."""

    iq, span_start, config = downconvert_analysis_span(event)
    times, powers = band_power_timeseries(
        iq, config.fs_out, span_start, event.carrier_offset_hz
    )
    windows = analysis_windows(event)
    masks = {
        name: (times >= limits[0]) & (times < limits[1])
        for name, limits in windows.items()
    }
    if any(np.count_nonzero(mask) == 0 for mask in masks.values()):
        raise RuntimeError(f"At least one measurement window is empty for {event.label}")

    on_values = powers[masks["on"]]
    pre_values = powers[masks["pre"]]
    post_values = powers[masks["post"]]
    off_values = np.concatenate((pre_values, post_values))
    on_power = float(np.mean(on_values))
    off_power = float(np.mean(off_values))
    off_median = float(np.median(off_values))
    excess_ratio = on_power / off_power - 1.0
    excess_ratio_median_off = on_power / off_median - 1.0

    # Formal standard error from non-overlapping 17.1 ms spectra.  It does not
    # include the dominant SEFD, beam, polarisation, or gain-linearity systematics.
    on_sem = float(np.std(on_values, ddof=1) / np.sqrt(on_values.size))
    off_sem = float(np.std(off_values, ddof=1) / np.sqrt(off_values.size))
    ratio_sem = float(
        np.sqrt((on_sem / off_power) ** 2 + (on_power * off_sem / off_power**2) ** 2)
    )

    calibration = calibration_values()
    tsys = calibration["system_temperature_137mhz_k"]
    sefd = calibration["nominal_main_beam_sefd_137mhz_jy"]
    delta_temperature = tsys * excess_ratio
    apparent_flux_jy = sefd * excess_ratio
    apparent_flux_sem_jy = sefd * ratio_sem
    signal_power_w = BOLTZMANN_J_PER_K * delta_temperature * LORA_BANDWIDTH_HZ
    signal_power_dbm = 10.0 * np.log10(signal_power_w / 1e-3)
    apparent_integrated_flux_w_m2 = apparent_flux_jy * 1e-26 * LORA_BANDWIDTH_HZ

    result: dict[str, Any] = {
        "event": event.label,
        "input_path": str(event.input_path),
        "preamble_start_s": event.preamble_start_s,
        "packet_stop_s": event.packet_stop_s,
        "packet_duration_s": event.packet_stop_s - event.preamble_start_s,
        "carrier_offset_hz": event.carrier_offset_hz,
        "observed_center_hz": RF_FREQUENCY_HZ + event.carrier_offset_hz,
        "analysis_bandwidth_hz": LORA_BANDWIDTH_HZ,
        "pre_background_start_s": windows["pre"][0],
        "pre_background_stop_s": windows["pre"][1],
        "post_background_start_s": windows["post"][0],
        "post_background_stop_s": windows["post"][1],
        "on_spectrum_count": int(on_values.size),
        "off_spectrum_count": int(off_values.size),
        "on_power_arbitrary": on_power,
        "off_power_arbitrary": off_power,
        "power_ratio_on_off": on_power / off_power,
        "excess_power_ratio": excess_ratio,
        "excess_power_ratio_formal_sem": ratio_sem,
        "excess_power_ratio_using_median_off": excess_ratio_median_off,
        "equivalent_antenna_temperature_increase_k": delta_temperature,
        "nominal_main_beam_apparent_flux_density_jy": apparent_flux_jy,
        "nominal_main_beam_apparent_flux_density_formal_sem_jy": apparent_flux_sem_jy,
        "nominal_main_beam_apparent_integrated_flux_w_m2": apparent_integrated_flux_w_m2,
        "equivalent_receiver_input_signal_power_w": signal_power_w,
        "equivalent_receiver_input_signal_power_dbm": signal_power_dbm,
        **adc_stats(event),
    }
    return result, times, powers


def write_outputs(
    results: list[dict[str, Any]],
    series: list[tuple[Event, np.ndarray, np.ndarray]],
) -> None:
    """Write structured tables and the two-event channel-power diagnostic plot."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "apparent_flux_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "method": {
            "on_window": "decoded preamble start through decoded packet stop",
            "off_window": "1.0 s before and after, each separated by a 0.30 s guard",
            "power_measurement": "linear Hann PSD integrated over 41.6667 kHz; non-overlapping 4096-sample spectra at 240 kS/s",
            "interpretation": "nominal-main-beam-equivalent apparent flux; no E8 sidelobe or polarisation correction",
            **calibration_values(),
            "effective_area_m2": EFFECTIVE_AREA_M2,
        },
        "events": results,
    }
    (OUTPUT_DIR / "apparent_flux_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, (event, times, powers), result in zip(axes, series, results):
        relative_db = 10.0 * np.log10(powers / result["off_power_arbitrary"])
        axis.plot(times, relative_db, linewidth=0.9, color="black")
        windows = analysis_windows(event)
        axis.axvspan(*windows["pre"], color="tab:blue", alpha=0.13, label="background")
        axis.axvspan(*windows["on"], color="tab:red", alpha=0.13, label="decoded packet")
        axis.axvspan(*windows["post"], color="tab:blue", alpha=0.13)
        axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{event.label}: excess={result['excess_power_ratio']:.3f}, "
            f"S_app={result['nominal_main_beam_apparent_flux_density_jy'] / 1e3:.2f} kJy"
        )
        axis.set_xlabel("File-relative time (s)")
        axis.set_ylabel("41.667 kHz power / off (dB)")
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
        table_path = OUTPUT_DIR / f"{event.label}_channel_power_timeseries.csv"
        with table_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("file_time_s", "band_power_arbitrary", "relative_to_off_db"))
            writer.writerows(zip(times, powers, relative_db))
    fig.savefig(OUTPUT_DIR / "channel_power_timeseries.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Measure both events and write all reproducible outputs."""

    events = [load_event(path) for path in SUMMARY_PATHS]
    results: list[dict[str, Any]] = []
    series: list[tuple[Event, np.ndarray, np.ndarray]] = []
    for event in events:
        print(f"Processing {event.label}: {event.input_path}", flush=True)
        result, times, powers = measure_event(event)
        results.append(result)
        series.append((event, times, powers))
        print(
            f"  excess={result['excess_power_ratio']:.6g}; "
            f"S_app={result['nominal_main_beam_apparent_flux_density_jy']:.6g} Jy",
            flush=True,
        )
    write_outputs(results, series)
    print(f"Wrote results to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
