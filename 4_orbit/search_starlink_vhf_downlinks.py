"""Search five coordinated Starlink VHF downlink channels in 21CMA data.

This project-specific program deliberately separates a fast, sensitive energy
search from the existing, reusable LFdemod Starlink VHF decoder.

Main functions and outputs
--------------------------
``build_inventory(data_root, datasets, ...)``
    Finds numbered ``.dat`` files, checks sizes and numbering, and returns a
    continuous per-dataset time index.  ``run_scan`` writes the result to
    ``dataset_inventory.csv``.
``scan_one_file(file_record, search_config)``
    Opens one raw file read-only.  Every 25 ms it takes one 65,536-sample Hann
    FFT (0.1365 ms at 480 MS/s), evaluates all five RF channels from that one
    FFT, and returns only samples at least 4 dB above the file/channel median.
    These short FFTs find candidates; they are never used as LoRa packets.
``form_candidates(hits, search_config)``
    Groups nearby hits and accepts a group by any of three rules: one strong
    sample, several persistent weak samples, or two medium samples.  It also
    records the coarse peak-frequency span needed by the next morphology
    filter; it writes no dense sparse-sample table.
``classify_sparse_morphology(candidates, search_config)``
    Separates frequency-sweeping candidates from fixed narrow tones using the
    sparse FFT peak track; returns ``(retained, rejected)`` audit rows.
``probe_candidate_chirp_morphology(candidate, inventory, search_config, ...)``
    Reads at most four 20 ms raw-voltage windows, forms a dense local STFT,
    and scores tracks near the expected Starlink-LoRa upchirp slope without
    running DDC; returns one compact morphology row and optionally a PNG.
``probe_candidate_preamble(candidate, inventory, search_config)``
    Reads a short window, calls LFdemod's public DDC/resampler, then uses the
    pinned lora-phy detector to test 15 preamble upchirps, SFD synchronization,
    and the Starlink ``0x12`` sync word.  It returns one aggregate probe row;
    it does not decode a payload or claim a Starlink frame.
``verify_candidates(...)``
    Builds a multi-second window around every candidate and calls
    ``lfdemod.starlink_vhf.demodulate_starlink_vhf_file``.  File-boundary
    windows are temporarily stitched from adjacent read-only source files.
    Rejected energy candidates leave only one aggregate CSV row.  Full LFdemod
    output is retained only for strict Starlink-profile detections and likely
    LoRa packets.
``deduplicate_packet_rows(rows, ...)``
    Collapses the same physical packet when overlapping 137.8463/137.8763 MHz
    searches or adjacent candidates decode it more than once.
``run_scan(args)``
    Writes inventory, configuration, candidates, candidate overview, and a
    scan summary beneath the user-selected output directory.
``run_verify(args)``
    Writes aggregate verification results, de-duplicated detections, likely
    events, a final overview, and a machine-readable final summary.  Detailed
    decoder artifacts are kept only under ``detailed/strict`` or
    ``detailed/likely``.

The program never modifies raw observations.  DDC and LoRa resampling are
delegated to the installed LFdemod package.  The preamble probe intentionally
uses the private detection/synchronization methods of pinned ``lora-phy==0.3.0``
so that a short window can stop before payload decoding; the final CRC and
Starlink frame decision remain delegated to LFdemod.  Injection/recovery
completeness tests are explicitly outside this first search version and are
documented as future work in ``TECHNICAL_METHOD.md``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from lfdemod.starlink_vhf import (
    StarlinkVHFDDCConfig,
    StarlinkVHFFileConfig,
    StarlinkVHFLoRaConfig,
    demodulate_starlink_vhf_file,
    downconvert_starlink_vhf_voltage,
    prepare_lora_iq,
    starlink_sync_symbol_offsets,
)


DEFAULT_SAMPLE_RATE_HZ = 480e6
DEFAULT_DTYPE = "<i2"
DEFAULT_DATASETS = (
    "202504151940",
    "202504152100",
    "202504152200",
    "202504160036",
    "202504160735",
)
STARLINK_DOWNLINK_HZ = (
    137.055e6,
    137.145e6,
    137.8463e6,
    137.8763e6,
    137.9663e6,
)
NUMBER_RE = re.compile(r"-(\d+)\.dat$")

# These two already decoded packets are used only as a search-regression
# check.  They do not participate in threshold estimation or classification.
KNOWN_SEARCH_CHECKS = (
    ("202504152200", 0, 5.330904, 137.055e6),
    ("202504152200", 12, 2.930004, 137.055e6),
)


@dataclass(frozen=True)
class SearchConfig:
    """Parameters for candidate finding, not for LoRa demodulation."""

    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    dtype: str = DEFAULT_DTYPE
    channels_hz: tuple[float, ...] = STARLINK_DOWNLINK_HZ
    nfft: int = 65_536
    step_seconds: float = 0.025
    target_half_width_hz: float = 35_000.0
    reference_low_hz: float = 136.98e6
    reference_high_hz: float = 138.04e6
    reference_exclusion_half_width_hz: float = 50_000.0
    retained_hit_floor_db: float = 4.0
    strong_threshold_db: float = 8.0
    persistent_threshold_db: float = 4.0
    persistent_count: int = 3
    persistent_span_seconds: float = 1.2
    medium_threshold_db: float = 6.0
    medium_count: int = 2
    medium_span_seconds: float = 0.8
    maximum_hit_gap_seconds: float = 0.6
    morphology_minimum_unique_peak_bins: int = 4
    morphology_minimum_peak_span_hz: float = 25_000.0
    chirp_probe_window_seconds: float = 0.020
    chirp_probe_max_windows: int = 4
    chirp_stft_nfft: int = 131_072
    chirp_stft_hop: int = 65_536
    chirp_stft_half_width_hz: float = 60_000.0
    chirp_track_duration_seconds: float = 0.0030
    chirp_expected_slope_hz_per_second: float = 6.781684e6
    chirp_slope_tolerance_fraction: float = 0.30
    # Empirical current-dataset gates.  Event 2, the weaker validated packet,
    # gives approximately (2.05 z, 0.55 occupancy, 1.19 z margin), while the
    # tested fixed-line/non-LoRa controls remain below (0.79, 0.10, 0.13).
    # These are compute-saving prefilter values, not completeness limits.
    chirp_minimum_track_score_z: float = 1.5
    chirp_minimum_track_occupancy: float = 0.40
    chirp_minimum_control_margin_z: float = 0.60
    preamble_probe_guard_seconds: float = 0.35
    preamble_probe_minimum_seconds: float = 1.6
    verification_window_seconds: float = 5.5
    deduplication_time_tolerance_seconds: float = 0.025

    def validate(self) -> None:
        """Raise ``ValueError`` when search parameters are inconsistent."""

        if self.sample_rate_hz <= 0 or self.nfft <= 0:
            raise ValueError("sample_rate_hz and nfft must be positive.")
        if self.nfft % 2:
            raise ValueError("nfft must be even.")
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive.")
        if not self.channels_hz:
            raise ValueError("At least one RF channel is required.")
        if not 0 < min(self.channels_hz) < max(self.channels_hz) < self.sample_rate_hz / 2:
            raise ValueError("RF channels must lie inside the real-sampled Nyquist band.")
        if not self.reference_low_hz < self.reference_high_hz:
            raise ValueError("reference_low_hz must be below reference_high_hz.")
        if self.retained_hit_floor_db > min(
            self.strong_threshold_db,
            self.persistent_threshold_db,
            self.medium_threshold_db,
        ):
            raise ValueError("retained_hit_floor_db would discard qualifying hits.")
        if self.morphology_minimum_unique_peak_bins < 2:
            raise ValueError("morphology_minimum_unique_peak_bins must be at least two.")
        if self.morphology_minimum_peak_span_hz <= 0:
            raise ValueError("morphology_minimum_peak_span_hz must be positive.")
        if self.chirp_probe_window_seconds <= 0:
            raise ValueError("chirp_probe_window_seconds must be positive.")
        if self.chirp_probe_max_windows <= 0:
            raise ValueError("chirp_probe_max_windows must be positive.")
        if self.chirp_stft_nfft <= 0 or self.chirp_stft_nfft % 2:
            raise ValueError("chirp_stft_nfft must be a positive even integer.")
        if not 0 < self.chirp_stft_hop <= self.chirp_stft_nfft:
            raise ValueError("chirp_stft_hop must lie in (0, chirp_stft_nfft].")
        if self.chirp_stft_half_width_hz <= 0:
            raise ValueError("chirp_stft_half_width_hz must be positive.")
        if self.chirp_track_duration_seconds <= 0:
            raise ValueError("chirp_track_duration_seconds must be positive.")
        if self.chirp_expected_slope_hz_per_second <= 0:
            raise ValueError("chirp_expected_slope_hz_per_second must be positive.")
        if not 0 <= self.chirp_slope_tolerance_fraction < 1:
            raise ValueError("chirp_slope_tolerance_fraction must lie in [0, 1).")
        if not 0 <= self.chirp_minimum_track_occupancy <= 1:
            raise ValueError("chirp_minimum_track_occupancy must lie in [0, 1].")
        if self.chirp_minimum_track_score_z < 0:
            raise ValueError("chirp_minimum_track_score_z must be non-negative.")
        if self.chirp_minimum_control_margin_z < 0:
            raise ValueError("chirp_minimum_control_margin_z must be non-negative.")
        if self.preamble_probe_guard_seconds < 0:
            raise ValueError("preamble_probe_guard_seconds must be non-negative.")
        if self.preamble_probe_minimum_seconds <= 0:
            raise ValueError("preamble_probe_minimum_seconds must be positive.")


INVENTORY_FIELDS = (
    "dataset",
    "file_number",
    "file_name",
    "input_path",
    "file_bytes",
    "sample_count",
    "duration_s",
    "dataset_start_s",
    "dataset_stop_s",
    "segment_id",
    "contiguous_with_previous",
    "mtime_utc",
)

CANDIDATE_FIELDS = (
    "candidate_id",
    "dataset",
    "segment_id",
    "rf_frequency_hz",
    "peak_dataset_time_s",
    "peak_file_number",
    "peak_file_time_s",
    "peak_excess_db",
    "peak_frequency_hz_coarse",
    "first_hit_dataset_time_s",
    "last_hit_dataset_time_s",
    "hit_count_ge_4db",
    "hit_count_ge_6db",
    "hit_count_ge_8db",
    "hit_span_s",
    "unique_peak_bin_count",
    "peak_frequency_span_hz",
    "peak_frequency_std_hz",
    "chirp_probe_dataset_times_s",
    "qualification_rules",
    "source_file_numbers",
)

CHIRP_MORPHOLOGY_FIELDS = (
    "candidate_id",
    "dataset",
    "rf_frequency_hz",
    "tested_window_count",
    "best_window_dataset_time_s",
    "best_slope_hz_per_second",
    "best_slope_khz_per_ms",
    "best_track_score_z",
    "best_track_occupancy",
    "best_horizontal_track_score_z",
    "best_negative_track_score_z",
    "control_margin_z",
    "frequency_sweep_span_hz",
    "chirp_morphology_pass",
    "classification",
    "diagnostic_path",
    "error",
)

PREAMBLE_PROBE_FIELDS = (
    "candidate_id",
    "dataset",
    "rf_frequency_hz",
    "probe_dataset_start_s",
    "probe_duration_s",
    "source_segments",
    "edge_limited",
    "preamble_detected_count",
    "sfd_synchronized_count",
    "starlink_sync_match_count",
    "first_preamble_dataset_time_s",
    "first_carrier_offset_hz",
    "first_sync_symbol_1",
    "first_sync_symbol_2",
    "expected_sync_symbol_1",
    "expected_sync_symbol_2",
    "classification",
    "error",
)

VERIFICATION_FIELDS = (
    "candidate_id",
    "dataset",
    "searched_rf_frequency_hz",
    "classification",
    "packet_count",
    "crc_valid_packet_count",
    "starlink_profile_match_count",
    "verification_dataset_start_s",
    "verification_duration_s",
    "source_segments",
    "edge_limited",
    "detailed_output_dir",
    "error",
)

PACKET_FIELDS = (
    "event_id",
    "candidate_id",
    "dataset",
    "classification",
    "searched_rf_frequency_hz",
    "assigned_channel_hz",
    "preamble_dataset_time_s",
    "packet_stop_dataset_time_s",
    "carrier_offset_hz",
    "observed_center_hz",
    "payload_bytes",
    "symbol_count",
    "header_valid",
    "crc_valid",
    "starlink_profile_match",
    "spacecraft_id_u16_le",
    "message_number_u24_le",
    "protocol_id_u16_le",
    "payload_hex",
    "detailed_output_dir",
)


def _format_frequency_mhz(value_hz: float) -> str:
    """Return a filesystem-safe MHz label."""

    return f"{value_hz / 1e6:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write flat dictionaries with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    """Write indented UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def numbered_dat_files(dataset_dir: Path) -> list[tuple[int, Path]]:
    """Return numbered DAT files in numerical, not lexical, order."""

    files: list[tuple[int, Path]] = []
    for path in dataset_dir.glob("*.dat"):
        match = NUMBER_RE.search(path.name)
        if match:
            files.append((int(match.group(1)), path.resolve()))
    return sorted(files, key=lambda item: item[0])


def build_inventory(
    data_root: Path,
    datasets: Sequence[str],
    config: SearchConfig,
) -> list[dict[str, Any]]:
    """Return checked file metadata and a cumulative timeline per dataset."""

    dtype = np.dtype(config.dtype)
    inventory: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_dir = (data_root / dataset).resolve()
        files = numbered_dat_files(dataset_dir)
        if not files:
            raise FileNotFoundError(f"No numbered .dat files found in {dataset_dir}")
        dataset_time_s = 0.0
        previous_number: int | None = None
        segment_id = 0
        for file_number, path in files:
            file_bytes = path.stat().st_size
            if file_bytes % dtype.itemsize:
                raise ValueError(f"File size is not an integer number of samples: {path}")
            sample_count = file_bytes // dtype.itemsize
            duration_s = sample_count / config.sample_rate_hz
            contiguous = previous_number is None or file_number == previous_number + 1
            if previous_number is not None and not contiguous:
                segment_id += 1
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            inventory.append(
                {
                    "dataset": dataset,
                    "file_number": file_number,
                    "file_name": path.name,
                    "input_path": str(path),
                    "file_bytes": file_bytes,
                    "sample_count": sample_count,
                    "duration_s": duration_s,
                    "dataset_start_s": dataset_time_s,
                    "dataset_stop_s": dataset_time_s + duration_s,
                    "segment_id": segment_id,
                    "contiguous_with_previous": contiguous,
                    "mtime_utc": mtime.isoformat(),
                }
            )
            dataset_time_s += duration_s
            previous_number = file_number
    return inventory


def _search_masks(config: SearchConfig) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Return FFT frequencies, target-bin indices, and reference mask."""

    frequencies = np.fft.rfftfreq(config.nfft, d=1.0 / config.sample_rate_hz)
    targets = [
        np.flatnonzero(np.abs(frequencies - channel) <= config.target_half_width_hz)
        for channel in config.channels_hz
    ]
    reference = (
        (frequencies >= config.reference_low_hz)
        & (frequencies <= config.reference_high_hz)
    )
    for channel in config.channels_hz:
        reference &= (
            np.abs(frequencies - channel)
            > config.reference_exclusion_half_width_hz
        )
    if any(indices.size < 3 for indices in targets):
        raise ValueError("A target mask contains fewer than three FFT bins.")
    if np.count_nonzero(reference) < 20:
        raise ValueError("The reference mask contains too few FFT bins.")
    return frequencies, targets, reference


def scan_one_file(
    record: dict[str, Any],
    config: SearchConfig,
    frequencies: np.ndarray,
    target_indices: Sequence[np.ndarray],
    reference_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Return only above-floor five-channel scores from one read-only file."""

    raw = np.memmap(str(record["input_path"]), dtype=config.dtype, mode="r")
    half = config.nfft // 2
    first_time = half / config.sample_rate_hz
    last_time = (raw.size - half) / config.sample_rate_hz
    times = np.arange(first_time, last_time, config.step_seconds, dtype=np.float64)
    if not times.size:
        raise ValueError(f"File is shorter than one FFT block: {record['input_path']}")

    window = np.hanning(config.nfft).astype(np.float32)
    scores = np.empty((times.size, len(config.channels_hz)), dtype=np.float32)
    peaks = np.empty_like(scores, dtype=np.float64)
    tiny = np.finfo(np.float32).tiny
    batch_size = 32
    for batch_start in range(0, times.size, batch_size):
        batch_times = times[batch_start : batch_start + batch_size]
        blocks: list[np.ndarray] = []
        for time_s in batch_times:
            center = int(round(time_s * config.sample_rate_hz))
            blocks.append(np.asarray(raw[center - half : center + half], dtype=np.float32))
        block_array = np.stack(blocks)
        power = np.abs(fft.rfft(block_array * window, axis=1, workers=-1)) ** 2
        reference_floor = np.median(power[:, reference_mask], axis=1)
        stop = batch_start + len(batch_times)
        for channel_index, indices in enumerate(target_indices):
            target_power = power[:, indices]
            top_count = min(3, target_power.shape[1])
            top_power = np.partition(target_power, -top_count, axis=1)[:, -top_count:]
            target_statistic = np.mean(top_power, axis=1)
            scores[batch_start:stop, channel_index] = 10.0 * np.log10(
                np.maximum(target_statistic, tiny) / np.maximum(reference_floor, tiny)
            )
            peaks[batch_start:stop, channel_index] = frequencies[
                indices[np.argmax(target_power, axis=1)]
            ]

    # Removing the median per file and channel suppresses stable bandpass
    # structure while retaining transient excess.  It is not a flux scale.
    scores -= np.median(scores, axis=0, keepdims=True)
    hits: list[dict[str, Any]] = []
    for channel_index, channel_hz in enumerate(config.channels_hz):
        selected = np.flatnonzero(scores[:, channel_index] >= config.retained_hit_floor_db)
        for index in selected:
            hits.append(
                {
                    "dataset": str(record["dataset"]),
                    "segment_id": int(record["segment_id"]),
                    "file_number": int(record["file_number"]),
                    "file_time_s": float(times[index]),
                    "dataset_time_s": float(record["dataset_start_s"]) + float(times[index]),
                    "rf_frequency_hz": float(channel_hz),
                    "excess_db": float(scores[index, channel_index]),
                    "peak_frequency_hz_coarse": float(peaks[index, channel_index]),
                }
            )
    del raw
    return hits


def _has_n_within(
    rows: Sequence[dict[str, Any]],
    threshold_db: float,
    count: int,
    span_seconds: float,
) -> bool:
    """Return whether ``count`` thresholded hits fit inside one time span."""

    times = [
        float(row["dataset_time_s"])
        for row in rows
        if float(row["excess_db"]) >= threshold_db
    ]
    return any(
        times[index + count - 1] - times[index] <= span_seconds
        for index in range(max(0, len(times) - count + 1))
    )


def form_candidates(
    hits: Sequence[dict[str, Any]],
    config: SearchConfig,
) -> list[dict[str, Any]]:
    """Group nearby hits and return groups satisfying any candidate rule."""

    ordered = sorted(
        hits,
        key=lambda row: (
            str(row["dataset"]),
            int(row["segment_id"]),
            float(row["rf_frequency_hz"]),
            float(row["dataset_time_s"]),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    for row in ordered:
        if not groups:
            groups.append([row])
            continue
        current = groups[-1]
        first = current[0]
        previous = current[-1]
        same_stream = (
            row["dataset"] == first["dataset"]
            and int(row["segment_id"]) == int(first["segment_id"])
            and math.isclose(
                float(row["rf_frequency_hz"]),
                float(first["rf_frequency_hz"]),
                abs_tol=1.0,
            )
        )
        time_value = float(row["dataset_time_s"])
        close_to_previous = (
            time_value - float(previous["dataset_time_s"])
            <= config.maximum_hit_gap_seconds
        )
        within_group_span = (
            time_value - float(first["dataset_time_s"])
            <= config.persistent_span_seconds
        )
        if same_stream and close_to_previous and within_group_span:
            current.append(row)
        else:
            groups.append([row])

    candidates: list[dict[str, Any]] = []
    for group in groups:
        rules: list[str] = []
        if max(float(row["excess_db"]) for row in group) >= config.strong_threshold_db:
            rules.append("strong_single")
        if _has_n_within(
            group,
            config.persistent_threshold_db,
            config.persistent_count,
            config.persistent_span_seconds,
        ):
            rules.append("persistent_weak")
        if _has_n_within(
            group,
            config.medium_threshold_db,
            config.medium_count,
            config.medium_span_seconds,
        ):
            rules.append("medium_pair")
        if not rules:
            continue
        peak = max(group, key=lambda row: float(row["excess_db"]))
        # Preserve a small, reproducible set of raw-data probe times so the
        # later dense-STFT morphology layer can run from candidates.csv alone.
        # Include the first hit (useful near a preamble), the strongest hit,
        # then the remaining strongest distinct snapshots up to the limit.
        probe_rows: list[dict[str, Any]] = []
        for row in [group[0], peak, *sorted(
            group, key=lambda item: float(item["excess_db"]), reverse=True
        )]:
            if any(
                math.isclose(
                    float(row["dataset_time_s"]),
                    float(existing["dataset_time_s"]),
                    abs_tol=config.step_seconds / 4.0,
                )
                for existing in probe_rows
            ):
                continue
            probe_rows.append(row)
            if len(probe_rows) >= config.chirp_probe_max_windows:
                break
        coarse_peak_frequencies = np.asarray(
            [float(row["peak_frequency_hz_coarse"]) for row in group],
            dtype=np.float64,
        )
        candidates.append(
            {
                "dataset": str(peak["dataset"]),
                "segment_id": int(peak["segment_id"]),
                "rf_frequency_hz": float(peak["rf_frequency_hz"]),
                "peak_dataset_time_s": float(peak["dataset_time_s"]),
                "peak_file_number": int(peak["file_number"]),
                "peak_file_time_s": float(peak["file_time_s"]),
                "peak_excess_db": float(peak["excess_db"]),
                "peak_frequency_hz_coarse": float(peak["peak_frequency_hz_coarse"]),
                "first_hit_dataset_time_s": float(group[0]["dataset_time_s"]),
                "last_hit_dataset_time_s": float(group[-1]["dataset_time_s"]),
                "hit_count_ge_4db": sum(
                    float(row["excess_db"]) >= config.persistent_threshold_db
                    for row in group
                ),
                "hit_count_ge_6db": sum(
                    float(row["excess_db"]) >= config.medium_threshold_db
                    for row in group
                ),
                "hit_count_ge_8db": sum(
                    float(row["excess_db"]) >= config.strong_threshold_db
                    for row in group
                ),
                "hit_span_s": (
                    float(group[-1]["dataset_time_s"])
                    - float(group[0]["dataset_time_s"])
                ),
                "unique_peak_bin_count": int(
                    np.unique(coarse_peak_frequencies).size
                ),
                "peak_frequency_span_hz": float(
                    np.ptp(coarse_peak_frequencies)
                ),
                "peak_frequency_std_hz": float(
                    np.std(coarse_peak_frequencies)
                ),
                "chirp_probe_dataset_times_s": ";".join(
                    f"{float(row['dataset_time_s']):.9f}" for row in probe_rows
                ),
                "qualification_rules": ";".join(rules),
                "source_file_numbers": ";".join(
                    str(value) for value in sorted({int(row["file_number"]) for row in group})
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            str(row["dataset"]),
            float(row["peak_dataset_time_s"]),
            float(row["rf_frequency_hz"]),
        )
    )
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"C{index:05d}"
    return candidates


def classify_sparse_morphology(
    candidates: Sequence[dict[str, Any]],
    config: SearchConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return frequency-sweeping and fixed-tone-like candidate audit rows.

    A LoRa upchirp crosses the approximately 41.7 kHz occupied band, whereas
    a persistent carrier repeatedly selects the same one or two coarse FFT
    bins.  The test is deliberately a computational prefilter, not a physical
    signal classification.  Rejected rows therefore mean only that they will
    not enter the expensive preamble probe under this configured search.
    """

    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        unique_bins = int(candidate["unique_peak_bin_count"])
        span_hz = float(candidate["peak_frequency_span_hz"])
        reasons: list[str] = []
        if unique_bins < config.morphology_minimum_unique_peak_bins:
            reasons.append("too_few_unique_coarse_peak_bins")
        if span_hz < config.morphology_minimum_peak_span_hz:
            reasons.append("coarse_peak_span_too_narrow")
        row = dict(candidate)
        row["sparse_morphology_pass"] = not reasons
        row["sparse_morphology_reasons"] = ";".join(reasons)
        if reasons:
            rejected.append(row)
        else:
            retained.append(row)
    return retained, rejected


def _read_candidate_raw_window(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    center_dataset_time_s: float,
    duration_s: float,
    config: SearchConfig,
) -> tuple[np.ndarray, float, bool]:
    """Return one short raw-voltage window and its dataset start time."""

    centered = dict(candidate)
    centered["peak_dataset_time_s"] = float(center_dataset_time_s)
    segments, global_start, _, edge_limited = verification_segments(
        centered, inventory, duration_s
    )
    blocks: list[np.ndarray] = []
    for segment in segments:
        raw = np.memmap(
            str(segment["input_path"]), dtype=config.dtype, mode="r"
        )
        first_sample = int(round(
            float(segment["file_start_s"]) * config.sample_rate_hz
        ))
        sample_count = int(round(
            float(segment["duration_s"]) * config.sample_rate_hz
        ))
        blocks.append(np.asarray(
            raw[first_sample : first_sample + sample_count], dtype=np.float32
        ).copy())
        del raw
    if not blocks:
        raise ValueError("The chirp morphology probe selected an empty window.")
    return np.concatenate(blocks), global_start, edge_limited


def _dense_local_stft(
    raw: np.ndarray,
    rf_frequency_hz: float,
    config: SearchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local ``(times_s, frequencies_hz, robust_residual_z)``."""

    nfft = config.chirp_stft_nfft
    hop = config.chirp_stft_hop
    if raw.size < nfft:
        raise ValueError("Raw chirp-probe window is shorter than one STFT frame.")
    starts = np.arange(0, raw.size - nfft + 1, hop, dtype=np.int64)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / config.sample_rate_hz)
    selected = np.flatnonzero(
        np.abs(frequencies - rf_frequency_hz)
        <= config.chirp_stft_half_width_hz
    )
    if selected.size < 8:
        raise ValueError("Dense-STFT target band contains too few frequency bins.")
    window = np.hanning(nfft).astype(np.float32)
    power = np.empty((starts.size, selected.size), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    batch_size = 24
    for batch_start in range(0, starts.size, batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        blocks = np.stack([
            raw[start : start + nfft] for start in batch_indices
        ])
        spectra = fft.rfft(blocks * window, axis=1, workers=-1)
        power[batch_start : batch_start + len(batch_indices)] = (
            np.abs(spectra[:, selected]) ** 2
        )
    magnitude_db = 10.0 * np.log10(np.maximum(power, tiny))
    # Remove persistent bandpass/fixed-line structure per frequency and then
    # remove common broadband variation per time frame.  The result is only a
    # morphology contrast, not calibrated power.
    residual_db = magnitude_db - np.median(magnitude_db, axis=0, keepdims=True)
    residual_db -= np.median(residual_db, axis=1, keepdims=True)
    median_value = float(np.median(residual_db))
    mad = float(np.median(np.abs(residual_db - median_value)))
    robust_sigma_db = max(mad / 0.6744897501960817, 0.25)
    residual_z = (residual_db - median_value) / robust_sigma_db
    times_s = (starts + nfft / 2.0) / config.sample_rate_hz
    return times_s, frequencies[selected], residual_z


def _best_linear_track(
    residual_z: np.ndarray,
    frequencies_hz: np.ndarray,
    frame_step_seconds: float,
    track_duration_seconds: float,
    slopes_hz_per_second: Sequence[float],
) -> dict[str, Any]:
    """Return the strongest straight spectrogram track over supplied slopes."""

    frame_count = max(5, int(round(track_duration_seconds / frame_step_seconds)))
    if residual_z.shape[0] < frame_count:
        raise ValueError("Dense STFT is shorter than the requested track duration.")
    frequency_step_hz = float(np.median(np.diff(frequencies_hz)))
    best: dict[str, Any] | None = None
    time_offsets = np.arange(frame_count, dtype=np.int64)
    for slope in slopes_hz_per_second:
        bin_offsets = np.rint(
            slope * frame_step_seconds * time_offsets / frequency_step_hz
        ).astype(np.int64)
        minimum_offset = int(bin_offsets.min())
        maximum_offset = int(bin_offsets.max())
        first_frequency_index = -minimum_offset
        last_frequency_index = residual_z.shape[1] - maximum_offset
        if first_frequency_index >= last_frequency_index:
            continue
        for time_index in range(residual_z.shape[0] - frame_count + 1):
            rows = time_index + time_offsets
            for frequency_index in range(
                first_frequency_index, last_frequency_index
            ):
                columns = frequency_index + bin_offsets
                values = residual_z[rows, columns]
                score = float(np.mean(values))
                occupancy = float(np.mean(values >= 2.0))
                if best is None or (score, occupancy) > (
                    float(best["score_z"]), float(best["occupancy"])
                ):
                    best = {
                        "score_z": score,
                        "occupancy": occupancy,
                        "slope_hz_per_second": float(slope),
                        "time_index": int(time_index),
                        "frequency_index": int(frequency_index),
                        "row_indices": rows,
                        "column_indices": columns,
                    }
    if best is None:
        raise ValueError("No valid line track fits inside the local STFT band.")
    return best


def _save_chirp_morphology_diagnostic(
    output_path: Path,
    candidate: dict[str, Any],
    window_center_s: float,
    window_start_s: float,
    local_times_s: np.ndarray,
    frequencies_hz: np.ndarray,
    residual_z: np.ndarray,
    positive_track: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Write one dense-STFT diagnostic with the best expected-slope track."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_times = window_start_s + local_times_s
    relative_times_ms = (absolute_times - window_center_s) * 1e3
    relative_frequencies_khz = (
        frequencies_hz - float(candidate["rf_frequency_hz"])
    ) / 1e3
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    mesh = axis.pcolormesh(
        relative_times_ms,
        relative_frequencies_khz,
        residual_z.T,
        shading="auto",
        cmap="viridis",
        vmin=-2.0,
        vmax=max(5.0, float(np.percentile(residual_z, 99.5))),
    )
    rows = np.asarray(positive_track["row_indices"], dtype=int)
    columns = np.asarray(positive_track["column_indices"], dtype=int)
    axis.plot(
        relative_times_ms[rows],
        relative_frequencies_khz[columns],
        color="white",
        linewidth=1.8,
        label="Best expected-slope track",
    )
    axis.set(
        xlabel="Time from sparse-hit probe center (ms)",
        ylabel="Frequency relative to searched center (kHz)",
        title=(
            f"{candidate['candidate_id']} dense raw-STFT morphology | "
            f"score={result['best_track_score_z']:.2f} z, "
            f"occupancy={result['best_track_occupancy']:.2f}"
        ),
    )
    axis.legend(loc="lower right")
    fig.colorbar(mesh, ax=axis, label="Robust local residual (z)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def probe_candidate_chirp_morphology(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    config: SearchConfig,
    diagnostic_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score expected-slope LoRa-like tracks in short raw-voltage STFTs."""

    result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "dataset": candidate["dataset"],
        "rf_frequency_hz": float(candidate["rf_frequency_hz"]),
        "tested_window_count": 0,
        "best_window_dataset_time_s": "",
        "best_slope_hz_per_second": "",
        "best_slope_khz_per_ms": "",
        "best_track_score_z": "",
        "best_track_occupancy": "",
        "best_horizontal_track_score_z": "",
        "best_negative_track_score_z": "",
        "control_margin_z": "",
        "frequency_sweep_span_hz": "",
        "chirp_morphology_pass": False,
        "classification": "error",
        "diagnostic_path": "",
        "error": "",
    }
    try:
        probe_times_text = str(candidate.get("chirp_probe_dataset_times_s", ""))
        probe_times = [
            float(value) for value in probe_times_text.split(";") if value.strip()
        ]
        if not probe_times:
            probe_times = [
                float(candidate["first_hit_dataset_time_s"]),
                float(candidate["peak_dataset_time_s"]),
            ]
        probe_times = probe_times[: config.chirp_probe_max_windows]
        expected = config.chirp_expected_slope_hz_per_second
        tolerance = config.chirp_slope_tolerance_fraction
        positive_slopes = np.linspace(
            expected * (1.0 - tolerance),
            expected * (1.0 + tolerance),
            7,
        )
        negative_slopes = -positive_slopes
        frame_step_seconds = config.chirp_stft_hop / config.sample_rate_hz
        best_bundle: dict[str, Any] | None = None
        for center_time in probe_times:
            raw, window_start, edge_limited = _read_candidate_raw_window(
                candidate,
                inventory,
                center_time,
                config.chirp_probe_window_seconds,
                config,
            )
            local_times, local_frequencies, residual_z = _dense_local_stft(
                raw, float(candidate["rf_frequency_hz"]), config
            )
            positive = _best_linear_track(
                residual_z,
                local_frequencies,
                frame_step_seconds,
                config.chirp_track_duration_seconds,
                positive_slopes,
            )
            horizontal = _best_linear_track(
                residual_z,
                local_frequencies,
                frame_step_seconds,
                config.chirp_track_duration_seconds,
                [0.0],
            )
            negative = _best_linear_track(
                residual_z,
                local_frequencies,
                frame_step_seconds,
                config.chirp_track_duration_seconds,
                negative_slopes,
            )
            margin = float(positive["score_z"]) - max(
                float(horizontal["score_z"]), float(negative["score_z"])
            )
            bundle = {
                "center_time": float(center_time),
                "window_start": float(window_start),
                "edge_limited": bool(edge_limited),
                "times": local_times,
                "frequencies": local_frequencies,
                "residual_z": residual_z,
                "positive": positive,
                "horizontal": horizontal,
                "negative": negative,
                "margin": margin,
            }
            if best_bundle is None or (
                float(positive["score_z"]), margin
            ) > (
                float(best_bundle["positive"]["score_z"]),
                float(best_bundle["margin"]),
            ):
                best_bundle = bundle
        if best_bundle is None:
            raise RuntimeError("No dense-STFT probe window was evaluated.")
        positive = best_bundle["positive"]
        horizontal = best_bundle["horizontal"]
        negative = best_bundle["negative"]
        margin = float(best_bundle["margin"])
        passed = (
            float(positive["score_z"]) >= config.chirp_minimum_track_score_z
            and float(positive["occupancy"])
            >= config.chirp_minimum_track_occupancy
            and margin >= config.chirp_minimum_control_margin_z
        )
        result.update(
            {
                "tested_window_count": len(probe_times),
                "best_window_dataset_time_s": best_bundle["center_time"],
                "best_slope_hz_per_second": float(
                    positive["slope_hz_per_second"]
                ),
                "best_slope_khz_per_ms": float(
                    positive["slope_hz_per_second"] / 1e6
                ),
                "best_track_score_z": float(positive["score_z"]),
                "best_track_occupancy": float(positive["occupancy"]),
                "best_horizontal_track_score_z": float(horizontal["score_z"]),
                "best_negative_track_score_z": float(negative["score_z"]),
                "control_margin_z": margin,
                "frequency_sweep_span_hz": float(
                    positive["slope_hz_per_second"]
                    * config.chirp_track_duration_seconds
                ),
                "chirp_morphology_pass": bool(passed),
                "classification": (
                    "chirp_morphology_match"
                    if passed else "rejected_non_chirp_morphology"
                ),
            }
        )
        near_threshold = (
            float(positive["score_z"])
            >= 0.8 * config.chirp_minimum_track_score_z
            and float(positive["occupancy"])
            >= 0.5 * config.chirp_minimum_track_occupancy
        )
        if diagnostic_path is not None and (passed or near_threshold):
            diagnostic = Path(diagnostic_path)
            _save_chirp_morphology_diagnostic(
                diagnostic,
                candidate,
                float(best_bundle["center_time"]),
                float(best_bundle["window_start"]),
                np.asarray(best_bundle["times"]),
                np.asarray(best_bundle["frequencies"]),
                np.asarray(best_bundle["residual_z"]),
                positive,
                result,
            )
            result["diagnostic_path"] = str(diagnostic)
        return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result


def _known_search_recovery(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report whether the candidate stage recovers the two known packets."""

    checks: list[dict[str, Any]] = []
    for dataset, file_number, file_time_s, channel_hz in KNOWN_SEARCH_CHECKS:
        matched = [
            row
            for row in candidates
            if row["dataset"] == dataset
            and int(row["peak_file_number"]) == file_number
            and math.isclose(float(row["rf_frequency_hz"]), channel_hz, abs_tol=1.0)
            and abs(float(row["peak_file_time_s"]) - file_time_s) <= 1.5
        ]
        checks.append(
            {
                "dataset": dataset,
                "file_number": file_number,
                "known_preamble_file_time_s": file_time_s,
                "channel_hz": channel_hz,
                "recovered": bool(matched),
                "matching_candidate_ids": [row["candidate_id"] for row in matched],
            }
        )
    return checks


def save_candidate_overview(
    path: Path,
    candidates: Sequence[dict[str, Any]],
    datasets: Sequence[str],
) -> None:
    """Plot candidate time versus searched channel, coloured by excess dB."""

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(14, max(3.0 * len(datasets), 4.0)),
        constrained_layout=True,
        squeeze=False,
    )
    scatter = None
    for axis, dataset in zip(axes[:, 0], datasets):
        rows = [row for row in candidates if row["dataset"] == dataset]
        if rows:
            scatter = axis.scatter(
                [float(row["peak_dataset_time_s"]) for row in rows],
                [float(row["rf_frequency_hz"]) / 1e6 for row in rows],
                c=[float(row["peak_excess_db"]) for row in rows],
                cmap="viridis",
                vmin=4.0,
                s=34,
                edgecolor="black",
                linewidth=0.35,
            )
        axis.set_title(dataset)
        axis.set_ylabel("Searched RF center (MHz)")
        axis.grid(axis="x", alpha=0.2)
    axes[-1, 0].set_xlabel("Time from start of dataset (s)")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes[:, 0], label="Peak transient excess (dB)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_scan(args: argparse.Namespace, config: SearchConfig) -> None:
    """Run inventory and five-channel candidate search."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.csv"
    if candidate_path.exists():
        raise FileExistsError(
            f"{candidate_path} already exists. Choose a new output directory for a new run."
        )
    inventory = build_inventory(Path(args.data_root), args.datasets, config)
    _write_csv(output_dir / "dataset_inventory.csv", inventory, INVENTORY_FIELDS)
    _write_json(
        output_dir / "search_config.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "datasets": list(args.datasets),
            "search_config": asdict(config),
            "candidate_policy": (
                "A compact energy candidate only. Final classification is made by LFdemod."
            ),
        },
    )
    frequencies, targets, reference = _search_masks(config)
    hits: list[dict[str, Any]] = []
    total = len(inventory)
    for index, record in enumerate(inventory, start=1):
        print(
            f"[scan {index}/{total}] {record['dataset']}/{record['file_name']}",
            flush=True,
        )
        hits.extend(scan_one_file(record, config, frequencies, targets, reference))
    candidates = form_candidates(hits, config)
    _write_csv(candidate_path, candidates, CANDIDATE_FIELDS)
    save_candidate_overview(
        output_dir / "candidate_overview.png", candidates, args.datasets
    )
    checks = _known_search_recovery(candidates)
    summary = {
        "file_count": len(inventory),
        "total_duration_s": sum(float(row["duration_s"]) for row in inventory),
        "retained_above_floor_hit_count_in_memory": len(hits),
        "candidate_count": len(candidates),
        "candidate_counts_by_channel": {
            f"{channel / 1e6:.4f}_MHz": sum(
                math.isclose(float(row["rf_frequency_hz"]), channel, abs_tol=1.0)
                for row in candidates
            )
            for channel in config.channels_hz
        },
        "known_search_checks": checks,
        "dense_sparse_samples_saved": False,
        "scope_limit": (
            "No injection/recovery completeness measurement was performed; a non-detection "
            "means no event passed this configured search and LFdemod verification chain."
        ),
    }
    _write_json(output_dir / "scan_summary.json", summary)
    print(f"Wrote {len(candidates)} candidates to {candidate_path}")
    for check in checks:
        if not check["recovered"]:
            print("WARNING: known search regression check was not recovered:", check)


def _typed_inventory(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert inventory CSV values back to numeric Python values."""

    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for key in ("file_number", "file_bytes", "sample_count", "segment_id"):
            item[key] = int(item[key])
        for key in ("duration_s", "dataset_start_s", "dataset_stop_s"):
            item[key] = float(item[key])
        item["contiguous_with_previous"] = item["contiguous_with_previous"] == "True"
        converted.append(item)
    return converted


def _typed_candidates(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert candidate CSV values back to numeric Python values."""

    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for key in (
            "segment_id",
            "peak_file_number",
            "hit_count_ge_4db",
            "hit_count_ge_6db",
            "hit_count_ge_8db",
            "unique_peak_bin_count",
        ):
            if key in item and item[key] != "":
                item[key] = int(item[key])
        for key in (
            "rf_frequency_hz",
            "peak_dataset_time_s",
            "peak_file_time_s",
            "peak_excess_db",
            "peak_frequency_hz_coarse",
            "first_hit_dataset_time_s",
            "last_hit_dataset_time_s",
            "hit_span_s",
            "peak_frequency_span_hz",
            "peak_frequency_std_hz",
        ):
            if key in item and item[key] != "":
                item[key] = float(item[key])
        converted.append(item)
    return converted


def verification_segments(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    duration_s: float,
) -> tuple[list[dict[str, Any]], float, float, bool]:
    """Return source byte ranges for a centered, segment-bounded decode window."""

    eligible = [
        row
        for row in inventory
        if row["dataset"] == candidate["dataset"]
        and int(row["segment_id"]) == int(candidate["segment_id"])
    ]
    if not eligible:
        raise ValueError(f"No inventory segment found for {candidate['candidate_id']}")
    segment_start = min(float(row["dataset_start_s"]) for row in eligible)
    segment_stop = max(float(row["dataset_stop_s"]) for row in eligible)
    desired_start = float(candidate["peak_dataset_time_s"]) - duration_s / 2.0
    desired_stop = desired_start + duration_s
    selected_start = max(segment_start, desired_start)
    selected_stop = min(segment_stop, desired_stop)
    edge_limited = not (
        math.isclose(selected_start, desired_start, abs_tol=1e-9)
        and math.isclose(selected_stop, desired_stop, abs_tol=1e-9)
    )
    segments: list[dict[str, Any]] = []
    for row in eligible:
        overlap_start = max(selected_start, float(row["dataset_start_s"]))
        overlap_stop = min(selected_stop, float(row["dataset_stop_s"]))
        if overlap_stop <= overlap_start:
            continue
        segments.append(
            {
                "input_path": str(row["input_path"]),
                "file_number": int(row["file_number"]),
                "file_start_s": overlap_start - float(row["dataset_start_s"]),
                "duration_s": overlap_stop - overlap_start,
                "dataset_start_s": overlap_start,
            }
        )
    if not segments:
        raise ValueError(f"Empty verification window for {candidate['candidate_id']}")
    return segments, selected_start, selected_stop - selected_start, edge_limited


def _copy_segmented_raw(
    segments: Sequence[dict[str, Any]],
    destination: Path,
    sample_rate_hz: float,
    dtype: str,
) -> None:
    """Stream selected byte ranges into one temporary contiguous raw file."""

    itemsize = np.dtype(dtype).itemsize
    chunk_bytes = 64 * 1024 * 1024
    with destination.open("wb") as output:
        for segment in segments:
            start_sample = int(round(float(segment["file_start_s"]) * sample_rate_hz))
            sample_count = int(round(float(segment["duration_s"]) * sample_rate_hz))
            remaining = sample_count * itemsize
            with Path(str(segment["input_path"])).open("rb") as source:
                source.seek(start_sample * itemsize)
                while remaining:
                    block = source.read(min(chunk_bytes, remaining))
                    if not block:
                        raise IOError(f"Unexpected EOF while stitching {segment['input_path']}")
                    output.write(block)
                    remaining -= len(block)


def _preamble_probe_window(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    config: SearchConfig,
) -> tuple[list[dict[str, Any]], float, float, bool]:
    """Return a short, guarded source window covering all sparse hits."""

    first_hit = float(candidate["first_hit_dataset_time_s"])
    last_hit = float(candidate["last_hit_dataset_time_s"])
    center = 0.5 * (first_hit + last_hit)
    duration = max(
        config.preamble_probe_minimum_seconds,
        last_hit - first_hit + 2.0 * config.preamble_probe_guard_seconds,
    )
    centered = dict(candidate)
    centered["peak_dataset_time_s"] = center
    return verification_segments(centered, inventory, duration)


def _load_pinned_lora_detector() -> tuple[Any, Any, Any]:
    """Return private lora-phy detector types after enforcing version 0.3.0."""

    # lora-phy 0.3.0 uses an alias removed by newer NumPy releases.  LFdemod
    # installs the same narrow compatibility alias when its decoder is used.
    if not hasattr(np, "bitwise_right_shift"):
        np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]
    try:
        from lora_phy import LoRaReceiver
        from lora_phy.errors import NoPreambleError, NoSyncError
    except ImportError as error:
        raise ImportError(
            "The preamble probe needs the LFdemod Starlink optional dependency: "
            "python -m pip install -e '/XXX/LFdemod[starlink-vhf]'"
        ) from error
    try:
        backend_version = version("lora-phy")
    except PackageNotFoundError:
        backend_version = "unknown"
    if backend_version != "0.3.0":
        raise RuntimeError(
            "The short preamble probe uses version-specific lora-phy private APIs; "
            f"expected 0.3.0, found {backend_version}."
        )
    return LoRaReceiver, NoPreambleError, NoSyncError


def _circular_symbol_distance(left: float, right: float, modulus: int) -> float:
    """Return the shortest absolute distance on a cyclic symbol alphabet."""

    difference = (float(left) - float(right)) % modulus
    return min(difference, modulus - difference)


def _detect_starlink_preamble_sync(
    channel_iq: np.ndarray,
    sample_rate_hz: float,
    rf_frequency_hz: float,
    lora_config: StarlinkVHFLoRaConfig,
) -> dict[str, Any]:
    """Detect LoRa preambles/SFDs and count Starlink-sync matches, without payload decode."""

    LoRaReceiver, NoPreambleError, NoSyncError = _load_pinned_lora_detector()
    lora_iq, lora_rate = prepare_lora_iq(
        channel_iq, sample_rate_hz, lora_config
    )
    receiver = LoRaReceiver(
        rf_frequency_hz,
        lora_config.spreading_factor,
        lora_config.bandwidth_hz,
        lora_rate,
        has_header=lora_config.has_explicit_header,
        implicit_header_payload_len=lora_config.expected_payload_bytes,
        implicit_header_coding_rate=lora_config.expected_coding_rate,
        implicit_header_enable_crc=lora_config.expected_crc_enabled,
        preamble_len=lora_config.preamble_symbols,
        resample_to_2x=False,
    )
    receiver._proc.low_data_rate_optimization = (
        lora_config.force_low_data_rate_optimization
    )
    filtered_iq = np.asarray(receiver.lowpass(lora_iq))
    expected_sync = starlink_sync_symbol_offsets(lora_config.sync_word)
    symbol_modulus = 2 ** lora_config.spreading_factor
    symbol_duration_s = symbol_modulus / lora_config.bandwidth_hz
    preamble_to_data_symbols = lora_config.preamble_symbols + 2.0 + 2.25

    preamble_count = 0
    synchronized_count = 0
    detections: list[dict[str, Any]] = []
    search_start = 0
    while search_start < len(filtered_iq):
        try:
            detected_start = int(receiver._detect(filtered_iq, search_start))
        except NoPreambleError:
            break
        preamble_count += 1
        next_search = max(search_start + receiver._sample_num, detected_start + receiver._sample_num)
        try:
            phy_data_start, preamble_bin, carrier_offset_hz = receiver._sync(
                filtered_iq, detected_start
            )
        except NoSyncError:
            search_start = next_search
            continue
        synchronized_count += 1
        sync_bin_1 = receiver._dechirp(
            filtered_iq, round(phy_data_start - 4.25 * receiver._sample_num)
        )[1]
        sync_bin_2 = receiver._dechirp(
            filtered_iq, round(phy_data_start - 3.25 * receiver._sample_num)
        )[1]
        sync_symbol_1 = receiver._peak_idx_to_symbol(sync_bin_1, preamble_bin)
        sync_symbol_2 = receiver._peak_idx_to_symbol(sync_bin_2, preamble_bin)
        sync_match = (
            _circular_symbol_distance(
                sync_symbol_1, expected_sync[0], symbol_modulus
            ) <= lora_config.sync_tolerance_symbols
            and _circular_symbol_distance(
                sync_symbol_2, expected_sync[1], symbol_modulus
            ) <= lora_config.sync_tolerance_symbols
        )
        phy_data_start_seconds = float(phy_data_start) / lora_rate
        detections.append(
            {
                "preamble_start_seconds": phy_data_start_seconds
                - preamble_to_data_symbols * symbol_duration_s,
                "phy_data_start_seconds": phy_data_start_seconds,
                "carrier_offset_hz": float(carrier_offset_hz),
                "sync_symbol_1": int(sync_symbol_1),
                "sync_symbol_2": int(sync_symbol_2),
                "starlink_sync_match": bool(sync_match),
            }
        )
        search_start = max(next_search, int(phy_data_start + receiver._sample_num))

    matching = [row for row in detections if row["starlink_sync_match"]]
    if matching:
        classification = "starlink_sync_match"
        representative = matching[0]
    elif detections:
        classification = "lora_sync_other"
        representative = detections[0]
    elif preamble_count:
        classification = "preamble_without_sfd"
        representative = None
    else:
        classification = "no_preamble"
        representative = None
    return {
        "preamble_detected_count": preamble_count,
        "sfd_synchronized_count": synchronized_count,
        "starlink_sync_match_count": len(matching),
        "classification": classification,
        "representative": representative,
        "expected_sync_symbol_1": int(expected_sync[0]),
        "expected_sync_symbol_2": int(expected_sync[1]),
    }


def probe_candidate_preamble(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    config: SearchConfig,
) -> dict[str, Any]:
    """Return one short-window LoRa/Starlink preamble probe result.

    This stage deliberately stops before PHY-header and payload decoding.  A
    positive result means that a 15-upchirp preamble, LoRa SFD, and the two
    expected Starlink sync symbols were found.  Only a later full LFdemod run
    can establish header, CRC, payload, and Starlink-profile validity.
    """

    segments, global_start, duration_s, edge_limited = _preamble_probe_window(
        candidate, inventory, config
    )
    segment_text = ";".join(
        f"{Path(str(item['input_path'])).name}:{float(item['file_start_s']):.6f}+"
        f"{float(item['duration_s']):.6f}"
        for item in segments
    )
    result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "dataset": candidate["dataset"],
        "rf_frequency_hz": float(candidate["rf_frequency_hz"]),
        "probe_dataset_start_s": global_start,
        "probe_duration_s": duration_s,
        "source_segments": segment_text,
        "edge_limited": edge_limited,
        "preamble_detected_count": 0,
        "sfd_synchronized_count": 0,
        "starlink_sync_match_count": 0,
        "first_preamble_dataset_time_s": "",
        "first_carrier_offset_hz": "",
        "first_sync_symbol_1": "",
        "first_sync_symbol_2": "",
        "expected_sync_symbol_1": "",
        "expected_sync_symbol_2": "",
        "classification": "error",
        "error": "",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="cma21_starlink_preamble_") as temporary:
            temporary_path = Path(temporary)
            if len(segments) == 1:
                source = np.memmap(
                    str(segments[0]["input_path"]), dtype=config.dtype, mode="r"
                )
                first_sample = int(round(
                    float(segments[0]["file_start_s"]) * config.sample_rate_hz
                ))
                sample_count = int(round(
                    float(segments[0]["duration_s"]) * config.sample_rate_hz
                ))
                raw = source[first_sample : first_sample + sample_count]
            else:
                stitched = temporary_path / "preamble_window.dat"
                _copy_segmented_raw(
                    segments, stitched, config.sample_rate_hz, config.dtype
                )
                source = np.memmap(str(stitched), dtype=config.dtype, mode="r")
                raw = source

            ddc_config = StarlinkVHFDDCConfig(fs_in=config.sample_rate_hz)
            channel_iq, _ = downconvert_starlink_vhf_voltage(
                raw,
                float(candidate["rf_frequency_hz"]),
                ddc_config,
            )
            del raw
            del source
            detected = _detect_starlink_preamble_sync(
                channel_iq,
                ddc_config.fs_out,
                float(candidate["rf_frequency_hz"]),
                StarlinkVHFLoRaConfig(),
            )
            representative = detected.pop("representative")
            result.update(detected)
            if representative is not None:
                result.update(
                    {
                        "first_preamble_dataset_time_s": global_start
                        + float(representative["preamble_start_seconds"]),
                        "first_carrier_offset_hz": float(
                            representative["carrier_offset_hz"]
                        ),
                        "first_sync_symbol_1": int(
                            representative["sync_symbol_1"]
                        ),
                        "first_sync_symbol_2": int(
                            representative["sync_symbol_2"]
                        ),
                    }
                )
            return result
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result


def _replace_path_prefix(value: Any, old: str, new: str) -> Any:
    """Recursively replace temporary output paths in copied LFdemod metadata."""

    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, old, new) for key, item in value.items()}
    return value


def _retain_detail_output(
    probe_dir: Path,
    final_dir: Path,
    summary: dict[str, Any],
    candidate: dict[str, Any],
    segments: Sequence[dict[str, Any]],
    verification_start_s: float,
) -> dict[str, Any]:
    """Copy one accepted LFdemod run and make its provenance self-contained."""

    if final_dir.exists():
        raise FileExistsError(f"Detailed output already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(probe_dir, final_dir)
    rewritten = _replace_path_prefix(summary, str(probe_dir), str(final_dir))
    rewritten["search_candidate"] = candidate
    rewritten["search_source_segments"] = list(segments)
    rewritten["verification_dataset_start_s"] = verification_start_s
    rewritten["stitched_temporary_input_used"] = len(segments) > 1
    _write_json(final_dir / "summary.json", rewritten)
    return rewritten


def _nearest_channel(observed_hz: float, channels_hz: Sequence[float]) -> float:
    """Return the coordinated center closest to an observed packet center."""

    return float(min(channels_hz, key=lambda value: abs(value - observed_hz)))


def _packet_rows(
    summary: dict[str, Any],
    candidate: dict[str, Any],
    verification_start_s: float,
    detailed_output_dir: str,
    config: SearchConfig,
) -> list[dict[str, Any]]:
    """Flatten retained LFdemod packet summaries into event rows."""

    rows: list[dict[str, Any]] = []
    request_start = float(summary["requested_start_seconds"])
    for packet in summary.get("packets", []):
        decoded = packet.get("decoded_payload", {}) or {}
        packet_classification = (
            "strict" if packet.get("starlink_profile_match") is True else "likely"
        )
        preamble_in_input = float(packet["preamble_start_in_file_seconds"])
        stop_in_input = float(packet["packet_stop_in_file_seconds"])
        # Direct-file verification may start inside the source file, whereas a
        # stitched temporary file always starts at zero.  Subtract the request
        # start to express packet times relative to the selected global window.
        preamble_dataset = verification_start_s + preamble_in_input - request_start
        stop_dataset = verification_start_s + stop_in_input - request_start
        observed = float(packet["observed_center_hz"])
        rows.append(
            {
                "event_id": "",
                "candidate_id": candidate["candidate_id"],
                "dataset": candidate["dataset"],
                # Classification is packet-level here.  A strict candidate can
                # contain an additional non-matching LoRa packet, which must not
                # be promoted into detections.csv merely by sharing its window.
                "classification": packet_classification,
                "searched_rf_frequency_hz": float(candidate["rf_frequency_hz"]),
                "assigned_channel_hz": _nearest_channel(observed, config.channels_hz),
                "preamble_dataset_time_s": preamble_dataset,
                "packet_stop_dataset_time_s": stop_dataset,
                "carrier_offset_hz": packet.get("carrier_offset_hz"),
                "observed_center_hz": observed,
                "payload_bytes": decoded.get("payload_length_bytes"),
                "symbol_count": packet.get("symbol_count"),
                "header_valid": packet.get("header_valid"),
                "crc_valid": packet.get("crc_valid"),
                "starlink_profile_match": packet.get("starlink_profile_match"),
                "spacecraft_id_u16_le": decoded.get("spacecraft_id_u16_le"),
                "message_number_u24_le": decoded.get("message_number_u24_le"),
                "protocol_id_u16_le": decoded.get("protocol_id_u16_le"),
                "payload_hex": decoded.get("payload_hex", ""),
                "detailed_output_dir": detailed_output_dir,
            }
        )
    return rows


def verify_one_candidate(
    candidate: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    output_dir: Path,
    config: SearchConfig,
    retain_iq: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one temporary LFdemod probe; retain detail only for decoded packets."""

    segments, global_start, duration_s, edge_limited = verification_segments(
        candidate, inventory, config.verification_window_seconds
    )
    segment_text = ";".join(
        f"{Path(str(item['input_path'])).name}:{float(item['file_start_s']):.6f}+"
        f"{float(item['duration_s']):.6f}"
        for item in segments
    )
    base_result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "dataset": candidate["dataset"],
        "searched_rf_frequency_hz": float(candidate["rf_frequency_hz"]),
        "classification": "error",
        "packet_count": 0,
        "crc_valid_packet_count": 0,
        "starlink_profile_match_count": 0,
        "verification_dataset_start_s": global_start,
        "verification_duration_s": duration_s,
        "source_segments": segment_text,
        "edge_limited": edge_limited,
        "detailed_output_dir": "",
        "error": "",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="cma21_starlink_verify_") as temporary:
            temporary_path = Path(temporary)
            if len(segments) == 1:
                pipeline_input = Path(str(segments[0]["input_path"]))
                pipeline_start = float(segments[0]["file_start_s"])
                input_sample_count = pipeline_input.stat().st_size // np.dtype(
                    config.dtype
                ).itemsize
                available_duration = (
                    input_sample_count / config.sample_rate_hz - pipeline_start
                )
                pipeline_duration = min(duration_s, available_duration)
            else:
                pipeline_input = temporary_path / "stitched_window.dat"
                _copy_segmented_raw(
                    segments,
                    pipeline_input,
                    config.sample_rate_hz,
                    config.dtype,
                )
                pipeline_start = 0.0
                pipeline_duration = (
                    pipeline_input.stat().st_size
                    / np.dtype(config.dtype).itemsize
                    / config.sample_rate_hz
                )
            # Rounding each byte range to an integer sample can change a
            # stitched window by a few nanoseconds.  Passing its measured
            # duration avoids requesting one sample beyond the temporary file.
            base_result["verification_duration_s"] = pipeline_duration
            probe_dir = temporary_path / "probe"
            file_config = StarlinkVHFFileConfig(
                rf_frequency_hz=float(candidate["rf_frequency_hz"]),
                start_seconds=pipeline_start,
                duration_seconds=pipeline_duration,
                dtype=config.dtype,
                save_iq=retain_iq,
                label=f"five-channel search {candidate['candidate_id']}",
            )
            summary = demodulate_starlink_vhf_file(
                pipeline_input,
                probe_dir,
                file_config,
                overwrite=False,
            )
            strict_count = int(summary["starlink_profile_match_count"])
            packet_count = int(summary["packet_count"])
            if strict_count > 0:
                classification = "strict"
            elif packet_count > 0:
                classification = "likely"
            else:
                classification = "rejected_energy_only"
            detailed_output = ""
            if classification in {"strict", "likely"}:
                slug = (
                    f"{candidate['candidate_id']}_"
                    f"{_format_frequency_mhz(float(candidate['rf_frequency_hz']))}MHz"
                )
                final_dir = output_dir / "detailed" / classification / slug
                summary = _retain_detail_output(
                    probe_dir,
                    final_dir,
                    summary,
                    candidate,
                    segments,
                    global_start,
                )
                detailed_output = str(final_dir)
            base_result.update(
                {
                    "classification": classification,
                    "packet_count": packet_count,
                    "crc_valid_packet_count": int(summary["crc_valid_packet_count"]),
                    "starlink_profile_match_count": strict_count,
                    "detailed_output_dir": detailed_output,
                }
            )
            packets = _packet_rows(
                summary,
                candidate,
                global_start,
                detailed_output,
                config,
            )
            return base_result, packets
    except Exception as error:  # preserve other candidates and record provenance
        base_result["error"] = f"{type(error).__name__}: {error}"
        return base_result, []


def deduplicate_packet_rows(
    rows: Sequence[dict[str, Any]],
    tolerance_s: float,
) -> list[dict[str, Any]]:
    """Return one row per physical packet across overlapping searches."""

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            float(row["preamble_dataset_time_s"]),
            0 if row["classification"] == "strict" else 1,
        ),
    )
    kept: list[dict[str, Any]] = []
    for row in ordered:
        duplicate_index: int | None = None
        for index in range(len(kept) - 1, -1, -1):
            previous = kept[index]
            if previous["dataset"] != row["dataset"]:
                continue
            time_difference = abs(
                float(previous["preamble_dataset_time_s"])
                - float(row["preamble_dataset_time_s"])
            )
            if time_difference > tolerance_s:
                if float(previous["preamble_dataset_time_s"]) < float(row["preamble_dataset_time_s"]):
                    break
                continue
            payload_same = bool(row.get("payload_hex")) and (
                row.get("payload_hex") == previous.get("payload_hex")
            )
            center_close = abs(
                float(row["observed_center_hz"]) - float(previous["observed_center_hz"])
            ) <= 5_000.0
            if payload_same or center_close:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(dict(row))
        else:
            current = kept[duplicate_index]
            combined_candidate_ids = ";".join(
                sorted(set(str(current["candidate_id"]).split(";")) | {str(row["candidate_id"])})
            )
            if row["classification"] == "strict" and current["classification"] != "strict":
                kept[duplicate_index] = dict(row)
            kept[duplicate_index]["candidate_id"] = combined_candidate_ids
    kept.sort(key=lambda row: (str(row["dataset"]), float(row["preamble_dataset_time_s"])))
    for index, row in enumerate(kept, start=1):
        row["event_id"] = f"E{index:04d}"
    return kept


def save_verification_overview(
    path: Path,
    verification_rows: Sequence[dict[str, Any]],
    datasets: Sequence[str],
    channels_hz: Sequence[float],
) -> None:
    """Write a heatmap of candidate classifications by dataset and channel."""

    labels = ["strict", "likely", "rejected_energy_only", "error"]
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(16, max(4.0, 0.65 * len(datasets) + 2.0)),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, classification in zip(axes[0], labels):
        matrix = np.zeros((len(datasets), len(channels_hz)), dtype=int)
        for row in verification_rows:
            if row["classification"] != classification:
                continue
            dataset_index = datasets.index(str(row["dataset"]))
            channel_index = int(
                np.argmin(
                    np.abs(
                        np.asarray(channels_hz)
                        - float(row["searched_rf_frequency_hz"])
                    )
                )
            )
            matrix[dataset_index, channel_index] += 1
        axis.imshow(matrix, cmap="Blues", vmin=0, aspect="auto")
        axis.set_title(classification.replace("_", " "))
        axis.set_xticks(range(len(channels_hz)))
        axis.set_xticklabels(
            [f"{value / 1e6:.4f}" for value in channels_hz],
            rotation=50,
            ha="right",
        )
        axis.set_yticks(range(len(datasets)))
        axis.set_yticklabels(datasets if axis is axes[0, 0] else [])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="black",
                )
    axes[0, 0].set_ylabel("Dataset")
    fig.supxlabel("Searched RF center (MHz)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_verify(args: argparse.Namespace, config: SearchConfig) -> None:
    """Verify compact candidates with LFdemod and write final summaries."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    candidates_path = output_dir / "candidates.csv"
    inventory_path = output_dir / "dataset_inventory.csv"
    verification_path = output_dir / "verification_results.csv"
    if not candidates_path.is_file() or not inventory_path.is_file():
        raise FileNotFoundError("Run the scan stage first in the same output directory.")
    if verification_path.exists():
        raise FileExistsError(
            f"{verification_path} already exists. Choose a new output directory for a new run."
        )
    candidates = _typed_candidates(_read_csv(candidates_path))
    inventory = _typed_inventory(_read_csv(inventory_path))
    verification_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[verify {index}/{total}] {candidate['candidate_id']} "
            f"{candidate['rf_frequency_hz'] / 1e6:.4f} MHz",
            flush=True,
        )
        result, packets = verify_one_candidate(
            candidate,
            inventory,
            output_dir,
            config,
            args.retain_iq,
        )
        verification_rows.append(result)
        packet_rows.extend(packets)

    _write_csv(verification_path, verification_rows, VERIFICATION_FIELDS)
    unique_packets = deduplicate_packet_rows(
        packet_rows, config.deduplication_time_tolerance_seconds
    )
    strict = [row for row in unique_packets if row["classification"] == "strict"]
    likely = [row for row in unique_packets if row["classification"] == "likely"]
    _write_csv(output_dir / "detections.csv", strict, PACKET_FIELDS)
    _write_csv(output_dir / "likely_events.csv", likely, PACKET_FIELDS)
    save_verification_overview(
        output_dir / "verification_overview.png",
        verification_rows,
        list(args.datasets),
        config.channels_hz,
    )
    counts = {
        label: sum(row["classification"] == label for row in verification_rows)
        for label in ("strict", "likely", "rejected_energy_only", "error")
    }
    _write_json(
        output_dir / "search_summary.json",
        {
            "candidate_count": len(candidates),
            "candidate_classification_counts": counts,
            "deduplicated_strict_packet_count": len(strict),
            "deduplicated_likely_packet_count": len(likely),
            "retained_detailed_output_policy": (
                "Detailed LFdemod artifacts retained only for strict and likely classifications."
            ),
            "strict_definition": (
                "At least one packet passes the installed LFdemod Starlink VHF profile checks."
            ),
            "likely_definition": (
                "LFdemod found at least one LoRa PHY packet, but none passed all Starlink profile checks."
            ),
            "scope_limit": (
                "No injection/recovery completeness measurement was performed. Threshold-dependent "
                "non-detections must not be presented as an absolute absence of transmissions."
            ),
        },
    )
    print(f"Verification complete. Strict packets: {len(strict)}; likely packets: {len(likely)}")


def build_parser() -> argparse.ArgumentParser:
    """Return the documented command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Search five Starlink VHF downlinks in selected 21CMA raw datasets, "
            "then verify candidates with the installed LFdemod decoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        "-s",
        choices=("scan", "verify", "all"),
        default="all",
        help="Run candidate search, LFdemod verification, or both.",
    )
    parser.add_argument(
        "--data-root",
        "-i",
        default="/XXX/data",
        help="Parent directory containing the named observation datasets.",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directory names to process in the supplied order.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="New run directory for compact tables and retained positive details.",
    )
    parser.add_argument(
        "--retain-iq",
        action="store_true",
        help="Keep 240 kS/s channel IQ only for strict/likely detailed outputs.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.025,
        help="Spacing between short first-stage FFT snapshots.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=65_536,
        help="Number of 480 MS/s samples in each first-stage Hann FFT.",
    )
    parser.add_argument(
        "--verification-window-seconds",
        type=float,
        default=5.5,
        help="Multi-second LFdemod window centred on each candidate peak.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the selected non-destructive search stages."""

    parser = build_parser()
    args = parser.parse_args()
    config = SearchConfig(
        nfft=args.nfft,
        step_seconds=args.step_seconds,
        verification_window_seconds=args.verification_window_seconds,
    )
    config.validate()
    if args.stage in {"scan", "all"}:
        run_scan(args, config)
    if args.stage in {"verify", "all"}:
        run_verify(args, config)


if __name__ == "__main__":
    main()
