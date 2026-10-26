"""Shared paths and observation metadata for the paper-reproduction notebooks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
EVENT_METADATA_PATH = DATA_DIR / "observation" / "event_files.csv"
OBSERVATION_WINDOWS_PATH = DATA_DIR / "observation" / "observation_windows.csv"


def event_metadata() -> pd.DataFrame:
    """Load the immutable acquisition metadata for the two Zenodo files."""

    frame = pd.read_csv(EVENT_METADATA_PATH)
    frame["file_start_utc"] = pd.to_datetime(frame["file_start_utc"], utc=True)
    return frame


def event_raw_path(event_id: str) -> Path:
    """Return the expected local raw-data path for an event label."""

    rows = event_metadata().loc[lambda table: table["event_id"] == event_id]
    if len(rows) != 1:
        raise KeyError(f"Expected one metadata row for {event_id!r}; found {len(rows)}")
    return RAW_DATA_DIR / str(rows.iloc[0]["filename"])


def verify_raw_data(*, calculate_md5: bool = False) -> None:
    """Check downloaded filenames/sizes and optionally their Zenodo MD5 hashes."""

    missing: list[Path] = []
    problems: list[str] = []
    for row in event_metadata().itertuples(index=False):
        path = RAW_DATA_DIR / row.filename
        if not path.is_file():
            missing.append(path)
            continue
        if path.stat().st_size != int(row.size_bytes):
            problems.append(
                f"{path}: {path.stat().st_size} bytes; expected {int(row.size_bytes)}"
            )
        if calculate_md5:
            digest = hashlib.md5()  # noqa: S324 - checksum supplied by Zenodo
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != row.zenodo_md5:
                problems.append(f"{path}: MD5 mismatch")
    if missing:
        raise FileNotFoundError(
            "Download the two Zenodo DAT files into data/raw/:\n"
            + "\n".join(str(path) for path in missing)
        )
    if problems:
        raise ValueError("Raw-data validation failed:\n" + "\n".join(problems))


def file_start_utc(filename: str) -> pd.Timestamp:
    """Return acquisition UTC from committed metadata, never filesystem mtime."""

    rows = event_metadata().loc[lambda table: table["filename"] == Path(filename).name]
    if len(rows) != 1:
        raise KeyError(f"No acquisition timestamp for {filename!r}")
    return rows.iloc[0]["file_start_utc"]

