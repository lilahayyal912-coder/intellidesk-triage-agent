"""
incident_history.py – Persistent incident triage history.

Appends one record per successfully triaged incident to a CSV file and
provides a clean load function for the dashboard.

Public API
----------
    save_incident_history(record: HistoryRecord, path: Path = DEFAULT_HISTORY_PATH) -> None
    load_incident_history(path: Path = DEFAULT_HISTORY_PATH) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH: Path = _REPO_ROOT / "data" / "incident_history.csv"

# ── Schema ────────────────────────────────────────────────────────────────────
HISTORY_COLUMNS = [
    "timestamp",
    "service",
    "error_message",
    "severity",
    "short_reason",
    "responsible_team",
    "requires_human_escalation",
    "recommended_next_steps",
]


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class HistoryRecord:
    """One history entry produced after a successful end-to-end triage."""
    service: str
    error_message: str
    severity: str                        # from Agent 1
    short_reason: str                    # from Agent 1
    responsible_team: str                # from Agent 3
    requires_human_escalation: bool      # from Agent 3
    recommended_next_steps: str          # pipe-joined list from Agent 3
    timestamp: str = ""                  # auto-filled if empty

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_incident_history(
    record: HistoryRecord,
    path: Path = DEFAULT_HISTORY_PATH,
) -> None:
    """
    Append *record* to the history CSV at *path*.

    Creates the file (and any parent directories) on first use.
    Writes the header row only when the file is new or empty.
    Does nothing and logs a warning if the write fails.

    Parameters
    ----------
    record:
        A fully populated ``HistoryRecord``.
    path:
        Destination CSV file.  Defaults to ``data/incident_history.csv``.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        row = {col: getattr(record, col) for col in HISTORY_COLUMNS}
        new_df = pd.DataFrame([row], columns=HISTORY_COLUMNS)

        write_header = not path.exists() or path.stat().st_size == 0
        new_df.to_csv(path, mode="a", index=False, header=write_header)
        logger.debug("History record saved to %s", path)

    except Exception as exc:
        # Never crash the application on a history write failure.
        logger.warning("Failed to save incident history to %s: %s", path, exc)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_incident_history(
    path: Path = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    """
    Load the incident history CSV and return a validated DataFrame.

    Always returns a DataFrame with exactly the columns in ``HISTORY_COLUMNS``,
    even when the file is missing, empty, or has unexpected columns.

    Parameters
    ----------
    path:
        Source CSV file.  Defaults to ``data/incident_history.csv``.

    Returns
    -------
    pd.DataFrame
        Rows sorted newest-first by ``timestamp``.
        Returns an empty DataFrame (correct schema) when no data is available.
    """
    empty = pd.DataFrame(columns=HISTORY_COLUMNS)

    path = Path(path)
    if not path.exists():
        logger.debug("History file not found at %s — returning empty DataFrame", path)
        return empty

    if path.stat().st_size == 0:
        logger.debug("History file is empty at %s", path)
        return empty

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        logger.warning("Failed to read history file %s: %s", path, exc)
        return empty

    if df.empty:
        return empty

    # Fill any missing expected columns with empty strings, drop unknown extras.
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[HISTORY_COLUMNS]

    # Sort newest-first; ignore errors if timestamp column is malformed.
    try:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    return df
