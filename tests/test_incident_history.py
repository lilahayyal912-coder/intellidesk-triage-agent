"""
tests/test_incident_history.py

Unit tests for src/services/incident_history.py.
All tests use temporary files — no live API, no permanent state.
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from services.incident_history import (
    HISTORY_COLUMNS,
    HistoryRecord,
    load_incident_history,
    save_incident_history,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_record(
    service="Auth-Service",
    error_message="All users unable to log in.",
    severity="P1 - Critical",
    short_reason="Full auth outage.",
    responsible_team="IAM Squad + SRE On-Call",
    requires_human_escalation=True,
    recommended_next_steps="Check logs. | Page on-call.",
    timestamp="",
) -> HistoryRecord:
    return HistoryRecord(
        service=service,
        error_message=error_message,
        severity=severity,
        short_reason=short_reason,
        responsible_team=responsible_team,
        requires_human_escalation=requires_human_escalation,
        recommended_next_steps=recommended_next_steps,
        timestamp=timestamp,
    )


def _tmp_path() -> Path:
    """Return a path to a file that does not exist yet (in a temp dir)."""
    td = tempfile.mkdtemp()
    return Path(td) / "test_history.csv"


# ── HistoryRecord tests ───────────────────────────────────────────────────────

class TestHistoryRecord(unittest.TestCase):

    def test_timestamp_auto_filled_when_empty(self):
        rec = _make_record(timestamp="")
        self.assertTrue(rec.timestamp, "timestamp should be auto-filled")
        self.assertIn("UTC", rec.timestamp)

    def test_explicit_timestamp_preserved(self):
        rec = _make_record(timestamp="2024-11-01 10:00:00 UTC")
        self.assertEqual(rec.timestamp, "2024-11-01 10:00:00 UTC")

    def test_all_fields_accessible(self):
        rec = _make_record()
        for field in HISTORY_COLUMNS:
            self.assertTrue(hasattr(rec, field), f"Missing field: {field}")


# ── save_incident_history tests ───────────────────────────────────────────────

class TestSaveIncidentHistory(unittest.TestCase):

    def test_creates_file_on_first_save(self):
        path = _tmp_path()
        self.assertFalse(path.exists())
        save_incident_history(_make_record(), path=path)
        self.assertTrue(path.exists())

    def test_file_contains_header_row(self):
        path = _tmp_path()
        save_incident_history(_make_record(), path=path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, HISTORY_COLUMNS)

    def test_file_contains_data_row(self):
        path = _tmp_path()
        rec = _make_record(severity="P2 - High")
        save_incident_history(rec, path=path)
        df = load_incident_history(path=path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["severity"], "P2 - High")

    def test_second_save_appends_not_overwrites(self):
        path = _tmp_path()
        save_incident_history(_make_record(severity="P1 - Critical"), path=path)
        save_incident_history(_make_record(severity="P4 - Low"), path=path)
        df = load_incident_history(path=path)
        self.assertEqual(len(df), 2)

    def test_header_written_only_once(self):
        path = _tmp_path()
        save_incident_history(_make_record(), path=path)
        save_incident_history(_make_record(), path=path)
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        # 1 header + 2 data rows = 3 lines
        self.assertEqual(len(lines), 3, "Header should appear exactly once")

    def test_creates_parent_directories(self):
        td = tempfile.mkdtemp()
        nested = Path(td) / "a" / "b" / "history.csv"
        self.assertFalse(nested.parent.exists())
        save_incident_history(_make_record(), path=nested)
        self.assertTrue(nested.exists())

    def test_does_not_raise_on_write_failure(self):
        # Pass a path that cannot be written (a directory path)
        bad_path = Path(tempfile.mkdtemp())  # is a directory, not a file
        # Should log a warning but not raise
        try:
            save_incident_history(_make_record(), path=bad_path)
        except Exception as exc:
            self.fail(f"save_incident_history raised unexpectedly: {exc}")

    def test_saves_all_history_columns(self):
        path = _tmp_path()
        save_incident_history(_make_record(), path=path)
        df = load_incident_history(path=path)
        for col in HISTORY_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column in output: {col}")

    def test_requires_human_escalation_false_saved(self):
        path = _tmp_path()
        save_incident_history(_make_record(requires_human_escalation=False), path=path)
        df = load_incident_history(path=path)
        val = str(df.iloc[0]["requires_human_escalation"]).lower()
        self.assertIn(val, ("false", "0"), f"Unexpected value: {val}")


# ── load_incident_history tests ───────────────────────────────────────────────

class TestLoadIncidentHistory(unittest.TestCase):

    def test_returns_empty_df_when_file_missing(self):
        path = _tmp_path()
        df = load_incident_history(path=path)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), HISTORY_COLUMNS)

    def test_returns_empty_df_for_empty_file(self):
        path = _tmp_path()
        path.touch()
        df = load_incident_history(path=path)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), HISTORY_COLUMNS)

    def test_returns_empty_df_for_header_only_file(self):
        path = _tmp_path()
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(HISTORY_COLUMNS) + "\n")
        df = load_incident_history(path=path)
        self.assertTrue(df.empty)

    def test_returns_correct_schema_for_valid_file(self):
        path = _tmp_path()
        save_incident_history(_make_record(), path=path)
        df = load_incident_history(path=path)
        self.assertEqual(list(df.columns), HISTORY_COLUMNS)

    def test_returns_all_saved_rows(self):
        path = _tmp_path()
        for sev in ["P1 - Critical", "P2 - High", "P3 - Medium"]:
            save_incident_history(_make_record(severity=sev), path=path)
        df = load_incident_history(path=path)
        self.assertEqual(len(df), 3)

    def test_sorted_newest_first(self):
        path = _tmp_path()
        save_incident_history(_make_record(timestamp="2024-11-01 08:00:00 UTC"), path=path)
        save_incident_history(_make_record(timestamp="2024-11-01 10:00:00 UTC"), path=path)
        save_incident_history(_make_record(timestamp="2024-11-01 09:00:00 UTC"), path=path)
        df = load_incident_history(path=path)
        self.assertEqual(df.iloc[0]["timestamp"], "2024-11-01 10:00:00 UTC")

    def test_missing_column_filled_with_empty_string(self):
        path = _tmp_path()
        # Write a CSV missing 'responsible_team'
        cols_without = [c for c in HISTORY_COLUMNS if c != "responsible_team"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols_without)
            writer.writeheader()
            writer.writerow({c: "val" for c in cols_without})
        df = load_incident_history(path=path)
        self.assertIn("responsible_team", df.columns)
        self.assertEqual(df.iloc[0]["responsible_team"], "")

    def test_returns_empty_df_for_corrupt_csv(self):
        path = _tmp_path()
        path.write_text("this is not valid csv\x00\x01\x02\n", encoding="latin-1")
        # Should not raise; returns empty or partial DF
        try:
            df = load_incident_history(path=path)
            self.assertIsNotNone(df)
        except Exception as exc:
            self.fail(f"load_incident_history raised on corrupt file: {exc}")

    def test_extra_columns_dropped(self):
        path = _tmp_path()
        extra_cols = HISTORY_COLUMNS + ["extra_col"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=extra_cols)
            writer.writeheader()
            writer.writerow({c: "x" for c in extra_cols})
        df = load_incident_history(path=path)
        self.assertEqual(list(df.columns), HISTORY_COLUMNS)
        self.assertNotIn("extra_col", df.columns)


# ── Filtering tests (data-level) ──────────────────────────────────────────────

class TestFiltering(unittest.TestCase):
    """
    Filtering is performed by the dashboard UI, but the underlying
    load_incident_history data supports it correctly.
    """

    def setUp(self):
        self.path = _tmp_path()
        records = [
            _make_record(service="Auth-Service",     severity="P1 - Critical", requires_human_escalation=True),
            _make_record(service="Database",          severity="P2 - High",     requires_human_escalation=True),
            _make_record(service="Frontend-UI",       severity="P4 - Low",      requires_human_escalation=False),
            _make_record(service="Payment-Gateway",   severity="P3 - Medium",   requires_human_escalation=False),
        ]
        for rec in records:
            save_incident_history(rec, path=self.path)
        self.df = load_incident_history(path=self.path)

    def test_filter_by_severity_p1(self):
        filtered = self.df[self.df["severity"] == "P1 - Critical"]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["service"], "Auth-Service")

    def test_filter_by_service(self):
        filtered = self.df[self.df["service"] == "Database"]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["severity"], "P2 - High")

    def test_filter_escalated(self):
        filtered = self.df[
            self.df["requires_human_escalation"].astype(str).str.lower() == "true"
        ]
        self.assertEqual(len(filtered), 2)

    def test_filter_not_escalated(self):
        filtered = self.df[
            self.df["requires_human_escalation"].astype(str).str.lower() == "false"
        ]
        self.assertEqual(len(filtered), 2)

    def test_all_four_records_loaded(self):
        self.assertEqual(len(self.df), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
