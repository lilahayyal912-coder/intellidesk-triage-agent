"""
tests/test_pipeline.py

Unit tests for src/pipeline.py focusing on _process_ticket error-handling paths.
No live API, no file I/O — all agents are mocked.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

# Import the function under test and the exception types
from pipeline import _process_ticket, _error_row, OUTPUT_COLUMNS
from services.severity_classifier import ClassificationResult, ClassificationError
from services.escalation_agent import EscalationResult, EscalationError
from services.runbook_retriever import RetrievalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_row(
    ticket_id="TKT-TEST",
    service="Auth-Service",
    error_message="All users unable to log in.",
) -> pd.Series:
    return pd.Series({
        "ticket_id": ticket_id,
        "service": service,
        "error_message": error_message,
    })


def _make_classification(
    severity="P1 - Critical",
    short_reason="Full auth outage.",
    immediate_action="Page IAM on-call.",
    requires_human_escalation=True,
) -> ClassificationResult:
    return ClassificationResult(
        severity=severity,
        short_reason=short_reason,
        immediate_action=immediate_action,
        requires_human_escalation=requires_human_escalation,
        raw_response=json.dumps({"severity": severity}),
    )


def _make_escalation(
    escalated_team="IAM Squad + SRE On-Call",
    final_severity="P1 - Critical",
    justification="Confirmed P1 per runbook.",
    recommended_next_steps=None,
    requires_human_escalation=True,
) -> EscalationResult:
    return EscalationResult(
        escalated_team=escalated_team,
        final_severity=final_severity,
        justification=justification,
        recommended_next_steps=recommended_next_steps or ["Check logs.", "Page on-call."],
        requires_human_escalation=requires_human_escalation,
        raw_response="{}",
    )


def _make_no_match_retrieval() -> RetrievalResult:
    return RetrievalResult(
        matched_entry=None,
        relevance_score=0.0,
        escalation_team="",
        recommended_action="",
        found=False,
    )


# ── _error_row tests ──────────────────────────────────────────────────────────

class TestErrorRow(unittest.TestCase):

    def test_agent1_error_row_has_error_severity(self):
        row = _error_row("TKT-001", "Auth-Service", "Login down", "Agent1", "timeout")
        self.assertEqual(row["severity"], "ERROR (Agent1)")
        self.assertEqual(row["final_severity"], "ERROR (Agent1)")

    def test_error_row_requires_escalation(self):
        row = _error_row("TKT-001", "Auth-Service", "Login down", "Agent1", "timeout")
        self.assertTrue(row["requires_human_escalation"])

    def test_error_row_contains_all_output_columns(self):
        row = _error_row("TKT-001", "Auth-Service", "Login down", "Agent1", "timeout")
        for col in OUTPUT_COLUMNS:
            self.assertIn(col, row, f"Missing column: {col}")


# ── _process_ticket: happy path ───────────────────────────────────────────────

class TestProcessTicketHappyPath(unittest.TestCase):

    def _run(self, severity="P1 - Critical"):
        classification = _make_classification(severity=severity)
        escalation = _make_escalation(final_severity=severity)

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_no_match_retrieval()

        with patch("pipeline.classify", return_value=classification), \
             patch("pipeline.escalate", return_value=escalation):
            return _process_ticket(_make_row(), retriever, MagicMock())

    def test_returns_all_output_columns(self):
        result = self._run()
        for col in OUTPUT_COLUMNS:
            self.assertIn(col, result, f"Missing column: {col}")

    def test_severity_from_agent1(self):
        result = self._run("P2 - High")
        self.assertEqual(result["severity"], "P2 - High")

    def test_final_severity_from_agent3(self):
        result = self._run("P2 - High")
        self.assertEqual(result["final_severity"], "P2 - High")

    def test_steps_joined_with_pipe(self):
        result = self._run()
        self.assertIn("|", result["recommended_next_steps"])


# ── _process_ticket: Agent 1 failure ─────────────────────────────────────────

class TestProcessTicketAgent1Failure(unittest.TestCase):

    def _run(self):
        retriever = MagicMock()
        with patch("pipeline.classify", side_effect=ClassificationError("API timeout")), \
             patch("pipeline.escalate") as mock_escalate:
            result = _process_ticket(_make_row(), retriever, MagicMock())
            # Agent 3 must NOT be called when Agent 1 fails
            mock_escalate.assert_not_called()
            return result

    def test_severity_is_error_agent1(self):
        result = self._run()
        self.assertEqual(result["severity"], "ERROR (Agent1)")

    def test_final_severity_is_error_agent1(self):
        result = self._run()
        self.assertEqual(result["final_severity"], "ERROR (Agent1)")

    def test_reason_contains_original_error(self):
        result = self._run()
        self.assertIn("API timeout", result["short_reason"])

    def test_requires_escalation_true(self):
        result = self._run()
        self.assertTrue(result["requires_human_escalation"])

    def test_all_output_columns_present(self):
        result = self._run()
        for col in OUTPUT_COLUMNS:
            self.assertIn(col, result)


# ── _process_ticket: Agent 3 failure — the fixed behaviour ───────────────────

class TestProcessTicketAgent3Failure(unittest.TestCase):
    """
    Regression tests for Risk 2 fix:
    When Agent 3 raises EscalationError after Agent 1 succeeds, the
    Agent 1 classification (severity, short_reason) must be preserved
    in the output row.
    """

    def _run(self, severity="P1 - Critical", short_reason="Full auth outage."):
        classification = _make_classification(severity=severity, short_reason=short_reason)
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_no_match_retrieval()

        with patch("pipeline.classify", return_value=classification), \
             patch("pipeline.escalate", side_effect=EscalationError("LLM returned bad JSON")):
            return _process_ticket(_make_row(), retriever, MagicMock())

    def test_severity_is_preserved_from_agent1(self):
        """Agent 1 severity must survive an Agent 3 failure."""
        result = self._run(severity="P1 - Critical")
        self.assertEqual(
            result["severity"],
            "P1 - Critical",
            "Agent 1 severity was discarded on Agent 3 failure",
        )

    def test_short_reason_is_preserved_from_agent1(self):
        """Agent 1 short_reason must survive an Agent 3 failure."""
        result = self._run(short_reason="Full auth outage.")
        self.assertEqual(
            result["short_reason"],
            "Full auth outage.",
            "Agent 1 short_reason was discarded on Agent 3 failure",
        )

    def test_final_severity_marked_as_error_agent3(self):
        result = self._run()
        self.assertIn("ERROR (Agent3)", result["final_severity"])

    def test_severity_is_not_marked_as_error(self):
        """The classifier severity column must NOT be overwritten with ERROR."""
        result = self._run()
        self.assertNotIn("ERROR", result["severity"])

    def test_justification_explains_agent3_failure(self):
        result = self._run()
        self.assertIn("Agent 3 failed", result["justification"])
        self.assertIn("LLM returned bad JSON", result["justification"])

    def test_requires_escalation_true(self):
        result = self._run()
        self.assertTrue(result["requires_human_escalation"])

    def test_escalated_team_empty(self):
        """No team can be confirmed when Agent 3 did not complete."""
        result = self._run()
        self.assertEqual(result["escalated_team"], "")

    def test_all_output_columns_present(self):
        result = self._run()
        for col in OUTPUT_COLUMNS:
            self.assertIn(col, result, f"Missing column: {col}")

    def test_p2_severity_preserved(self):
        """Verify preservation works for severities other than P1."""
        result = self._run(severity="P2 - High", short_reason="Partial degradation.")
        self.assertEqual(result["severity"], "P2 - High")
        self.assertEqual(result["short_reason"], "Partial degradation.")

    def test_p4_severity_preserved(self):
        result = self._run(severity="P4 - Low", short_reason="Cosmetic issue.")
        self.assertEqual(result["severity"], "P4 - Low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
