"""
tests/test_severity_classifier.py

Unit tests for src/services/severity_classifier.py.

All tests use unittest.mock to patch the openai.OpenAI client so no live
API key or network connection is required.
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Make src/ importable without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Patch config-level env vars before importing the classifier so _require()
# does not blow up in a CI environment that has no .env file.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from services.severity_classifier import (
    ClassificationError,
    ClassificationResult,
    _parse_response,
    classify,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_client(content: str) -> MagicMock:
    """Return a mock openai.OpenAI client whose chat completion returns *content*."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _valid_payload(**overrides) -> dict:
    base = {
        "severity": "P1 - Critical",
        "short_reason": "Complete authentication failure across all regions.",
        "immediate_action": "Page the IAM on-call engineer and check cert validity.",
        "requires_human_escalation": True,
    }
    base.update(overrides)
    return base


# ── _parse_response tests (pure, no mock needed) ──────────────────────────────

class TestParseResponse(unittest.TestCase):

    def test_valid_p1_payload(self):
        raw = json.dumps(_valid_payload())
        result = _parse_response(raw)
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.severity, "P1 - Critical")
        self.assertTrue(result.requires_human_escalation)

    def test_valid_p4_payload(self):
        raw = json.dumps(_valid_payload(
            severity="P4 - Low",
            short_reason="Cosmetic tooltip truncation on one viewport.",
            immediate_action="Log a low-priority bug ticket.",
            requires_human_escalation=False,
        ))
        result = _parse_response(raw)
        self.assertEqual(result.severity, "P4 - Low")
        self.assertFalse(result.requires_human_escalation)

    def test_requires_escalation_as_string_true(self):
        raw = json.dumps(_valid_payload(requires_human_escalation="true"))
        result = _parse_response(raw)
        self.assertIs(result.requires_human_escalation, True)

    def test_requires_escalation_as_string_false(self):
        raw = json.dumps(_valid_payload(
            severity="P3 - Medium",
            requires_human_escalation="false",
        ))
        result = _parse_response(raw)
        self.assertIs(result.requires_human_escalation, False)

    def test_invalid_json_raises(self):
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response("not json at all")
        self.assertIn("non-JSON", str(ctx.exception))

    def test_missing_severity_field_raises(self):
        payload = _valid_payload()
        del payload["severity"]
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response(json.dumps(payload))
        self.assertIn("severity", str(ctx.exception))

    def test_missing_short_reason_raises(self):
        payload = _valid_payload()
        del payload["short_reason"]
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response(json.dumps(payload))
        self.assertIn("short_reason", str(ctx.exception))

    def test_missing_immediate_action_raises(self):
        payload = _valid_payload()
        del payload["immediate_action"]
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response(json.dumps(payload))
        self.assertIn("immediate_action", str(ctx.exception))

    def test_missing_requires_escalation_raises(self):
        payload = _valid_payload()
        del payload["requires_human_escalation"]
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response(json.dumps(payload))
        self.assertIn("requires_human_escalation", str(ctx.exception))

    def test_invalid_severity_value_raises(self):
        raw = json.dumps(_valid_payload(severity="P0 - Catastrophic"))
        with self.assertRaises(ClassificationError) as ctx:
            _parse_response(raw)
        self.assertIn("P0 - Catastrophic", str(ctx.exception))

    def test_invalid_escalation_value_raises(self):
        raw = json.dumps(_valid_payload(requires_human_escalation="maybe"))
        with self.assertRaises(ClassificationError):
            _parse_response(raw)

    def test_raw_response_preserved(self):
        raw = json.dumps(_valid_payload())
        result = _parse_response(raw)
        self.assertEqual(result.raw_response, raw)

    def test_all_four_severities_accepted(self):
        for sev in ("P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"):
            escalate = sev in ("P1 - Critical", "P2 - High")
            raw = json.dumps(_valid_payload(severity=sev, requires_human_escalation=escalate))
            result = _parse_response(raw)
            self.assertEqual(result.severity, sev)


# ── classify() integration tests (mock client injected) ───────────────────────

class TestClassify(unittest.TestCase):

    def test_successful_classification(self):
        mock_client = _make_mock_client(json.dumps(_valid_payload()))
        result = classify("Auth-Service", "Login failure rate 100%", client=mock_client)
        self.assertEqual(result.severity, "P1 - Critical")
        self.assertTrue(result.requires_human_escalation)

    def test_empty_service_raises(self):
        with self.assertRaises(ClassificationError) as ctx:
            classify("", "Some error", client=MagicMock())
        self.assertIn("service", str(ctx.exception))

    def test_whitespace_only_service_raises(self):
        with self.assertRaises(ClassificationError):
            classify("   ", "Some error", client=MagicMock())

    def test_empty_error_message_raises(self):
        with self.assertRaises(ClassificationError) as ctx:
            classify("Auth-Service", "", client=MagicMock())
        self.assertIn("error_message", str(ctx.exception))

    def test_whitespace_only_error_message_raises(self):
        with self.assertRaises(ClassificationError):
            classify("Auth-Service", "   ", client=MagicMock())

    def test_api_status_error_raises_classification_error(self):
        import openai as _openai
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.APIStatusError(
            message="Unauthorized",
            response=MagicMock(status_code=401),
            body=None,
        )
        with self.assertRaises(ClassificationError) as ctx:
            classify("Database", "Primary node down", client=mock_client)
        self.assertIn("401", str(ctx.exception))

    def test_generic_exception_raises_classification_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("timeout")
        with self.assertRaises(ClassificationError) as ctx:
            classify("Network", "Packet loss", client=mock_client)
        self.assertIn("timeout", str(ctx.exception))

    def test_invalid_json_from_api_raises(self):
        mock_client = _make_mock_client("Sorry, I cannot classify this.")
        with self.assertRaises(ClassificationError) as ctx:
            classify("Frontend-UI", "White screen", client=mock_client)
        self.assertIn("non-JSON", str(ctx.exception))

    def test_model_is_passed_to_api_call(self):
        mock_client = _make_mock_client(json.dumps(_valid_payload()))
        classify("Payment-Gateway", "HTTP 500 on charges", client=mock_client)
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertIn("model", call_kwargs.kwargs)

    def test_temperature_is_zero(self):
        mock_client = _make_mock_client(json.dumps(_valid_payload()))
        classify("Email-Service", "SMTP relay down", client=mock_client)
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("temperature"), 0)

    def test_p4_no_escalation(self):
        payload = _valid_payload(
            severity="P4 - Low",
            short_reason="Tooltip text truncated.",
            immediate_action="Log as low-priority bug.",
            requires_human_escalation=False,
        )
        mock_client = _make_mock_client(json.dumps(payload))
        result = classify("Frontend-UI", "Tooltip truncated at 64 chars", client=mock_client)
        self.assertEqual(result.severity, "P4 - Low")
        self.assertFalse(result.requires_human_escalation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
