"""
tests/test_escalation_agent.py

Unit tests for src/services/escalation_agent.py.
All tests use mock clients – no live API key required.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from services.escalation_agent import (
    EscalationError,
    EscalationRequest,
    EscalationResult,
    _build_user_prompt,
    _parse_response,
    escalate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_request(
    *,
    service="Auth-Service",
    error_message="All users unable to log in, HTTP 500 from auth service.",
    classifier_severity="P1 - Critical",
    classifier_short_reason="Full authentication outage.",
    runbook_sop_id="SOP-001",
    runbook_title="Authentication and Login Failures",
    runbook_escalation_team="Identity & Access Management (IAM) Squad + SRE On-Call",
    runbook_recommended_action="Check auth-service error logs for JWT or cert exceptions.",
    runbook_severity_guidance="P1 if failure rate > 25% or full regional outage.",
) -> EscalationRequest:
    return EscalationRequest(
        service=service,
        error_message=error_message,
        classifier_severity=classifier_severity,
        classifier_short_reason=classifier_short_reason,
        runbook_sop_id=runbook_sop_id,
        runbook_title=runbook_title,
        runbook_escalation_team=runbook_escalation_team,
        runbook_recommended_action=runbook_recommended_action,
        runbook_severity_guidance=runbook_severity_guidance,
    )


def _valid_payload(**overrides) -> dict:
    base = {
        "escalated_team": "Identity & Access Management (IAM) Squad + SRE On-Call",
        "final_severity": "P1 - Critical",
        "justification": (
            "Authentication is completely down for all users. "
            "The runbook confirms P1 and routes to the IAM Squad."
        ),
        "recommended_next_steps": [
            "Page the IAM on-call engineer immediately.",
            "Check auth-service logs for JWT or cert exceptions.",
            "Verify token-signing key integrity in secrets manager.",
        ],
        "requires_human_escalation": True,
    }
    base.update(overrides)
    return base


def _make_mock_client(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ── _build_user_prompt tests ──────────────────────────────────────────────────

class TestBuildUserPrompt(unittest.TestCase):

    def test_includes_service(self):
        req = _make_request()
        prompt = _build_user_prompt(req)
        self.assertIn("Auth-Service", prompt)

    def test_includes_error_message(self):
        req = _make_request()
        prompt = _build_user_prompt(req)
        self.assertIn("HTTP 500", prompt)

    def test_includes_classifier_severity(self):
        req = _make_request()
        prompt = _build_user_prompt(req)
        self.assertIn("P1 - Critical", prompt)

    def test_includes_runbook_sop_id_when_present(self):
        req = _make_request()
        prompt = _build_user_prompt(req)
        self.assertIn("SOP-001", prompt)

    def test_includes_runbook_team_when_present(self):
        req = _make_request()
        prompt = _build_user_prompt(req)
        self.assertIn("Identity & Access Management", prompt)

    def test_no_runbook_shows_not_found_message(self):
        req = _make_request(
            runbook_sop_id=None,
            runbook_title=None,
            runbook_escalation_team=None,
            runbook_recommended_action=None,
            runbook_severity_guidance=None,
        )
        prompt = _build_user_prompt(req)
        self.assertIn("No matching runbook entry", prompt)


# ── _parse_response tests (pure, no mock) ────────────────────────────────────

class TestParseResponse(unittest.TestCase):

    def test_valid_payload_returns_result(self):
        result = _parse_response(json.dumps(_valid_payload()))
        self.assertIsInstance(result, EscalationResult)
        self.assertEqual(result.final_severity, "P1 - Critical")
        self.assertTrue(result.requires_human_escalation)

    def test_p4_no_escalation(self):
        payload = _valid_payload(
            final_severity="P4 - Low",
            requires_human_escalation=False,
            recommended_next_steps=["Log a low-priority bug ticket."],
        )
        result = _parse_response(json.dumps(payload))
        self.assertFalse(result.requires_human_escalation)
        self.assertEqual(result.final_severity, "P4 - Low")

    def test_all_four_severities_accepted(self):
        for sev, esc in [("P1 - Critical", True), ("P2 - High", True),
                          ("P3 - Medium", False), ("P4 - Low", False)]:
            p = _valid_payload(final_severity=sev, requires_human_escalation=esc)
            result = _parse_response(json.dumps(p))
            self.assertEqual(result.final_severity, sev)

    def test_steps_stored_as_list(self):
        result = _parse_response(json.dumps(_valid_payload()))
        self.assertIsInstance(result.recommended_next_steps, list)
        self.assertGreater(len(result.recommended_next_steps), 0)

    def test_raw_response_preserved(self):
        raw = json.dumps(_valid_payload())
        result = _parse_response(raw)
        self.assertEqual(result.raw_response, raw)

    def test_non_json_raises(self):
        with self.assertRaises(EscalationError) as ctx:
            _parse_response("Sorry, I cannot help with that.")
        self.assertIn("non-JSON", str(ctx.exception))

    def test_missing_escalated_team_raises(self):
        p = _valid_payload()
        del p["escalated_team"]
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("escalated_team", str(ctx.exception))

    def test_missing_final_severity_raises(self):
        p = _valid_payload()
        del p["final_severity"]
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("final_severity", str(ctx.exception))

    def test_missing_justification_raises(self):
        p = _valid_payload()
        del p["justification"]
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("justification", str(ctx.exception))

    def test_missing_next_steps_raises(self):
        p = _valid_payload()
        del p["recommended_next_steps"]
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("recommended_next_steps", str(ctx.exception))

    def test_missing_requires_escalation_raises(self):
        p = _valid_payload()
        del p["requires_human_escalation"]
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("requires_human_escalation", str(ctx.exception))

    def test_invalid_severity_raises(self):
        p = _valid_payload(final_severity="P0 - Catastrophic")
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("P0 - Catastrophic", str(ctx.exception))

    def test_empty_steps_list_raises(self):
        p = _valid_payload(recommended_next_steps=[])
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("recommended_next_steps", str(ctx.exception))

    def test_steps_not_list_raises(self):
        p = _valid_payload(recommended_next_steps="just do it")
        with self.assertRaises(EscalationError) as ctx:
            _parse_response(json.dumps(p))
        self.assertIn("recommended_next_steps", str(ctx.exception))

    def test_string_true_escalation_coerced(self):
        p = _valid_payload(requires_human_escalation="true")
        result = _parse_response(json.dumps(p))
        self.assertIs(result.requires_human_escalation, True)

    def test_string_false_escalation_coerced(self):
        p = _valid_payload(
            final_severity="P4 - Low",
            requires_human_escalation="false",
            recommended_next_steps=["Log ticket."],
        )
        result = _parse_response(json.dumps(p))
        self.assertIs(result.requires_human_escalation, False)

    def test_invalid_escalation_value_raises(self):
        p = _valid_payload(requires_human_escalation="maybe")
        with self.assertRaises(EscalationError):
            _parse_response(json.dumps(p))


# ── escalate() integration tests (mock client injected) ──────────────────────

class TestEscalate(unittest.TestCase):

    def test_successful_escalation(self):
        client = _make_mock_client(json.dumps(_valid_payload()))
        result = escalate(_make_request(), client=client)
        self.assertEqual(result.final_severity, "P1 - Critical")
        self.assertIn("IAM", result.escalated_team)
        self.assertTrue(result.requires_human_escalation)

    def test_empty_service_raises(self):
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(service=""), client=MagicMock())
        self.assertIn("service", str(ctx.exception))

    def test_whitespace_service_raises(self):
        with self.assertRaises(EscalationError):
            escalate(_make_request(service="   "), client=MagicMock())

    def test_empty_error_message_raises(self):
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(error_message=""), client=MagicMock())
        self.assertIn("error_message", str(ctx.exception))

    def test_empty_severity_raises(self):
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(classifier_severity=""), client=MagicMock())
        self.assertIn("classifier_severity", str(ctx.exception))

    def test_api_status_error_raises_escalation_error(self):
        import openai as _openai
        client = MagicMock()
        client.chat.completions.create.side_effect = _openai.APIStatusError(
            message="Unauthorized",
            response=MagicMock(status_code=401),
            body=None,
        )
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(), client=client)
        self.assertIn("401", str(ctx.exception))

    def test_generic_exception_raises_escalation_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("timeout")
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(), client=client)
        self.assertIn("timeout", str(ctx.exception))

    def test_non_json_api_response_raises(self):
        client = _make_mock_client("I am unable to process this request.")
        with self.assertRaises(EscalationError) as ctx:
            escalate(_make_request(), client=client)
        self.assertIn("non-JSON", str(ctx.exception))

    def test_temperature_is_zero(self):
        client = _make_mock_client(json.dumps(_valid_payload()))
        escalate(_make_request(), client=client)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs.get("temperature"), 0)

    def test_model_kwarg_passed(self):
        client = _make_mock_client(json.dumps(_valid_payload()))
        escalate(_make_request(), client=client)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertIn("model", kwargs)

    def test_no_runbook_match_still_produces_result(self):
        req = _make_request(
            runbook_sop_id=None,
            runbook_title=None,
            runbook_escalation_team=None,
            runbook_recommended_action=None,
            runbook_severity_guidance=None,
        )
        payload = _valid_payload(escalated_team="Auth-Service On-Call Team")
        client = _make_mock_client(json.dumps(payload))
        result = escalate(req, client=client)
        self.assertIsInstance(result, EscalationResult)
        self.assertTrue(result.escalated_team)

    def test_p4_incident_no_escalation(self):
        payload = _valid_payload(
            final_severity="P4 - Low",
            requires_human_escalation=False,
            recommended_next_steps=["Log a cosmetic bug ticket."],
        )
        client = _make_mock_client(json.dumps(payload))
        result = escalate(
            _make_request(
                classifier_severity="P4 - Low",
                error_message="Dashboard button is slightly misaligned.",
            ),
            client=client,
        )
        self.assertFalse(result.requires_human_escalation)

    def test_steps_are_strings(self):
        client = _make_mock_client(json.dumps(_valid_payload()))
        result = escalate(_make_request(), client=client)
        for step in result.recommended_next_steps:
            self.assertIsInstance(step, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
