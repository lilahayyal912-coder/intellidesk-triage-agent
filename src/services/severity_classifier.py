"""
severity_classifier.py – Agent 1: Incident Severity Classifier

Classifies an IT incident into one of four priority levels (P1–P4) using
the configured Groq LLM endpoint.  Returns structured data ready for the
downstream escalation and reporting pipeline.

Public API
----------
    classify(service: str, error_message: str) -> ClassificationResult
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import openai

# Reuse the shared Groq configuration – no second .env load, no hardcoded keys.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_SEVERITIES = {"P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"}

SYSTEM_PROMPT = """\
You are an IT incident triage specialist. Given an incident's service name and \
error message, classify it and return a JSON object with exactly these fields:

{
  "severity": "<one of: P1 - Critical | P2 - High | P3 - Medium | P4 - Low>",
  "short_reason": "<one sentence explaining the classification>",
  "immediate_action": "<one concrete action the on-call engineer should take right now>",
  "requires_human_escalation": <true | false>
}

Severity definitions:
  P1 - Critical : Full service outage, data loss risk, security breach, or revenue impact.
  P2 - High     : Significant degradation affecting many users or a key business function.
  P3 - Medium   : Partial or intermittent issue with a workaround available.
  P4 - Low      : Cosmetic, minor, or non-urgent issue with no user impact.

Rules:
- Return ONLY the JSON object. No markdown fences, no explanation outside the JSON.
- "severity" must be exactly one of the four values above.
- "requires_human_escalation" must be a boolean (true for P1/P2, false for P3/P4).
"""

USER_PROMPT_TEMPLATE = """\
Service: {service}
Error message: {error_message}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    severity: str
    short_reason: str
    immediate_action: str
    requires_human_escalation: bool
    raw_response: str  # full model text, useful for debugging


# ── Exceptions ────────────────────────────────────────────────────────────────

class ClassificationError(Exception):
    """Raised when the classifier cannot produce a valid result."""


# ── Core function ─────────────────────────────────────────────────────────────

def classify(
    service: str,
    error_message: str,
    *,
    client: openai.OpenAI | None = None,
) -> ClassificationResult:
    """
    Classify an IT incident by severity.

    Parameters
    ----------
    service:
        Name of the affected service (e.g. "Auth-Service", "Database").
    error_message:
        Free-text description of the incident or error.
    client:
        Optional pre-built ``openai.OpenAI`` instance (used in tests to inject
        a mock).  When ``None`` the function creates one from ``config.py``.

    Returns
    -------
    ClassificationResult

    Raises
    ------
    ClassificationError
        If the input is empty, the API call fails, the response is not valid
        JSON, or required fields are missing/invalid.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    service = (service or "").strip()
    error_message = (error_message or "").strip()

    if not service:
        raise ClassificationError("'service' must not be empty.")
    if not error_message:
        raise ClassificationError("'error_message' must not be empty.")

    # ── Build client ──────────────────────────────────────────────────────────
    if client is None:
        client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    user_prompt = USER_PROMPT_TEMPLATE.format(
        service=service,
        error_message=error_message,
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0,          # deterministic classification
            response_format={"type": "json_object"},
        )
    except openai.APIStatusError as exc:
        raise ClassificationError(
            f"Groq API returned HTTP {exc.status_code}: {exc.message}"
        ) from exc
    except Exception as exc:
        raise ClassificationError(f"API request failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    logger.debug("Raw LLM response: %s", raw)

    # ── Parse and validate ────────────────────────────────────────────────────
    return _parse_response(raw)


def _parse_response(raw: str) -> ClassificationResult:
    """Parse and validate the raw JSON string from the model."""
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            f"Model returned non-JSON content: {raw!r}"
        ) from exc

    missing = [f for f in ("severity", "short_reason", "immediate_action", "requires_human_escalation") if f not in data]
    if missing:
        raise ClassificationError(
            f"Model response missing required fields: {missing}. Got: {data}"
        )

    severity = str(data["severity"]).strip()
    if severity not in VALID_SEVERITIES:
        raise ClassificationError(
            f"Invalid severity value {severity!r}. "
            f"Expected one of: {sorted(VALID_SEVERITIES)}"
        )

    requires_escalation = data["requires_human_escalation"]
    if not isinstance(requires_escalation, bool):
        # Tolerate "true"/"false" strings from models that ignore the schema.
        if str(requires_escalation).lower() == "true":
            requires_escalation = True
        elif str(requires_escalation).lower() == "false":
            requires_escalation = False
        else:
            raise ClassificationError(
                f"'requires_human_escalation' must be a boolean; got {requires_escalation!r}"
            )

    return ClassificationResult(
        severity=severity,
        short_reason=str(data["short_reason"]).strip(),
        immediate_action=str(data["immediate_action"]).strip(),
        requires_human_escalation=requires_escalation,
        raw_response=raw,
    )
