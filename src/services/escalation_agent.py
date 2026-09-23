"""
escalation_agent.py – Agent 3: Escalation and Explanation Agent

Consumes the outputs of Agent 1 (severity classifier) and Agent 2 (runbook
retriever) and uses the LLM to produce a final, human-readable escalation
decision with a justification, next steps, and confirmed escalation team.

Rules baked into the system prompt:
  - Use the runbook as primary operational guidance.
  - Do not invent an escalation team when the runbook provides one.
  - Preserve the classifier severity unless strong evidence justifies a change.
  - Explain the decision in plain English.
  - Never claim the incident has been fixed.

Public API
----------
    escalate(request: EscalationRequest) -> EscalationResult
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import openai

# Reuse the shared Groq configuration – no second .env load.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EscalationRequest:
    """Everything Agent 3 needs to produce an escalation decision."""
    service: str
    error_message: str
    # Agent 1 outputs
    classifier_severity: str          # e.g. "P1 - Critical"
    classifier_short_reason: str
    # Agent 2 outputs (may be None when no runbook entry was found)
    runbook_sop_id: Optional[str]     # e.g. "SOP-002"
    runbook_title: Optional[str]
    runbook_escalation_team: Optional[str]
    runbook_recommended_action: Optional[str]
    runbook_severity_guidance: Optional[str]


@dataclass
class EscalationResult:
    escalated_team: str
    final_severity: str
    justification: str
    recommended_next_steps: list[str]
    requires_human_escalation: bool
    raw_response: str                 # full model text for debugging


# ── Exceptions ────────────────────────────────────────────────────────────────

class EscalationError(Exception):
    """Raised when the escalation agent cannot produce a valid result."""


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior IT operations engineer responsible for finalising incident \
escalation decisions. You will be given:

1. The affected service and original error message.
2. A severity classification (P1–P4) from an automated classifier.
3. A matching Standard Operating Procedure (SOP) entry from the runbook, if available.

Your job is to return a single JSON object with exactly these fields:

{
  "escalated_team": "<team name to action this incident>",
  "final_severity": "<P1 - Critical | P2 - High | P3 - Medium | P4 - Low>",
  "justification": "<2-4 sentences explaining the decision in plain English>",
  "recommended_next_steps": ["<step 1>", "<step 2>", ...],
  "requires_human_escalation": <true | false>
}

Rules you MUST follow:
- If the runbook provides an escalation team, use it verbatim. Do NOT invent a team.
- If the runbook does not provide a team, derive a sensible one from the service name.
- Preserve the classifier severity unless the runbook or the error evidence \
  clearly justifies a change; if you change it, explain why in the justification.
- "recommended_next_steps" must be a JSON array of strings (2–5 items).
- "requires_human_escalation" must be true for P1 and P2; false for P3 and P4 \
  unless the runbook explicitly says to escalate.
- NEVER claim the incident has been fixed or resolved.
- Return ONLY the JSON object. No markdown fences, no extra text.
"""

_USER_PROMPT_TEMPLATE = """\
=== INCIDENT ===
Service:       {service}
Error message: {error_message}

=== CLASSIFIER OUTPUT (Agent 1) ===
Severity:      {classifier_severity}
Reason:        {classifier_short_reason}

=== RUNBOOK MATCH (Agent 2) ===
{runbook_section}
"""

_RUNBOOK_FOUND_TEMPLATE = """\
SOP ID:            {sop_id}
Title:             {title}
Escalation Team:   {escalation_team}
Severity Guidance: {severity_guidance}
Recommended Step:  {recommended_action}"""

_RUNBOOK_NOT_FOUND = "No matching runbook entry was found."

# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_SEVERITIES = {"P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"}


def _build_user_prompt(req: EscalationRequest) -> str:
    if req.runbook_sop_id:
        runbook_section = _RUNBOOK_FOUND_TEMPLATE.format(
            sop_id=req.runbook_sop_id or "",
            title=req.runbook_title or "",
            escalation_team=req.runbook_escalation_team or "",
            severity_guidance=req.runbook_severity_guidance or "",
            recommended_action=req.runbook_recommended_action or "",
        )
    else:
        runbook_section = _RUNBOOK_NOT_FOUND

    return _USER_PROMPT_TEMPLATE.format(
        service=req.service,
        error_message=req.error_message,
        classifier_severity=req.classifier_severity,
        classifier_short_reason=req.classifier_short_reason,
        runbook_section=runbook_section,
    )


def _parse_response(raw: str) -> EscalationResult:
    """Parse and validate the raw JSON from the model."""
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EscalationError(
            f"Model returned non-JSON content: {raw!r}"
        ) from exc

    required = [
        "escalated_team",
        "final_severity",
        "justification",
        "recommended_next_steps",
        "requires_human_escalation",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise EscalationError(
            f"Model response missing required fields: {missing}. Got: {data}"
        )

    final_severity = str(data["final_severity"]).strip()
    if final_severity not in _VALID_SEVERITIES:
        raise EscalationError(
            f"Invalid final_severity {final_severity!r}. "
            f"Expected one of: {sorted(_VALID_SEVERITIES)}"
        )

    steps = data["recommended_next_steps"]
    if not isinstance(steps, list) or not steps:
        raise EscalationError(
            f"'recommended_next_steps' must be a non-empty list; got {steps!r}"
        )

    requires = data["requires_human_escalation"]
    if not isinstance(requires, bool):
        if str(requires).lower() == "true":
            requires = True
        elif str(requires).lower() == "false":
            requires = False
        else:
            raise EscalationError(
                f"'requires_human_escalation' must be boolean; got {requires!r}"
            )

    return EscalationResult(
        escalated_team=str(data["escalated_team"]).strip(),
        final_severity=final_severity,
        justification=str(data["justification"]).strip(),
        recommended_next_steps=[str(s).strip() for s in steps],
        requires_human_escalation=requires,
        raw_response=raw,
    )


# ── Core function ─────────────────────────────────────────────────────────────

def escalate(
    request: EscalationRequest,
    *,
    client: openai.OpenAI | None = None,
) -> EscalationResult:
    """
    Produce a final escalation decision for an incident.

    Parameters
    ----------
    request:
        Populated ``EscalationRequest`` combining Agent 1 and Agent 2 outputs.
    client:
        Optional pre-built ``openai.OpenAI`` instance (injected in tests).

    Returns
    -------
    EscalationResult

    Raises
    ------
    EscalationError
        On empty inputs, API failure, non-JSON response, or invalid fields.
    """
    if not (request.service or "").strip():
        raise EscalationError("'service' must not be empty.")
    if not (request.error_message or "").strip():
        raise EscalationError("'error_message' must not be empty.")
    if not (request.classifier_severity or "").strip():
        raise EscalationError("'classifier_severity' must not be empty.")

    if client is None:
        client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    user_prompt = _build_user_prompt(request)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=768,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except openai.APIStatusError as exc:
        raise EscalationError(
            f"Groq API returned HTTP {exc.status_code}: {exc.message}"
        ) from exc
    except Exception as exc:
        raise EscalationError(f"API request failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    logger.debug("Raw escalation response: %s", raw)
    return _parse_response(raw)
