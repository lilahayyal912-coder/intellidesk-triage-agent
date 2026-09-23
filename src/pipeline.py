"""
pipeline.py – IntelliDesk end-to-end triage pipeline.

Workflow
--------
1. Load incidents from data/incidents.csv.
2. For each incident:
   a. Agent 1 – Severity Classifier    (Groq LLM call)
   b. Agent 2 – Runbook Retriever      (local TF-IDF, no API call)
   c. Agent 3 – Escalation Agent       (Groq LLM call)
3. Write combined results to output/triage_results.csv.

Usage
-----
    # Full run
    python src/pipeline.py

    # Quick smoke-test with the first 2 tickets only
    python src/pipeline.py --limit 2

    # Custom paths
    python src/pipeline.py --incidents data/incidents.csv --output output/triage_results.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import openai
import pandas as pd

# ── Make src/ importable regardless of working directory ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from services.severity_classifier import (
    ClassificationError,
    classify,
)
from services.runbook_retriever import RunbookRetriever
from services.escalation_agent import (
    EscalationError,
    EscalationRequest,
    escalate,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCIDENTS = _REPO_ROOT / "data" / "incidents.csv"
DEFAULT_OUTPUT    = _REPO_ROOT / "output" / "triage_results.csv"
DEFAULT_RUNBOOK   = _REPO_ROOT / "data" / "runbook.txt"

# ── Output columns (in order) ─────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "ticket_id",
    "service",
    "error_message",
    "severity",
    "short_reason",
    "escalated_team",
    "final_severity",
    "justification",
    "recommended_next_steps",
    "requires_human_escalation",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Per-ticket processing ─────────────────────────────────────────────────────

def _process_ticket(
    row: pd.Series,
    retriever: RunbookRetriever,
    client: openai.OpenAI,
) -> dict:
    """
    Run Agents 1-3 for a single ticket row.

    Returns a flat dict ready to become a DataFrame row.
    On failure, returns a row with ERROR markers so the batch continues.
    """
    ticket_id     = row["ticket_id"]
    service       = str(row["service"])
    error_message = str(row["error_message"])

    logger.info("[%s] Starting triage  service=%s", ticket_id, service)

    # ── Agent 1: Severity Classifier ──────────────────────────────────────────
    try:
        classification = classify(service, error_message, client=client)
        logger.info(
            "[%s] Agent 1 → %s  escalate=%s",
            ticket_id,
            classification.severity,
            classification.requires_human_escalation,
        )
    except ClassificationError as exc:
        logger.error("[%s] Agent 1 FAILED: %s", ticket_id, exc)
        return _error_row(ticket_id, service, error_message, stage="Agent1", reason=str(exc))

    # ── Agent 2: Runbook Retriever (local, no API) ────────────────────────────
    retrieval = retriever.retrieve(service, error_message)
    if retrieval.found:
        logger.info(
            "[%s] Agent 2 → %s (score=%.3f)",
            ticket_id,
            retrieval.matched_entry.sop_id,
            retrieval.relevance_score,
        )
    else:
        logger.warning("[%s] Agent 2 → no runbook match found", ticket_id)

    # ── Agent 3: Escalation Agent ─────────────────────────────────────────────
    entry = retrieval.matched_entry
    esc_request = EscalationRequest(
        service=service,
        error_message=error_message,
        classifier_severity=classification.severity,
        classifier_short_reason=classification.short_reason,
        runbook_sop_id=entry.sop_id if entry else None,
        runbook_title=entry.title if entry else None,
        runbook_escalation_team=entry.escalation_team if entry else None,
        runbook_recommended_action=entry.troubleshooting_steps[0] if entry and entry.troubleshooting_steps else None,
        runbook_severity_guidance=entry.severity_guidance if entry else None,
    )

    try:
        escalation = escalate(esc_request, client=client)
        logger.info(
            "[%s] Agent 3 → team=%s  final=%s",
            ticket_id,
            escalation.escalated_team[:40],
            escalation.final_severity,
        )
    except EscalationError as exc:
        logger.error("[%s] Agent 3 FAILED: %s", ticket_id, exc)
        return _error_row(ticket_id, service, error_message, stage="Agent3", reason=str(exc))

    # ── Assemble output row ───────────────────────────────────────────────────
    return {
        "ticket_id":                ticket_id,
        "service":                  service,
        "error_message":            error_message,
        "severity":                 classification.severity,
        "short_reason":             classification.short_reason,
        "escalated_team":           escalation.escalated_team,
        "final_severity":           escalation.final_severity,
        "justification":            escalation.justification,
        "recommended_next_steps":   " | ".join(escalation.recommended_next_steps),
        "requires_human_escalation": escalation.requires_human_escalation,
    }


def _error_row(
    ticket_id: str,
    service: str,
    error_message: str,
    stage: str,
    reason: str,
) -> dict:
    """Placeholder row used when one ticket fails, keeping the batch going."""
    return {
        "ticket_id":                ticket_id,
        "service":                  service,
        "error_message":            error_message,
        "severity":                 f"ERROR ({stage})",
        "short_reason":             reason,
        "escalated_team":           "",
        "final_severity":           f"ERROR ({stage})",
        "justification":            f"Pipeline error at {stage}: {reason}",
        "recommended_next_steps":   "Manual review required.",
        "requires_human_escalation": True,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    incidents_path: Path = DEFAULT_INCIDENTS,
    output_path: Path = DEFAULT_OUTPUT,
    runbook_path: Path = DEFAULT_RUNBOOK,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Execute the full triage pipeline and return the results DataFrame.

    Parameters
    ----------
    incidents_path: Path to incidents CSV.
    output_path:    Where to write triage_results.csv.
    runbook_path:   Path to the runbook text file.
    limit:          If set, process only the first *limit* tickets.
    """
    # ── Load incidents ────────────────────────────────────────────────────────
    logger.info("Loading incidents from %s", incidents_path)
    df = pd.read_csv(incidents_path)
    required_cols = {"ticket_id", "service", "error_message"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"incidents CSV is missing columns: {missing}")

    if limit:
        df = df.head(limit)
        logger.info("--limit %d: processing %d ticket(s)", limit, len(df))
    else:
        logger.info("Processing %d ticket(s)", len(df))

    # ── Initialise shared resources (one client, one retriever) ──────────────
    logger.info("Initialising Groq client  model=%s", GROQ_MODEL)
    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    logger.info("Loading and indexing runbook from %s", runbook_path)
    retriever = RunbookRetriever(runbook_path)
    logger.info("Runbook indexed: %d SOP entries", retriever.entry_count)

    # ── Process each ticket ───────────────────────────────────────────────────
    results: list[dict] = []
    total   = len(df)
    success = 0
    errors  = 0

    for idx, row in df.iterrows():
        result = _process_ticket(row, retriever, client)
        results.append(result)
        if result["severity"].startswith("ERROR"):
            errors += 1
        else:
            success += 1

    # ── Save output ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False)

    logger.info(
        "Pipeline complete – %d succeeded, %d failed. Output → %s",
        success,
        errors,
        output_path,
    )
    return out_df


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IntelliDesk AI Incident Triage Pipeline",
    )
    parser.add_argument(
        "--incidents",
        type=Path,
        default=DEFAULT_INCIDENTS,
        help=f"Path to incidents CSV (default: {DEFAULT_INCIDENTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--runbook",
        type=Path,
        default=DEFAULT_RUNBOOK,
        help=f"Path to runbook text file (default: {DEFAULT_RUNBOOK})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N tickets (useful for smoke testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        incidents_path=args.incidents,
        output_path=args.output,
        runbook_path=args.runbook,
        limit=args.limit,
    )
