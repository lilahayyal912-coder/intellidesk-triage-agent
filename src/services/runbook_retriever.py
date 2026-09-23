"""
runbook_retriever.py – Agent 2: Runbook / SOP Retrieval

Parses data/runbook.txt into structured entries, then retrieves the most
relevant SOP for a given (service, error_message) pair using TF-IDF cosine
similarity.  No external API, no embeddings model, no vector database.

Public API
----------
    retriever = RunbookRetriever()                    # loads & indexes once
    result    = retriever.retrieve(service, error_message)
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Default path to the runbook (relative to this file's package root) ────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNBOOK_PATH = _REPO_ROOT / "data" / "runbook.txt"

# Minimum cosine similarity to consider a match meaningful.
RELEVANCE_THRESHOLD = 0.05


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RunbookEntry:
    sop_id: str                          # e.g. "SOP-001"
    title: str                           # e.g. "Authentication and Login Failures"
    service: str                         # e.g. "Auth-Service"
    symptoms: list[str]
    likely_cause: list[str]
    troubleshooting_steps: list[str]
    escalation_team: str
    severity_guidance: str
    raw_text: str                        # full block, useful for display


@dataclass
class RetrievalResult:
    matched_entry: Optional[RunbookEntry]
    relevance_score: float               # cosine similarity [0, 1]
    escalation_team: str                 # "" when no match
    recommended_action: str             # first troubleshooting step, or ""
    found: bool                          # False when score < threshold


# ── Tokeniser / TF-IDF helpers ────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the and or but if in on at to for of with is are was were be been "
    "being have has had do does did will would could should may might shall "
    "from by this that these those it its we they them their all any more "
    "some such not no nor so yet both either each few further into through "
    "during before after above below between out off over under again then "
    "once only own same than too very just because as until while".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words and short tokens."""
    tokens = re.findall(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    shared = set(vec_a) & set(vec_b)
    if not shared:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Runbook parser ────────────────────────────────────────────────────────────

_SEPARATOR_RE = re.compile(r"^─{10,}$")
_SOP_HEADER_RE = re.compile(r"^(SOP-\d+):\s*(.+)$")
_FIELD_RE = re.compile(r"^([A-Za-z ]+):\s*(.*)$")


def _parse_runbook(text: str) -> list[RunbookEntry]:
    """Split the runbook text on separator lines and parse each SOP block.

    The runbook format wraps each SOP like this:
        ──────────────
        SOP-NNN: Title
        ──────────────
        Service: ...
        Symptoms:
          - ...

    So the title and its body are separated by a second separator line.
    We merge consecutive non-empty fragments so each SOP title stays with
    its field content.
    """
    # Split on separator lines into raw fragments
    fragments: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _SEPARATOR_RE.match(line.strip()):
            fragments.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if current:
        fragments.append("\n".join(current).strip())

    # Merge a bare SOP header fragment with the body fragment that follows it
    blocks: list[str] = []
    i = 0
    while i < len(fragments):
        frag = fragments[i].strip()
        if not frag:
            i += 1
            continue
        # If this fragment is just a SOP header line, merge with next fragment
        if _SOP_HEADER_RE.match(frag) and i + 1 < len(fragments):
            next_frag = fragments[i + 1].strip()
            blocks.append(frag + "\n" + next_frag)
            i += 2
        else:
            blocks.append(frag)
            i += 1

    entries: list[RunbookEntry] = []
    for block in blocks:
        entry = _parse_block(block)
        if entry:
            entries.append(entry)
    return entries


def _parse_block(block: str) -> Optional[RunbookEntry]:
    """Parse a single SOP block into a RunbookEntry."""
    lines = block.strip().splitlines()
    if not lines:
        return None

    # First non-empty line must be the SOP header.
    header_line = next((l for l in lines if l.strip()), "")
    m = _SOP_HEADER_RE.match(header_line.strip())
    if not m:
        return None
    sop_id, title = m.group(1), m.group(2).strip()

    service = ""
    symptoms: list[str] = []
    likely_cause: list[str] = []
    steps: list[str] = []
    escalation_team = ""
    severity_guidance = ""

    current_section = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Section header (e.g. "Symptoms:", "Escalation Team:   ...")
        field_m = _FIELD_RE.match(stripped)
        if field_m:
            key = field_m.group(1).strip().lower()
            value = field_m.group(2).strip()

            if key == "service":
                service = value
                current_section = ""
            elif key == "symptoms":
                current_section = "symptoms"
            elif key == "likely cause":
                current_section = "cause"
            elif key == "troubleshooting steps":
                current_section = "steps"
            elif key == "escalation team":
                escalation_team = value
                current_section = ""
            elif key == "severity guidance":
                severity_guidance = value
                current_section = ""
            continue

        # Bullet / numbered list item
        item = re.sub(r"^[-•*\d]+[.)]\s*", "", stripped).strip()
        if item:
            if current_section == "symptoms":
                symptoms.append(item)
            elif current_section == "cause":
                likely_cause.append(item)
            elif current_section == "steps":
                steps.append(item)

    return RunbookEntry(
        sop_id=sop_id,
        title=title,
        service=service,
        symptoms=symptoms,
        likely_cause=likely_cause,
        troubleshooting_steps=steps,
        escalation_team=escalation_team,
        severity_guidance=severity_guidance,
        raw_text=block,
    )


# ── Main retriever class ──────────────────────────────────────────────────────

class RunbookRetriever:
    """
    Loads and indexes a runbook file, then retrieves the best-matching SOP
    entry for a given incident using TF-IDF cosine similarity.

    Parameters
    ----------
    runbook_path:
        Path to the runbook text file.  Defaults to ``data/runbook.txt``
        relative to the repository root.
    """

    def __init__(self, runbook_path: str | Path = DEFAULT_RUNBOOK_PATH) -> None:
        self._path = Path(runbook_path)
        self.entries: list[RunbookEntry] = []
        self._entry_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._load_and_index()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_and_index(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Runbook not found at {self._path}. "
                "Ensure data/runbook.txt exists."
            )
        text = self._path.read_text(encoding="utf-8")
        self.entries = _parse_runbook(text)
        if not self.entries:
            raise ValueError("No SOP entries could be parsed from the runbook.")
        self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        """Build IDF weights and TF vectors for every entry."""
        N = len(self.entries)
        # Each entry's "document" = title + service + all bullets combined.
        docs: list[list[str]] = []
        for entry in self.entries:
            combined = " ".join(
                [entry.title, entry.service]
                + entry.symptoms
                + entry.likely_cause
                + entry.troubleshooting_steps
                + [entry.escalation_team, entry.severity_guidance]
            )
            docs.append(_tokenise(combined))

        # IDF: log(N / df) with +1 smoothing
        df: Counter = Counter()
        for tokens in docs:
            for term in set(tokens):
                df[term] += 1
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

        # TF-IDF vectors per entry
        self._entry_vectors = []
        for tokens in docs:
            tf = _term_freq(tokens)
            tfidf = {t: tf[t] * self._idf.get(t, 1.0) for t in tf}
            self._entry_vectors.append(tfidf)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, service: str, error_message: str) -> RetrievalResult:
        """
        Return the best-matching runbook entry for the given incident.

        Parameters
        ----------
        service:
            Affected service name (e.g. "Auth-Service").
        error_message:
            Free-text incident description.

        Returns
        -------
        RetrievalResult
            ``found=False`` when the best score is below RELEVANCE_THRESHOLD.
        """
        service = (service or "").strip()
        error_message = (error_message or "").strip()

        # Build query vector.  Weight the service name 3× to favour exact
        # service matches, then fold in the error message tokens.
        query_tokens = _tokenise(service) * 3 + _tokenise(error_message)
        if not query_tokens:
            return _no_match()

        qt = _term_freq(query_tokens)
        query_vec = {t: qt[t] * self._idf.get(t, 1.0) for t in qt}

        # Score every entry
        scored = [
            (idx, _cosine(query_vec, vec))
            for idx, vec in enumerate(self._entry_vectors)
        ]
        best_idx, best_score = max(scored, key=lambda x: x[1])

        if best_score < RELEVANCE_THRESHOLD:
            return _no_match()

        entry = self.entries[best_idx]
        first_step = entry.troubleshooting_steps[0] if entry.troubleshooting_steps else ""

        return RetrievalResult(
            matched_entry=entry,
            relevance_score=round(best_score, 4),
            escalation_team=entry.escalation_team,
            recommended_action=first_step,
            found=True,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        return len(self.entries)


def _no_match() -> RetrievalResult:
    return RetrievalResult(
        matched_entry=None,
        relevance_score=0.0,
        escalation_team="",
        recommended_action="",
        found=False,
    )
