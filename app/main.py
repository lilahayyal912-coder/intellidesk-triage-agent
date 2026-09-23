"""
app/main.py – IntelliDesk Streamlit Dashboard

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC_DIR  = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

from services.severity_classifier import ClassificationError, classify
from services.runbook_retriever import RunbookRetriever
from services.escalation_agent import (
    EscalationError,
    EscalationRequest,
    escalate,
)
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

import openai

# ── Constants ─────────────────────────────────────────────────────────────────
SERVICES = [
    "Auth-Service",
    "Database",
    "Payment-Gateway",
    "Frontend-UI",
    "Network",
    "Email-Service",
]

OUTPUT_CSV = _REPO_ROOT / "output" / "triage_results.csv"
RUNBOOK_PATH = _REPO_ROOT / "data" / "runbook.txt"

SEVERITY_CONFIG = {
    "P1 - Critical": {"color": "#c0392b", "bg": "#fdecea", "badge": "🔴 P1 – Critical"},
    "P2 - High":     {"color": "#e67e22", "bg": "#fef3e2", "badge": "🟠 P2 – High"},
    "P3 - Medium":   {"color": "#f1c40f", "bg": "#fefde2", "badge": "🟡 P3 – Medium"},
    "P4 - Low":      {"color": "#27ae60", "bg": "#eafaf1", "badge": "🟢 P4 – Low"},
}


# ── Cached shared resources (one client, one retriever per session) ───────────
@st.cache_resource(show_spinner=False)
def _get_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


@st.cache_resource(show_spinner=False)
def _get_retriever() -> RunbookRetriever:
    return RunbookRetriever(RUNBOOK_PATH)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _severity_badge(severity: str) -> str:
    cfg = SEVERITY_CONFIG.get(severity)
    if cfg:
        return cfg["badge"]
    return f"⚪ {severity}"


def _severity_style(severity: str) -> dict:
    return SEVERITY_CONFIG.get(severity, {"color": "#555", "bg": "#f0f0f0"})


def _render_severity_card(severity: str) -> None:
    style = _severity_style(severity)
    badge = _severity_badge(severity)
    st.markdown(
        f"""
        <div style="
            background:{style['bg']};
            border-left: 5px solid {style['color']};
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            color: {style['color']};
        ">{badge}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_triage_result(classification, retrieval, escalation) -> None:
    """Render the full Agent 1+2+3 result in the UI."""

    # ── Severity card ─────────────────────────────────────────────────────────
    _render_severity_card(escalation.final_severity)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classifier Severity", classification.severity)
    with col2:
        st.metric("Final Severity", escalation.final_severity)

    # ── Key fields ────────────────────────────────────────────────────────────
    st.markdown("---")

    st.markdown("**🎯 Escalated Team**")
    st.info(escalation.escalated_team or "—")

    st.markdown("**📋 Short Reason**")
    st.write(classification.short_reason)

    st.markdown("**📝 Justification**")
    st.write(escalation.justification)

    # ── Next steps ────────────────────────────────────────────────────────────
    st.markdown("**🔧 Recommended Next Steps**")
    for i, step in enumerate(escalation.recommended_next_steps, 1):
        st.markdown(f"{i}. {step}")

    # ── Runbook match ─────────────────────────────────────────────────────────
    if retrieval.found and retrieval.matched_entry:
        entry = retrieval.matched_entry
        with st.expander(
            f"📖 Runbook match: {entry.sop_id} – {entry.title}  "
            f"(relevance {retrieval.relevance_score:.2f})"
        ):
            st.markdown(f"**Severity Guidance:** {entry.severity_guidance}")
            st.markdown(f"**Escalation Team:** {entry.escalation_team}")
            if entry.troubleshooting_steps:
                st.markdown("**Troubleshooting Steps:**")
                for step in entry.troubleshooting_steps:
                    st.markdown(f"- {step}")

    # ── Human escalation flag ─────────────────────────────────────────────────
    st.markdown("---")
    if escalation.requires_human_escalation:
        st.error("⚠️ **Human escalation required** – assign this ticket to the escalation team immediately.")
    else:
        st.success("✅ **No immediate human escalation required** – ticket can be handled by automated or L1 support.")


# ── Page: Analyze new incident ────────────────────────────────────────────────

def _page_analyze() -> None:
    st.subheader("🔍 Analyze a New Incident")
    st.markdown(
        "Select the affected service, describe the incident, and click **Analyze Incident** "
        "to run all three AI agents."
    )

    with st.form("incident_form"):
        service = st.selectbox("Service", SERVICES)
        error_message = st.text_area(
            "Incident description",
            placeholder="e.g. All users are unable to log in. The authentication service is returning HTTP 500.",
            height=120,
        )
        submitted = st.form_submit_button("🚀 Analyze Incident", use_container_width=True)

    if not submitted:
        return

    if not error_message.strip():
        st.warning("Please enter an incident description before analyzing.")
        return

    try:
        client   = _get_client()
        retriever = _get_retriever()
    except EnvironmentError as exc:
        st.error(f"**Configuration error:** {exc}")
        return

    with st.spinner("Agent 1: Classifying severity…"):
        try:
            classification = classify(service, error_message, client=client)
        except ClassificationError as exc:
            st.error(f"**Severity classification failed:** {exc}")
            return

    with st.spinner("Agent 2: Searching runbook…"):
        retrieval = retriever.retrieve(service, error_message)

    with st.spinner("Agent 3: Generating escalation decision…"):
        entry = retrieval.matched_entry
        req = EscalationRequest(
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
            escalation = escalate(req, client=client)
        except EscalationError as exc:
            st.error(f"**Escalation agent failed:** {exc}")
            return

    st.markdown("---")
    st.markdown("### Triage Result")
    _render_triage_result(classification, retrieval, escalation)


# ── Page: View results ────────────────────────────────────────────────────────

def _severity_color_row(row: pd.Series) -> list[str]:
    """Return per-cell CSS for a triage results row."""
    sev = str(row.get("final_severity", ""))
    style = _severity_style(sev)
    bg = style["bg"]
    return [f"background-color: {bg}" for _ in row]


def _page_results() -> None:
    st.subheader("📊 Processed Incident Results")

    if not OUTPUT_CSV.exists():
        st.info(
            "No results file found yet. Run the pipeline first:\n"
            "```\npython src/pipeline.py --limit 5\n```"
        )
        return

    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        st.info("The results file is empty.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    total   = len(df)
    errors  = df["final_severity"].str.startswith("ERROR").sum()
    p1      = (df["final_severity"] == "P1 - Critical").sum()
    p2      = (df["final_severity"] == "P2 - High").sum()
    escalated = df["requires_human_escalation"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tickets", total)
    c2.metric("🔴 P1 Critical", int(p1))
    c3.metric("🟠 P2 High", int(p2))
    c4.metric("⚠️ Needs Escalation", int(escalated))
    c5.metric("❌ Errors", int(errors))

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sev_filter = st.multiselect(
            "Filter by severity",
            options=["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"],
            default=[],
        )
    with col_f2:
        svc_filter = st.multiselect(
            "Filter by service",
            options=sorted(df["service"].unique().tolist()),
            default=[],
        )

    filtered = df.copy()
    if sev_filter:
        filtered = filtered[filtered["final_severity"].isin(sev_filter)]
    if svc_filter:
        filtered = filtered[filtered["service"].isin(svc_filter)]

    st.markdown(f"Showing **{len(filtered)}** of **{total}** tickets")

    # ── Colour-coded table ────────────────────────────────────────────────────
    display_cols = [
        "ticket_id", "service", "final_severity",
        "escalated_team", "requires_human_escalation",
    ]
    st.dataframe(
        filtered[display_cols].style.apply(_severity_color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── Expandable full detail per ticket ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Full detail")
    for _, row in filtered.iterrows():
        sev   = str(row.get("final_severity", ""))
        badge = _severity_badge(sev)
        with st.expander(f"{row['ticket_id']} | {row['service']} | {badge}"):
            st.markdown(f"**Error message:** {row['error_message']}")
            st.markdown(f"**Classifier severity:** {row['severity']}")
            st.markdown(f"**Short reason:** {row['short_reason']}")
            st.markdown(f"**Escalated team:** {row['escalated_team']}")
            st.markdown(f"**Justification:** {row['justification']}")
            steps_raw = str(row.get("recommended_next_steps", ""))
            if steps_raw:
                st.markdown("**Recommended next steps:**")
                for step in steps_raw.split(" | "):
                    st.markdown(f"- {step.strip()}")
            esc = row.get("requires_human_escalation", False)
            if str(esc).lower() in ("true", "1", "yes"):
                st.error("⚠️ Human escalation required")
            else:
                st.success("✅ No immediate escalation required")


# ── App layout ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="IntelliDesk – AI Triage Agent",
        page_icon="🚨",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🚨 IntelliDesk – AI-Powered Incident Triage & Auto-Escalation Agent")
    st.markdown(
        """
        IntelliDesk uses a three-agent AI pipeline to automatically classify incoming IT support
        tickets by severity (P1–P4), retrieve the matching Standard Operating Procedure from the
        runbook, and produce a plain-English escalation decision with recommended next steps —
        reducing triage time and ensuring critical incidents reach the right team immediately.
        """
    )
    st.markdown("---")

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Select mode",
        options=["🔍 Analyze a new incident", "📊 View processed incident results"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("")

    if mode == "🔍 Analyze a new incident":
        _page_analyze()
    else:
        _page_results()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
        "IntelliDesk · Groq + LLaMA 3.1 · TF-IDF Runbook Retrieval · Built with Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
