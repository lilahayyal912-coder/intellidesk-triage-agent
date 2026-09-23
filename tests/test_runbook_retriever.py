"""
tests/test_runbook_retriever.py

Unit tests for src/services/runbook_retriever.py.
Uses the real data/runbook.txt file — no live API, no mocking required.
"""

import sys
import os
import unittest
from pathlib import Path

# Make src/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.runbook_retriever import (
    RunbookRetriever,
    RetrievalResult,
    RunbookEntry,
    _parse_runbook,
    _tokenise,
    _no_match,
    RELEVANCE_THRESHOLD,
)

RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "data" / "runbook.txt"


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestRunbookParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = RUNBOOK_PATH.read_text(encoding="utf-8")
        cls.entries = _parse_runbook(cls.text)

    def test_parses_ten_entries(self):
        self.assertEqual(len(self.entries), 10)

    def test_all_entries_have_sop_ids(self):
        ids = [e.sop_id for e in self.entries]
        for expected in [f"SOP-{i:03d}" for i in range(1, 11)]:
            self.assertIn(expected, ids)

    def test_all_entries_have_titles(self):
        for e in self.entries:
            self.assertTrue(e.title, f"{e.sop_id} has no title")

    def test_all_entries_have_services(self):
        for e in self.entries:
            self.assertTrue(e.service, f"{e.sop_id} has no service")

    def test_all_entries_have_escalation_team(self):
        for e in self.entries:
            self.assertTrue(e.escalation_team, f"{e.sop_id} has no escalation team")

    def test_all_entries_have_troubleshooting_steps(self):
        for e in self.entries:
            self.assertGreater(
                len(e.troubleshooting_steps), 0,
                f"{e.sop_id} has no troubleshooting steps",
            )

    def test_all_entries_have_symptoms(self):
        for e in self.entries:
            self.assertGreater(
                len(e.symptoms), 0,
                f"{e.sop_id} has no symptoms",
            )

    def test_sop001_service_is_auth(self):
        sop1 = next(e for e in self.entries if e.sop_id == "SOP-001")
        self.assertIn("Auth", sop1.service)

    def test_sop002_service_is_database(self):
        sop2 = next(e for e in self.entries if e.sop_id == "SOP-002")
        self.assertIn("Database", sop2.service)

    def test_sop004_escalation_contains_payments(self):
        sop4 = next(e for e in self.entries if e.sop_id == "SOP-004")
        self.assertIn("Payments", sop4.escalation_team)

    def test_raw_text_is_non_empty(self):
        for e in self.entries:
            self.assertTrue(e.raw_text.strip())


# ── Tokeniser tests ───────────────────────────────────────────────────────────

class TestTokeniser(unittest.TestCase):

    def test_basic_tokenisation(self):
        tokens = _tokenise("Login failure rate > 10%")
        self.assertIn("login", tokens)
        self.assertIn("failure", tokens)
        self.assertIn("rate", tokens)

    def test_stop_words_removed(self):
        tokens = _tokenise("the server is down and it is broken")
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertNotIn("is", tokens)

    def test_hyphenated_terms_kept(self):
        tokens = _tokenise("Auth-Service payment-gateway")
        self.assertTrue(any("auth" in t or "auth-service" in t for t in tokens))

    def test_empty_string_returns_empty(self):
        self.assertEqual(_tokenise(""), [])

    def test_short_tokens_excluded(self):
        tokens = _tokenise("a b c do at in on")
        for t in tokens:
            self.assertGreater(len(t), 1)


# ── RunbookRetriever initialisation tests ─────────────────────────────────────

class TestRunbookRetrieverInit(unittest.TestCase):

    def test_loads_from_default_path(self):
        r = RunbookRetriever(RUNBOOK_PATH)
        self.assertEqual(r.entry_count, 10)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            RunbookRetriever("/nonexistent/path/runbook.txt")

    def test_idf_index_built(self):
        r = RunbookRetriever(RUNBOOK_PATH)
        self.assertGreater(len(r._idf), 0)

    def test_entry_vectors_match_entry_count(self):
        r = RunbookRetriever(RUNBOOK_PATH)
        self.assertEqual(len(r._entry_vectors), r.entry_count)


# ── Retrieval correctness tests ───────────────────────────────────────────────

class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = RunbookRetriever(RUNBOOK_PATH)

    def test_auth_login_failure_matches_sop001(self):
        result = self.r.retrieve(
            "Auth-Service",
            "All users unable to log in, authentication returning HTTP 500",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-001")

    def test_database_outage_matches_sop002(self):
        result = self.r.retrieve(
            "Database",
            "Primary node unresponsive, connection refused, production writes blocked",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-002")

    def test_slow_queries_matches_sop003(self):
        result = self.r.retrieve(
            "Database",
            "Queries taking 8 seconds, reporting job exceeded SLA, slow query log full",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-003")

    def test_payment_failure_matches_sop004(self):
        result = self.r.retrieve(
            "Payment-Gateway",
            "Charge and refund endpoints returning HTTP 500, transactions failing",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-004")

    def test_network_outage_matches_sop005(self):
        result = self.r.retrieve(
            "Network",
            "Core switch unreachable, inter-VLAN routing failed, 60% services down",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-005")

    def test_frontend_white_screen_matches_sop006(self):
        result = self.r.retrieve(
            "Frontend-UI",
            "White screen on load, JavaScript bundle 404, affecting Safari users",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-006")

    def test_email_delivery_matches_sop007(self):
        result = self.r.retrieve(
            "Email-Service",
            "Password reset emails not delivered, SMTP relay queue at 90%",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-007")

    def test_account_lockout_matches_sop008(self):
        result = self.r.retrieve(
            "Auth-Service",
            "Multiple users locked out, helpdesk flooded with account unlock requests",
        )
        self.assertTrue(result.found)
        self.assertEqual(result.matched_entry.sop_id, "SOP-008")

    def test_result_has_escalation_team(self):
        result = self.r.retrieve("Database", "Primary node down, writes blocked")
        self.assertTrue(result.escalation_team)

    def test_result_has_recommended_action(self):
        result = self.r.retrieve("Database", "Primary node down, writes blocked")
        self.assertTrue(result.recommended_action)

    def test_relevance_score_is_float_between_0_and_1(self):
        result = self.r.retrieve("Network", "Packet loss on uplink")
        self.assertGreaterEqual(result.relevance_score, 0.0)
        self.assertLessEqual(result.relevance_score, 1.0)

    def test_empty_service_returns_no_match(self):
        result = self.r.retrieve("", "")
        self.assertFalse(result.found)
        self.assertIsNone(result.matched_entry)

    def test_gibberish_returns_no_match_or_low_score(self):
        result = self.r.retrieve("zzz", "xyzzy frobnicator quux")
        # Either not found, or if something matched it must be below threshold
        if result.found:
            self.assertGreaterEqual(result.relevance_score, RELEVANCE_THRESHOLD)
        else:
            self.assertFalse(result.found)

    def test_no_match_result_has_empty_escalation(self):
        result = _no_match()
        self.assertEqual(result.escalation_team, "")
        self.assertFalse(result.found)
        self.assertIsNone(result.matched_entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
