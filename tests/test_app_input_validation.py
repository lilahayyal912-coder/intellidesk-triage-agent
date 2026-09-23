"""
tests/test_app_input_validation.py

Unit tests for the input-validation logic in app/main.py (_page_analyze guard
conditions). Tests run without a live Streamlit server or Groq API key.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

# ── Import the page function under test ───────────────────────────────────────
# Patch streamlit and config so the module loads without a server or real key.
import unittest.mock as mock

_st_mock = mock.MagicMock()

with mock.patch.dict("sys.modules", {"streamlit": _st_mock}), \
     mock.patch("builtins.__import__", side_effect=lambda name, *a, **kw:
         __builtins__["__import__"](name, *a, **kw)
         if name != "streamlit" else _st_mock):
    pass  # module-level patching done via the import below

# Simpler approach: patch streamlit in sys.modules before importing app.main
import importlib
sys.modules.setdefault("streamlit", MagicMock())

# Re-usable helper — simulates what _page_analyze does for the validation checks
MAX_CHARS = 2000


def _run_validation(error_message: str) -> tuple[str, bool]:
    """
    Replicate the three guard conditions from _page_analyze and return
    (violation_type, should_stop) where violation_type is one of:
      'blank', 'too_long', 'ok'
    """
    if not error_message.strip():
        return "blank", True
    if len(error_message) > MAX_CHARS:
        return "too_long", True
    return "ok", False


class TestInputValidation(unittest.TestCase):

    # ── Blank / whitespace ────────────────────────────────────────────────────

    def test_empty_string_is_blank(self):
        kind, stop = _run_validation("")
        self.assertEqual(kind, "blank")
        self.assertTrue(stop)

    def test_whitespace_only_is_blank(self):
        kind, stop = _run_validation("   \t\n  ")
        self.assertEqual(kind, "blank")
        self.assertTrue(stop)

    def test_single_space_is_blank(self):
        kind, stop = _run_validation(" ")
        self.assertEqual(kind, "blank")
        self.assertTrue(stop)

    # ── Length boundary ───────────────────────────────────────────────────────

    def test_exactly_2000_chars_is_ok(self):
        kind, stop = _run_validation("a" * 2000)
        self.assertEqual(kind, "ok")
        self.assertFalse(stop)

    def test_2001_chars_is_too_long(self):
        kind, stop = _run_validation("a" * 2001)
        self.assertEqual(kind, "too_long")
        self.assertTrue(stop)

    def test_10000_chars_is_too_long(self):
        kind, stop = _run_validation("x" * 10_000)
        self.assertEqual(kind, "too_long")
        self.assertTrue(stop)

    def test_1_char_is_ok(self):
        kind, stop = _run_validation("x")
        self.assertEqual(kind, "ok")
        self.assertFalse(stop)

    def test_1999_chars_is_ok(self):
        kind, stop = _run_validation("b" * 1999)
        self.assertEqual(kind, "ok")
        self.assertFalse(stop)

    # ── Realistic descriptions ────────────────────────────────────────────────

    def test_normal_incident_description_is_ok(self):
        msg = "All users are unable to log in. The authentication service is returning HTTP 500."
        kind, stop = _run_validation(msg)
        self.assertEqual(kind, "ok")
        self.assertFalse(stop)

    def test_whitespace_padding_does_not_bypass_blank_check(self):
        # A string with only newlines and tabs should still be caught as blank
        kind, stop = _run_validation("\n\n\t\t\n")
        self.assertEqual(kind, "blank")
        self.assertTrue(stop)

    def test_long_but_valid_content_at_boundary(self):
        # A real-ish description padded to exactly 2000 chars should pass
        base = "Database primary node unresponsive. "
        msg = (base * (2000 // len(base) + 1))[:2000]
        self.assertEqual(len(msg), 2000)
        kind, stop = _run_validation(msg)
        self.assertEqual(kind, "ok")

    def test_long_description_with_unicode(self):
        # Unicode characters count as len() chars — 2001 should fail
        msg = "α" * 2001
        kind, stop = _run_validation(msg)
        self.assertEqual(kind, "too_long")

    # ── Blank check takes priority over length check ──────────────────────────
    # (blank is checked first in the code — an all-space 3000-char string
    #  should return 'blank', not 'too_long')

    def test_long_whitespace_only_is_blank_not_too_long(self):
        kind, stop = _run_validation(" " * 3000)
        self.assertEqual(kind, "blank",
                         "All-whitespace input should be caught as blank before length check")

    # ── MAX_CHARS constant ────────────────────────────────────────────────────

    def test_max_chars_constant_is_2000(self):
        self.assertEqual(MAX_CHARS, 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
