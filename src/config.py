"""
config.py – Load and expose environment variables for IntelliDesk.

Required .env keys
------------------
GROQ_API_KEY – Groq API key (https://console.groq.com/keys)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return the value of *key* from the environment, or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{key}'. "
            f"Copy .env.example to .env and set a value for {key}."
        )
    return value


# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = _require("GROQ_API_KEY")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_MODEL: str = "openai/gpt-oss-20b"
