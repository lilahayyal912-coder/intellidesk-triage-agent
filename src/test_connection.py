"""
test_connection.py – Smoke-test the Groq API connection.

Usage:
    python src/test_connection.py

Sends a single triage prompt to GROQ_MODEL via Groq's OpenAI-compatible
endpoint and prints the model's response.
"""

import openai

try:
    from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
except EnvironmentError as exc:
    raise SystemExit(f"[CONFIG ERROR] {exc}") from exc

PROMPT = "Classify this as high or low priority: server is down"

print(f"Model : {GROQ_MODEL}")
print(f"Endpoint: {GROQ_BASE_URL}")
print(f"Prompt: {PROMPT}")
print("-" * 60)

try:
    client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=256,
    )

    reply = response.choices[0].message.content.strip()
    print(f"Response:\n{reply}")

except openai.APIStatusError as exc:
    raise SystemExit(
        f"[API ERROR] Groq returned HTTP {exc.status_code}.\n"
        f"  Model : {GROQ_MODEL}\n"
        f"  Reason: {exc.message}"
    ) from exc
except Exception as exc:
    raise SystemExit(
        f"[API ERROR] Request to Groq failed.\n"
        f"  Model : {GROQ_MODEL}\n"
        f"  Reason: {exc}"
    ) from exc
