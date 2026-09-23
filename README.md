# IntelliDesk – AI Incident Triage & Escalation Agent

IntelliDesk is an AI-powered incident triage and escalation agent designed to streamline IT service desk operations. It leverages IBM watsonx.ai's large language models together with LangFlow orchestration to automatically classify incoming support tickets, assess severity, suggest resolutions from a knowledge base, and route unresolved incidents to the appropriate human escalation tier — all with minimal manual intervention. By combining natural language understanding with structured workflow automation, IntelliDesk reduces mean time to resolution (MTTR), cuts repetitive triage overhead for support engineers, and ensures critical incidents are never missed or misrouted.

## Project Structure

```
IntelliDesk-triage-agent/
├── data/           # Raw and processed input data (tickets, knowledge base, etc.)
├── src/            # Source code – agents, pipelines, utilities
├── output/         # Generated reports, escalation logs, model outputs
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd IntelliDesk-triage-agent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy `.env.example` to `.env` and set `GROQ_API_KEY` to your Groq API key (obtainable free at <https://console.groq.com/keys>).

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | OpenAI-compatible HTTP client (used with Groq's endpoint) |
| `requests` | HTTP utilities for webhook/API calls |
| `pandas` | Data ingestion and ticket preprocessing |
| `python-dotenv` | Environment variable management |

## Note on Model Selection

This project was designed around **IBM Granite** models (specifically `ibm-granite/granite-3.1-8b-instruct`) as the target LLM, in line with the IBM watsonx.ai tech-stack requirement.
During the build window, free-tier hosted access to Granite through the Hugging Face Inference Providers and NVIDIA NIM was unavailable — both routes returned capacity or access errors that could not be resolved without a paid tier or approved waitlist access.
To keep the agentic pipeline fully functional and demonstrable, **Llama 3.1 8B Instant** (`llama-3.1-8b-instant`) was substituted via **Groq's free Inference API**, which exposes the same OpenAI-compatible `/chat/completions` interface.

Swapping back to a Granite endpoint in production requires only two changes in [`src/config.py`](src/config.py):

```python
# Point at watsonx.ai (or any other OpenAI-compatible Granite endpoint)
GROQ_BASE_URL: str = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat"
GROQ_MODEL: str    = "ibm/granite-3-1-8b-instruct"
```

No other code changes are needed — the `openai.OpenAI(base_url=..., api_key=...)` client pattern in [`src/test_connection.py`](src/test_connection.py) and the triage agent are endpoint-agnostic by design.
