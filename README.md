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

   Copy `.env.example` to `.env` and fill in your IBM watsonx.ai credentials and any other required keys.

## Dependencies

| Package | Purpose |
|---|---|
| `langflow` | Visual LLM workflow orchestration |
| `ibm-watsonx-ai` | IBM watsonx.ai foundation model SDK |
| `pandas` | Data ingestion and ticket preprocessing |
| `python-dotenv` | Environment variable management |
