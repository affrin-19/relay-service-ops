# Relay

Relay is a service management platform for IT and operations teams. It provides ticket tracking, automated triage with LangGraph, knowledge base search using vector RAG, and an AI copilot for incident response.

## What It Does

- **Incident and Request Tracking**: Core ticketing system with priority-based SLA calculation (Critical, High, Medium, Low) and audit logging.
- **Triage Console**: Operational dashboard displaying in-flight tickets, SLA compliance, priority breakdown, and team workload.
- **Virtual Operator**: Chat assistant that answers technical questions from company SOPs and drafts incident tickets with a human confirmation step.
- **AI Copilot**: Summarizes ticket threads, matches incoming issues against past resolved tickets, and drafts customer closure notes.
- **Knowledge Base RAG**: Indexes PDF documentation into ChromaDB for similarity search with source citations.
- **MCP Server**: Exposes ticketing tools over Model Context Protocol stdio for use with external AI assistants.
- **REST API**: FastAPI backend providing endpoints for tickets, knowledge search, agent chat, and telemetry.

## Running Locally

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/affrin-19/relay-service-ops.git
   cd relay-service-ops
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env  # On Windows: copy .env.example .env
   ```
   Add your OpenAI API key in `.env` if you want live LLM generation. The system works without an API key using local fallbacks.

4. Start the backend API:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```

5. In another terminal, start the web interface:
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501 in your browser.

## Notes

- **Offline and Fallback Mode**: If no OpenAI API key is configured or if the API hits rate limits, the system falls back to a local CRC32 token-hashing embedding function and rule-based triage. Core ticket operations, search, and the UI continue working without external network calls.
- **Database**: SQLite is used by default (`usm.db`). The path is resolved relative to the project root so it remains consistent regardless of the working directory.
- **Testing**: Run tests using `python test_stage_5_api_and_analytics.py`, `python test_rag_pipeline.py`, or `python test_integration_review.py`.
