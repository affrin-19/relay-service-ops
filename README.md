<div align="center">

# ⚡ Relay
### Autonomous Enterprise Service Operations Platform

**Next-Generation Unified Service Management (USM/ESM) with LangGraph Multi-Agent Workflows, RAG, and ITIL SLA Automation.**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20StateGraph-orange.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple.svg)](https://www.trychroma.com/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Standard%20Tool%20Server-teal.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ITIL 4 Aligned](https://img.shields.io/badge/Framework-ITIL%204%20Aligned-success.svg)](https://www.peoplecert.org/itil-4)

[Explore Features](#-key-capabilities) • [System Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [REST API](#-restful-api-endpoints) • [MCP Server](#-model-context-protocol-mcp-integration)

</div>

---

## 📌 Executive Summary

Modern enterprise service desks suffer from severe ticket backlogs, slow mean time to resolution (MTTR), and fragmented workflows across IT, HR, and facilities. **Relay** is an autonomous enterprise service operations platform designed to solve service desk burnout. 

By pairing an **autonomous employee self-service agent** (achieving a **64.8% deflection rate** on routine inquiries using vector-grounded RAG) with an **AI Operations Copilot** (delivering 1-click incident briefings, historical fix correlation, and proactive SLA breach prediction), Relay drastically cuts MTTR while preserving strict ITIL 4 auditability and human-in-the-loop safety.

---

## 🏗️ System Architecture

```
               ┌────────────────────────────────────────────────────────┐
               │         STREAMLIT OPS CONSOLE (Frontend UI)            │
               │  • Editorial Design Tokens  • Single-render Caching    │
               │  • Asymmetric 8/4 Workspace • Interactive Telemetry    │
               └───────────────────────────┬────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │      FASTAPI REST GATEWAY   │               │     LANGGRAPH AI AGENT      │
     │  • Pydantic v2 Schemas      │               │  • Multi-node StateGraph    │
     │  • CORS & Latency Middleware│               │  • Intent Classification    │
     │  • Audit & Telemetry APIs   │               │  • Human-in-the-Loop Gate   │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
      ┌─────────────┴─────────────┐                 ┌─────────────┴─────────────┐
      ▼                           ▼                 ▼                           ▼
┌──────────────┐          ┌──────────────┐   ┌──────────────┐          ┌──────────────┐
│   SQLITE +   │          │  AI COPILOT  │   │  CHROMADB    │          │  MCP SERVER  │
│  SQLALCHEMY  │          │    ENGINE    │   │ VECTOR STORE │          │  (JSON-RPC)  │
│ • SLA Matrix │          │ • Summarizer │   │ • PDF RAG    │          │ • Standard   │
│ • Audit Logs │          │ • Correlation│   │ • Domain-hash│          │   Agent Tool │
│ • Dual Notes │          │ • Sentiment  │   │   Embeddings │          │   Handshake  │
└──────────────┘          └──────────────┘   └──────────────┘          └──────────────┘
```

---

## ⚡ Key Capabilities

### 1. 🧠 AI Operations Copilot (Agent Assist)
* **1-Click Executive Incident Summary**: Automatically synthesizes complex disruption threads and work notes into a 3-point briefing: *Blast Radius*, *Diagnostic Status*, and *Recommended Action*.
* **Historical Incident Correlation**: Scans database records to find similar historical outages (up to 98% similarity) and surfaces the **exact verified fix** applied in past incidents with 1-click copy into the dispatch box.
* **Sentiment & Urgency Analyzer**: Evaluates employee frustration keywords (`blocked`, `flickering`, `outage`) to determine ITIL priority and provide governance rationale.
* **Generative Closure Drafter**: Standardizes customer resolution emails and technical engineering post-mortems.

### 2. 🛡️ Proactive SLA Predictive Radar
* Scans open incidents for tickets whose SLA deadline expires within **under 2 hours**.
* Moves service desks from *reactive penalty panic* to *proactive intervention*, displaying emergency countdown cards directly on the triage console.

### 3. 🤖 LangGraph Stateful Virtual Operator
* **Intent Classification**: Evaluates user prompts into `information_retrieval`, `incident_creation`, or `general_chat`.
* **Grounded SOP Retrieval (RAG)**: Queries company PDF manuals in ChromaDB and returns step-by-step instructions with page citations.
* **Human-in-the-Loop (HITL) Safety Gate**: Enforces a mandatory confirmation checkpoint before writing incident tickets to production SQLite tables.
* **Deflection Feedback Loop**: Employees confirm self-service resolution (`✓ Solved my issue`), crediting **+22 minutes** of saved triage time to the live ESM dashboard.

### 4. 🌐 Unified Enterprise Service Management (ESM)
* Expands traditional ITSM into an enterprise-wide service mesh spanning 4 domains:
  1. **IT Hardware & Systems** (Workstations, 4K USB-C Hubs, YubiKeys)
  2. **Developer Tooling & Cloud IAM** (JetBrains, Docker, AWS Sandbox Roles, VPN)
  3. **Workplace & Facilities** (HID Smart Badges, Ergonomic Sit-to-Stand Desks, Meeting Room A/V)
  4. **People Operations (HR)** (New Hire Day-One Equipment Packs, WFH Stipends, Global Travel eSIMs)

### 5. 🔌 Standardized Model Context Protocol (MCP) Server
* Exposes 5 core service desk primitives over JSON-RPC stdio:
  * `create_ticket`, `get_ticket`, `search_tickets`, `update_ticket_status`, `search_knowledge_base`
* Enables third-party LLM clients (Claude Desktop, external IDE agents) to seamlessly discover and execute ticket actions.

### 6. 🔒 Air-Gapped & Offline Resilience
* **Dual-Mode Embedding Engine**: Automatically uses OpenAI `text-embedding-3-small` when an API key is present, and gracefully switches to a **deterministic CRC32 domain-boosted token-hashing algorithm** when offline or rate-limited.
* Vector similarity retrieval, ticket triage, and copilot operations continue running smoothly with **zero API costs and zero downtime**.

---

## 📂 Repository Directory Structure

```text
AI_USM_GenAI_Project/
│
├── app.py                      # Streamlit Operations Console (Editorial Dark Mode)
├── ai_copilot.py               # AI Copilot Engine (Summaries, Correlation, Sentiment, KPIs)
├── config.py                   # Centralized configuration & environment loader
├── database.py                 # SQLAlchemy database engine & automatic migrations
├── models.py                   # ORM models (Ticket, TicketComment, TicketAuditLog)
├── crud.py                     # CRUD operations, ITIL SLA matrix & audit trail
├── notifications.py            # Webhook alert dispatcher (Slack/Teams/Console)
├── requirements.txt            # Production-pinned dependency manifest
├── run_services.bat            # Windows 1-click dual-launcher (FastAPI + Streamlit)
├── run_services.ps1            # Windows PowerShell launcher
├── run_services.sh             # Linux/macOS launcher
├── .env.example                # Environment variables template
├── usm.db                      # Local SQLite database
│
├── api/                        # FastAPI RESTful Backend Gateway
│   ├── main.py                 # ASGI app, CORS, latency middleware, RFC 7807 handler
│   ├── routes.py               # REST route handlers (CRUD, Copilot, Knowledge, Analytics)
│   ├── schemas.py              # Pydantic v2 data validation & serialization schemas
│   └── analytics.py            # Real-time operational telemetry & KPI computation
│
├── agents/                     # LangGraph Multi-Agent Workflows
│   ├── state.py                # AgentState TypedDict schema
│   ├── nodes.py                # Classification, RAG retrieval, ticket creation, chat
│   └── graph.py                # StateGraph routing, conditional edges & compiled runnable
│
├── mcp_server/                 # Model Context Protocol (MCP) Server
│   └── server.py               # JSON-RPC tool server exposing 5 service desk primitives
│
├── rag/                        # Retrieval-Augmented Generation (RAG) Engine
│   ├── pdf_loader.py           # PDF document validation & text extraction
│   ├── text_splitter.py        # Sliding-window semantic chunking (700 chars / 120 overlap)
│   ├── vector_store.py         # ChromaDB client & dual-mode embedding function
│   └── rag_chain.py            # Context assembly & cited answer generator
│
├── data/                       # Persistent Data Storage
│   ├── chroma_db/              # ChromaDB vector store collections
│   └── knowledge_base/         # Indexed standard operating procedure (SOP) PDF manuals
│
└── tests/                      # Automated Verification & Test Suites
    ├── test_integration_review.py        # 10-point full-stack integration review
    ├── test_stage_5_api_and_analytics.py # FastAPI REST endpoints & analytics suite
    ├── test_rag_pipeline.py              # Vector store & similarity search suite
    └── test_stage_4_agent_and_mcp.py     # LangGraph agent & MCP tool server suite
```

---

## 🚀 Quick Start

### Prerequisites
* Python `3.11` or higher
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/relay-enterprise-service-ops.git
cd relay-enterprise-service-ops
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment:
# On Windows:
.\venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy example configuration
cp .env.example .env
# On Windows:
copy .env.example .env
```
*(Optional: Open `.env` and add your `OPENAI_API_KEY`. If left blank, Relay automatically operates using its local offline fallback embedding and triage engine).*

### 5. Launch the Platform
You can launch both the **FastAPI REST Gateway** and the **Streamlit Operations Console** simultaneously using the cross-platform launcher scripts:

* **Windows (Batch):** Double-click `run_services.bat`
* **Windows (PowerShell):** `.\run_services.ps1`
* **Linux / macOS (Bash):** `chmod +x run_services.sh && ./run_services.sh`

---

## 🌐 Live Service Endpoints

| Service | Port | Endpoint URL | Description |
| :--- | :--- | :--- | :--- |
| **Streamlit Operations Console** | `8501` | `http://localhost:8501` | Primary operations UI, dashboards & Copilot |
| **FastAPI REST Gateway** | `8000` | `http://localhost:8000` | High-performance asynchronous backend API |
| **Interactive API Documentation** | `8000` | `http://localhost:8000/docs` | Swagger / OpenAPI interactive explorer |
| **Alternative API Specs** | `8000` | `http://localhost:8000/redoc` | ReDoc API specifications |

---

## 📡 RESTful API Endpoints

| Method | Endpoint | Description | Auth / Validation |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Live system health (SQLite, ChromaDB, LangGraph) | Public |
| `GET` | `/api/v1/analytics/metrics` | Operational KPIs, category & department metrics | Public |
| `GET` | `/api/v1/telemetry/esm-kpis` | Real-time ESM ROI: Deflection Rate, Hours Saved, CSAT | Public |
| `GET` | `/api/v1/tickets` | Search, filter, and paginate ticket records | Query Params |
| `POST` | `/api/v1/tickets` | Create a new ticket with automatic ITIL SLA calculation | Pydantic Validated |
| `GET` | `/api/v1/tickets/{id}` | Fetch individual ticket details | 404 Handled |
| `PATCH` | `/api/v1/tickets/{id}` | Update status, priority, or technician assignment | Role-Based Check |
| `POST` | `/api/v1/tickets/{id}/comments` | Post a customer reply or internal technician work note | Validated |
| `GET` | `/api/v1/tickets/{id}/audit-logs` | Retrieve chronological compliance audit trail | Public |
| `GET` | `/api/v1/tickets/{id}/copilot-brief` | AI Copilot executive brief, similar cases, & draft fix | Public |
| `POST` | `/api/v1/agent/chat` | Proxy interaction with the LangGraph Virtual Operator | Session-tracked |
| `POST` | `/api/v1/knowledge/query` | Dense similarity search over ChromaDB knowledge base | Vector Top-K |

---

## 🔌 Model Context Protocol (MCP) Integration

Relay includes a native **Model Context Protocol (MCP)** server implemented in `mcp_server/server.py`.

### Connecting Claude Desktop to Relay
Add the following snippet to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "relay-service-desk": {
      "command": "python",
      "args": [
        "-m",
        "mcp_server.server"
      ],
      "env": {
        "PYTHONPATH": "c:/path/to/relay-enterprise-service-ops"
      }
    }
  }
}
```

Once connected, Claude can autonomously inspect tickets, create incidents, check corporate SOPs, and resolve service requests directly inside its conversation interface.

---

## 🧪 Testing & Verification

Relay maintains a strict test suite covering every layer of the architecture:

```bash
# Run comprehensive 10-point integration review
python test_integration_review.py

# Run FastAPI REST endpoint & analytics suite
python test_stage_5_api_and_analytics.py

# Run RAG vector store & chunking verification
python test_rag_pipeline.py

# Run LangGraph multi-agent workflow test suite
python test_stage_4_agent_and_mcp.py
```

---

## 📊 ITIL 4 SLA Target Matrix

| Severity | Target Response Deadline | Target Resolution Deadline | Escalation Policy |
| :--- | :--- | :--- | :--- |
| **P1 Critical** | **15 Minutes** | **4 Hours** | Immediate on-call pager & incident bridge |
| **P2 High** | **1 Hour** | **8 Hours** | Priority engineering team queue dispatch |
| **P3 Medium** | **4 Hours** | **24 Hours** | Standard operational backlog allocation |
| **P4 Low** | **8 Hours** | **72 Hours** | Best-effort routine fulfillment |

---

## 📄 License

This project is open-source and distributed under the terms of the **[MIT License](LICENSE)**.

---

<div align="center">
  <sub>Built with precision for enterprise service excellence. Engineered by the Relay Operations Team.</sub>
</div>
