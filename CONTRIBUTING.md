# Contributing to Relay

Thank you for your interest in contributing to **Relay: Autonomous Enterprise Service Operations Platform**! We welcome bug reports, feature enhancements, documentation improvements, and architectural feedback.

---

## 🛠️ Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/relay-enterprise-service-ops.git
   cd relay-enterprise-service-ops
   ```

2. **Create a Python 3.11+ virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Initialize local environment variables:**
   ```bash
   cp .env.example .env
   # On Windows:
   copy .env.example .env
   ```

5. **Start services:**
   * **FastAPI Backend:**
     ```bash
     uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
     ```
   * **Streamlit Frontend:**
     ```bash
     streamlit run app.py --server.port 8501
     ```

---

## 🧪 Running Tests

Ensure all test suites pass before submitting a pull request:

```bash
# Run integration review tests
python test_integration_review.py

# Run REST API & analytics tests
python test_stage_5_api_and_analytics.py

# Run RAG vector pipeline tests
python test_rag_pipeline.py

# Run LangGraph Agent & MCP server tests
python test_stage_4_agent_and_mcp.py
```

---

## 🌿 Git Branching & Commit Conventions

* **Branch Naming**:
  * `feature/<feature-name>` (e.g., `feature/slack-interactive-buttons`)
  * `fix/<bug-name>` (e.g., `fix/sla-breach-timezone`)
  * `docs/<topic>` (e.g., `docs/mcp-setup-guide`)

* **Conventional Commits**:
  * `feat: add predictive SLA radar in triage view`
  * `fix: prevent text truncation on metric cards`
  * `docs: update API routes in README`
  * `test: add unit tests for fallback vector hashing`

---

## 📋 Pull Request Checklist

- [ ] Code follows PEP 8 style conventions.
- [ ] New features include corresponding test cases.
- [ ] All tests pass locally (`test_integration_review.py`).
- [ ] No API keys, credentials, or `.env` files are committed.
- [ ] Documentation or README has been updated if API endpoints or configurations changed.
