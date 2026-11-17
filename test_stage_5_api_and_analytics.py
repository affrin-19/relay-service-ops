"""
Comprehensive Test Suite for Stage 5: FastAPI REST API, Telemetry, and Production Integration.

Executes end-to-end endpoint verification using FastAPI TestClient:
- Service Discovery & Health Checks
- Operational Analytics & KPIs
- Ticket CRUD Operations with Validation & Role Authorization
- LangGraph Multi-Agent Chat Workflow Proxy
- Vector Knowledge Base Retrieval Proxy
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402
from database import init_db, get_db  # noqa: E402
from crud import seed_initial_tickets  # noqa: E402

client = TestClient(app)


def setup_module():
    """Ensure database schema and initial test records exist."""
    init_db()
    db = next(get_db())
    try:
        seed_initial_tickets(db)
    finally:
        db.close()


def test_01_root_service_discovery():
    """Verify service discovery metadata and documentation URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AI-Powered Unified Service Management API"
    assert data["version"] == "1.5.0"
    assert "documentation" in data
    assert data["documentation"]["swagger_ui"] == "/docs"
    print("✅ Test 1 Passed: Root service discovery endpoint functional.")


def test_02_health_check():
    """Verify system health across SQLite, ChromaDB, and LangGraph."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "Healthy" in data["database"]
    assert "LangGraph" in data["agent_engine"]
    print(f"✅ Test 2 Passed: Health check functional (DB: {data['database']}, KB: {data['knowledge_base']}).")


def test_03_analytics_telemetry():
    """Verify aggregated operational KPIs and distribution breakdowns."""
    response = client.get("/api/v1/analytics/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_tickets" in data
    assert data["total_tickets"] >= 1
    assert "resolution_rate_pct" in data
    assert "severity_breakdown" in data
    assert "category_breakdown" in data
    assert "department_breakdown" in data
    print(f"✅ Test 3 Passed: Analytics engine functional (Total: {data['total_tickets']}, Resolution Rate: {data['resolution_rate_pct']}%).")


def test_04_create_ticket_valid():
    """Verify ticket creation with automatic sequential ID generation."""
    payload = {
        "employee_name": "Marcus Aurelius",
        "employee_email": "marcus.a@company.com",
        "department": "Engineering",
        "category": "Hardware",
        "title": "Dual 4K Monitor Flickering via USB-C Hub",
        "description": "External displays disconnect intermittently every 15 minutes during code compilation.",
        "priority": "High"
    }
    response = client.post("/api/v1/tickets", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["ticket_id"].startswith("USM")
    assert data["title"] == payload["title"]
    assert data["status"] == "Open"
    assert data["priority"] == "High"
    print(f"✅ Test 4 Passed: Ticket creation endpoint functional (Created: {data['ticket_id']}).")
    return data["ticket_id"]


def test_05_create_ticket_validation_error():
    """Verify Pydantic input validation rejects malformed requests."""
    # Invalid email and title too short
    invalid_payload = {
        "employee_name": "M",
        "employee_email": "not-an-email",
        "department": "IT",
        "category": "Hardware",
        "title": "Hi",
        "description": "Short",
        "priority": "High"
    }
    response = client.post("/api/v1/tickets", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
    print("✅ Test 5 Passed: Pydantic input validation correctly rejected malformed ticket payload.")


def test_06_list_tickets_with_filters():
    """Verify ticket querying, filtering, and pagination."""
    response = client.get("/api/v1/tickets?limit=10&status=Open")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "tickets" in data
    assert isinstance(data["tickets"], list)
    print(f"✅ Test 6 Passed: Ticket listing and filtering functional (Found {len(data['tickets'])} matching tickets).")


def test_07_get_ticket_by_id():
    """Verify ticket retrieval by unique business ID."""
    # Retrieve an existing ticket
    list_res = client.get("/api/v1/tickets?limit=1")
    target_id = list_res.json()["tickets"][0]["ticket_id"]

    response = client.get(f"/api/v1/tickets/{target_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == target_id
    print(f"✅ Test 7 Passed: Ticket retrieval by ID functional ({target_id}).")


def test_08_get_ticket_not_found():
    """Verify 404 response for non-existent ticket."""
    response = client.get("/api/v1/tickets/USM999999")
    assert response.status_code == 404
    print("✅ Test 8 Passed: 404 Not Found returned for non-existent ticket.")


def test_09_update_ticket_authorization():
    """Verify role-based access control on ticket modifications."""
    list_res = client.get("/api/v1/tickets?limit=1")
    target_id = list_res.json()["tickets"][0]["ticket_id"]

    # Unauthorized role
    unauth_payload = {
        "status": "In Progress",
        "auth_role": "guest_contractor"
    }
    res_unauth = client.patch(f"/api/v1/tickets/{target_id}", json=unauth_payload)
    assert res_unauth.status_code == 403

    # Authorized role
    auth_payload = {
        "status": "In Progress",
        "assigned_technician": "Alex Chen (Network)",
        "auth_role": "admin"
    }
    res_auth = client.patch(f"/api/v1/tickets/{target_id}", json=auth_payload)
    assert res_auth.status_code == 200
    assert res_auth.json()["status"] == "In Progress"
    print(f"✅ Test 9 Passed: RBAC authorization enforced (403 for unauthorized, 200 for admin).")


def test_10_langgraph_agent_chat_endpoint():
    """Verify LangGraph multi-agent triage and conversational proxy endpoint."""
    payload = {
        "query": "Hello, can you introduce yourself and tell me what you do?",
        "employee_name": "Helena Troy",
        "employee_email": "helena.t@company.com",
        "department": "Marketing"
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ai_response" in data
    assert len(data["ai_response"]) > 0
    assert data["intent"] in ["greeting", "general_chat", "consultation", "information_retrieval"]
    print(f"✅ Test 10 Passed: LangGraph agent chat endpoint functional (Intent: {data['intent']}).")


def test_11_knowledge_base_vector_query():
    """Verify ChromaDB semantic search via REST endpoint."""
    payload = {
        "query": "VPN connection and multi-factor authentication setup",
        "top_k": 2
    }
    response = client.post("/api/v1/knowledge/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "passages" in data
    assert isinstance(data["passages"], list)
    print(f"✅ Test 11 Passed: Vector knowledge query endpoint functional ({len(data['passages'])} passages returned).")


def test_12_knowledge_documents_list():
    """Verify listing indexed documents in vector store."""
    response = client.get("/api/v1/knowledge/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data
    print(f"✅ Test 12 Passed: Knowledge documents catalog endpoint functional ({data['total_documents']} documents indexed).")


def test_13_delete_ticket():
    """Verify ticket deletion endpoint."""
    # Create a temporary ticket to delete
    create_payload = {
        "employee_name": "Temp User",
        "employee_email": "temp.user@company.com",
        "department": "QA",
        "category": "Software",
        "title": "Ephemeral Test Ticket to Delete",
        "description": "This ticket will be deleted immediately to verify DELETE endpoint.",
        "priority": "Low"
    }
    res_create = client.post("/api/v1/tickets", json=create_payload)
    temp_id = res_create.json()["ticket_id"]

    # Delete it
    res_delete = client.delete(f"/api/v1/tickets/{temp_id}")
    assert res_delete.status_code == 200
    assert res_delete.json()["status"] == "success"

    # Verify it is gone
    res_get = client.get(f"/api/v1/tickets/{temp_id}")
    assert res_get.status_code == 404
    print(f"✅ Test 13 Passed: Ticket deletion functional ({temp_id} deleted and verified 404).")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 RUNNING STAGE 5 FASTAPI & ANALYTICS TEST SUITE")
    print("=" * 70)
    setup_module()
    test_01_root_service_discovery()
    test_02_health_check()
    test_03_analytics_telemetry()
    test_04_create_ticket_valid()
    test_05_create_ticket_validation_error()
    test_06_list_tickets_with_filters()
    test_07_get_ticket_by_id()
    test_08_get_ticket_not_found()
    test_09_update_ticket_authorization()
    test_10_langgraph_agent_chat_endpoint()
    test_11_knowledge_base_vector_query()
    test_12_knowledge_documents_list()
    test_13_delete_ticket()
    print("=" * 70)
    print("🎉 ALL 13/13 STAGE 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
