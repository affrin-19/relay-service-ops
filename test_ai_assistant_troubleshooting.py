"""
Comprehensive Automated Test Suite: AI Assistant Troubleshooting & RAG Verification.

Verifies resolution of the AI Assistant generic response issue across 5 scenarios:
1. VPN Setup Steps (RAG retrieval & gateway verification)
2. Password Policy (RAG retrieval & lockout policy verification)
3. Outlook Authentication Troubleshooting (Intent, Category, Priority, and non-generic instructions)
4. Wi-Fi Incident Ticket Creation (Intent, Critical priority, and SQLite ticket generation)
5. General IT Question (Conversational introduction & capability explanation)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import run_it_support_agent
from database import init_db


def test_01_vpn_setup_rag():
    """Verify that VPN setup queries invoke RAG and retrieve specific technical instructions."""
    query = "How do I configure AcmeConnect VPN and what gateway address should I enter?"
    result = run_it_support_agent(query)

    assert result["intent"] == "information_retrieval", f"Expected information_retrieval, got {result['intent']}"
    assert result["category"] == "Network", f"Expected Network category, got {result['category']}"
    assert result["action_taken"] == "retrieved_knowledge", f"Expected retrieved_knowledge action, got {result['action_taken']}"
    assert len(result.get("sources", [])) > 0, "Expected at least 1 verified document citation"

    response = result.get("ai_response", "")
    assert "acme" in response.lower() or "vpn" in response.lower(), "Response must contain VPN guidance"
    assert "gw-west" in response or "gateway" in response.lower(), "Response must include gateway information"
    print("✅ Test 1 Passed: VPN Setup query correctly invoked RAG pipeline and returned gateway instructions.")


def test_02_password_policy_rag():
    """Verify that password policy queries retrieve corporate complexity and lockout rules."""
    query = "What is the corporate password length requirement and lockout policy?"
    result = run_it_support_agent(query)

    assert result["intent"] == "information_retrieval", f"Expected information_retrieval, got {result['intent']}"
    assert result["category"] == "Access", f"Expected Access category, got {result['category']}"
    assert result["action_taken"] == "retrieved_knowledge", f"Expected retrieved_knowledge action, got {result['action_taken']}"
    assert any(s.get("page") == 2 for s in result.get("sources", [])), "Expected citation from SOP Page 2"

    response = result.get("ai_response", "")
    assert "14" in response or "character" in response.lower(), "Response must contain length requirement"
    assert "lockout" in response.lower() or "failed" in response.lower(), "Response must mention lockout policy"
    print("✅ Test 2 Passed: Password Policy query correctly retrieved Page 2 SOP documentation.")


def test_03_outlook_authentication_troubleshooting():
    """
    CRITICAL REGRESSION TEST:
    Verify that Outlook authentication questions receive actionable IT troubleshooting
    and are NEVER misclassified as 'general_chat' or return the generic welcome message.
    """
    query = "My Outlook is showing an authentication error. I cannot send or receive emails. Help me troubleshoot."
    result = run_it_support_agent(query)

    # 1. Intent Verification
    assert result["intent"] == "information_retrieval", (
        f"CRITICAL FAILURE: Outlook query misclassified as '{result['intent']}'. Must be 'information_retrieval'."
    )

    # 2. Category & Priority Verification
    assert result["category"] == "Email", f"Expected Email category, got {result['category']}"
    assert result["priority"] == "High", f"Expected High priority for unable to send/receive emails, got {result['priority']}"
    assert result["action_taken"] == "retrieved_knowledge", f"Expected retrieved_knowledge, got {result['action_taken']}"

    # 3. Content Verification: Must NOT be the generic welcome message
    response = result.get("ai_response", "")
    generic_welcome_snippet = "Hello! I am your AI-Powered Unified Service Management"
    assert generic_welcome_snippet not in response, (
        "CRITICAL FAILURE: Response returned the generic welcome message instead of troubleshooting steps!"
    )

    # 4. Must contain actual technical troubleshooting guidance
    assert any(k in response.lower() for k in ["sso", "credentials", "webmail", "outlook.office.com", "repair", "exchange"]), (
        "Response must contain actionable email troubleshooting steps."
    )
    print("✅ Test 3 Passed: Outlook Authentication Troubleshooting returned technical instructions (generic welcome bug resolved).")


def test_03b_outlook_not_opening_troubleshooting():
    """
    CRITICAL REGRESSION TEST:
    Verify that 'outlook is not opening' retrieves actual SOP document chunks from ChromaDB,
    identifies Category=Email, and NEVER returns 'The Knowledge Base is currently empty'.
    """
    query = "outlook is not opening"
    result = run_it_support_agent(query)

    assert result["intent"] == "information_retrieval", f"Expected information_retrieval, got {result['intent']}"
    assert result["category"] == "Email", f"Expected Email category, got {result['category']}"
    assert result["action_taken"] == "retrieved_knowledge", f"Expected retrieved_knowledge, got {result['action_taken']}"
    assert len(result.get("retrieved_chunks", [])) > 0, "Expected at least 1 retrieved chunk from ChromaDB"
    assert len(result.get("sources", [])) > 0, "Expected at least 1 verified source citation"

    response = result.get("ai_response", "")
    empty_kb_warning = "The Knowledge Base is currently empty or no relevant documents were found"
    assert empty_kb_warning not in response, (
        "CRITICAL FAILURE: Response falsely claimed the knowledge base was empty when 10 chunks exist!"
    )
    assert any(k in response.lower() for k in ["outlook", "safe", "webmail", "exchange", "credential", "sso"]), (
        "Response must contain verified Outlook troubleshooting guidance."
    )
    print("✅ Test 3b Passed: 'outlook is not opening' retrieved actual SOP chunks and returned troubleshooting guidance.")


def test_04_wifi_incident_creation():
    """Verify that incident reporting queries invoke the ticket creation workflow with Critical priority."""
    query = "Please create a ticket: The Wi-Fi gateway in Conference Room B is completely down and dropping all packets."
    result = run_it_support_agent(
        user_query=query,
        employee_name="Jordan Bell",
        employee_email="jordan.b@company.com",
        department="Operations",
        requires_approval=False
    )

    assert result["intent"] == "incident_creation", f"Expected incident_creation, got {result['intent']}"
    assert result["category"] == "Network", f"Expected Network category, got {result['category']}"
    assert result["priority"] == "Critical", f"Expected Critical priority for total Wi-Fi outage, got {result['priority']}"
    assert result["action_taken"] == "ticket_created", f"Expected ticket_created, got {result['action_taken']}"
    assert result.get("ticket_created") is not None, "Ticket record was not returned in agent state"
    assert result["ticket_created"]["Ticket ID"].startswith("USM"), "Ticket ID must follow sequential USM format"
    print(f"✅ Test 4 Passed: Wi-Fi outage query correctly created incident ticket ({result['ticket_created']['Ticket ID']}).")


def test_05_general_it_question():
    """Verify that pure greetings and questions about the assistant's capabilities receive friendly, contextual introductions."""
    query = "Hi, who are you and what can you do?"
    result = run_it_support_agent(
        user_query=query,
        employee_name="Taylor Swift"
    )

    assert result["intent"] == "general_chat", f"Expected general_chat, got {result['intent']}"
    assert result["action_taken"] == "general_chat", f"Expected general_chat, got {result['action_taken']}"

    response = result.get("ai_response", "")
    assert "Taylor Swift" in response or "Hello" in response or "Copilot" in response or "Assistant" in response
    assert "VPN" in response or "ticket" in response.lower(), "Should explain technical capabilities"
    print("✅ Test 5 Passed: General inquiry correctly routed to general_chat with contextual capabilities explanation.")


if __name__ == "__main__":
    print("=" * 75)
    print("🧪 RUNNING AI ASSISTANT TROUBLESHOOTING & RAG VERIFICATION SUITE")
    print("=" * 75)
    init_db()
    test_01_vpn_setup_rag()
    test_02_password_policy_rag()
    test_03_outlook_authentication_troubleshooting()
    test_03b_outlook_not_opening_troubleshooting()
    test_04_wifi_incident_creation()
    test_05_general_it_question()
    print("=" * 75)
    print("🎉 ALL 6/6 AI ASSISTANT VERIFICATION SCENARIOS PASSED WITH 100% ACCURACY!")
    print("=" * 75)

