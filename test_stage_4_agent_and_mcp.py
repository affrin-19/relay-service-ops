"""
Comprehensive Test Suite for Stage 4:
LangGraph Multi-Agent Workflows & Official MCP Server Integration.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import run_it_support_agent
from mcp_server.server import (
    create_ticket as mcp_create_ticket,
    get_ticket as mcp_get_ticket,
    search_tickets as mcp_search_tickets,
    update_ticket_status as mcp_update_ticket_status,
    search_knowledge_base as mcp_search_knowledge_base
)
from database import SessionLocal
from crud import get_all_tickets, get_dashboard_metrics
from rag.vector_store import USMVectorStore


def run_stage_4_tests():
    print("=" * 75)
    print("🚀 RUNNING STAGE 4 VERIFICATION: LANGGRAPH AGENT & MCP TOOLS")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # 1. LANGGRAPH AGENT: INFORMATION RETRIEVAL
    # --------------------------------------------------------------------------
    print("\n[PART 1.1] Testing LangGraph Agent: Information Retrieval Path...")
    res_rag = run_it_support_agent("How do I connect to the corporate AcmeConnect VPN?")
    print(f"  ✓ Intent Classified: {res_rag['intent']}")
    print(f"  ✓ Category Assigned: {res_rag['category']}")
    print(f"  ✓ Priority Assigned: {res_rag['priority']}")
    print(f"  ✓ Action Taken: {res_rag['action_taken']}")
    print(f"  ✓ Chunks Retrieved: {len(res_rag.get('retrieved_chunks', []))}")
    assert res_rag["intent"] == "information_retrieval"
    assert res_rag["action_taken"] == "retrieved_knowledge"
    assert len(res_rag.get("retrieved_chunks", [])) > 0
    print("  ✓ RAG Integration inside LangGraph verified.")

    # --------------------------------------------------------------------------
    # 2. LANGGRAPH AGENT: INCIDENT CREATION
    # --------------------------------------------------------------------------
    print("\n[PART 1.2] Testing LangGraph Agent: Incident Creation Path...")
    res_ticket = run_it_support_agent(
        user_query="Please create a ticket: The VPN gateway in West Coast is completely down for all engineers.",
        employee_name="Sarah Connor",
        employee_email="sarah.c@company.com",
        department="Engineering"
    )
    print(f"  ✓ Intent Classified: {res_ticket['intent']}")
    print(f"  ✓ Category Assigned: {res_ticket['category']}")
    print(f"  ✓ Priority Assigned: {res_ticket['priority']}")
    print(f"  ✓ Action Taken: {res_ticket['action_taken']}")
    created_ticket = res_ticket.get("ticket_created")
    assert created_ticket is not None, "Ticket was not created by agent"
    created_id = created_ticket.get("Ticket ID")
    print(f"  ✓ Ticket Automatically Logged in SQLite: {created_id}")
    assert created_id.startswith("USM")

    # --------------------------------------------------------------------------
    # 3. LANGGRAPH AGENT: GENERAL CONVERSATION
    # --------------------------------------------------------------------------
    print("\n[PART 1.3] Testing LangGraph Agent: General Chat Path...")
    res_chat = run_it_support_agent("Hi, who are you?")
    print(f"  ✓ Intent Classified: {res_chat['intent']}")
    print(f"  ✓ Action Taken: {res_chat['action_taken']}")
    assert res_chat["intent"] == "general_chat"
    assert res_chat["action_taken"] == "general_chat"

    # --------------------------------------------------------------------------
    # 4. MCP TOOL 1: create_ticket
    # --------------------------------------------------------------------------
    print("\n[PART 2.1] Testing MCP Tool: create_ticket...")
    mcp_t1 = json.loads(mcp_create_ticket(
        employee_name="Devin IT",
        employee_email="devin@company.com",
        department="Operations",
        category="Hardware",
        title="USB-C Monitor flickering",
        description="Monitor flickers every 30 seconds when connected via HDMI adapter.",
        priority="Medium"
    ))
    assert mcp_t1["status"] == "success"
    mcp_tid = mcp_t1["ticket"]["Ticket ID"]
    print(f"  ✓ MCP created ticket: {mcp_tid}")

    # --------------------------------------------------------------------------
    # 5. MCP TOOL 2: get_ticket
    # --------------------------------------------------------------------------
    print("\n[PART 2.2] Testing MCP Tool: get_ticket...")
    mcp_t2 = json.loads(mcp_get_ticket(mcp_tid))
    assert mcp_t2["status"] == "success"
    assert mcp_t2["ticket"]["Ticket ID"] == mcp_tid
    print(f"  ✓ MCP retrieved ticket: {mcp_tid} - {mcp_t2['ticket']['Title']}")

    # --------------------------------------------------------------------------
    # 6. MCP TOOL 3: search_tickets
    # --------------------------------------------------------------------------
    print("\n[PART 2.3] Testing MCP Tool: search_tickets...")
    mcp_t3 = json.loads(mcp_search_tickets(search_query="flickering", category="Hardware"))
    assert mcp_t3["status"] == "success"
    assert mcp_t3["total_matches"] >= 1
    print(f"  ✓ MCP search matched {mcp_t3['total_matches']} tickets.")

    # --------------------------------------------------------------------------
    # 7. MCP TOOL 4: update_ticket_status & Authorization
    # --------------------------------------------------------------------------
    print("\n[PART 2.4] Testing MCP Tool: update_ticket_status & Auth...")
    # Unauthorized
    unauth = json.loads(mcp_update_ticket_status(mcp_tid, "Resolved", auth_role="contractor"))
    assert unauth["status"] == "forbidden"
    print("  ✓ Security: Successfully blocked unauthorized role 'contractor'.")

    # Authorized
    auth = json.loads(mcp_update_ticket_status(mcp_tid, "In Progress", auth_role="admin", technician="Devon Miller (Cloud)"))
    assert auth["status"] == "success"
    assert auth["ticket"]["Status"] == "In Progress"
    print(f"  ✓ Authorized update confirmed: {mcp_tid} -> In Progress")

    # --------------------------------------------------------------------------
    # 8. MCP TOOL 5: search_knowledge_base
    # --------------------------------------------------------------------------
    print("\n[PART 2.5] Testing MCP Tool: search_knowledge_base...")
    mcp_t5 = json.loads(mcp_search_knowledge_base("password complexity requirement", top_k=2))
    assert mcp_t5["status"] == "success"
    assert mcp_t5["total_retrieved"] > 0
    print(f"  ✓ MCP vector search retrieved {mcp_t5['total_retrieved']} passages from ChromaDB.")

    # --------------------------------------------------------------------------
    # 9. REGRESSION CHECK: STAGES 1, 2, 3
    # --------------------------------------------------------------------------
    print("\n[REGRESSION CHECK] Validating Stages 1, 2, and 3 Integrity...")
    db = SessionLocal()
    total_tickets = len(get_all_tickets(db))
    metrics = get_dashboard_metrics(db)
    db.close()
    
    vs = USMVectorStore()
    kb_count = vs.count()

    print(f"  ✓ SQLite Total Tickets: {total_tickets}")
    print(f"  ✓ Live Metrics: {metrics}")
    print(f"  ✓ ChromaDB Vectors: {kb_count} chunks")
    assert total_tickets >= 6
    assert kb_count > 0

    print("\n" + "=" * 75)
    print("🎉 ALL STAGE 4 VERIFICATION TESTS PASSED WITHOUT REGRESSIONS!")
    print("=" * 75)


if __name__ == "__main__":
    run_stage_4_tests()

