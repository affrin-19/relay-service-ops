"""
LangGraph Workflow Architecture for IT Support Agent.

Constructs the StateGraph with nodes, conditional routing edges,
and compiled execution pipeline.
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.nodes import (
    classify_and_triage_node,
    retrieve_knowledge_node,
    generate_resolution_node,
    create_ticket_node,
    handle_general_chat_node
)


# ==============================================================================
# CONDITIONAL ROUTING LOGIC
# ==============================================================================
def route_after_triage(state: AgentState) -> str:
    """
    Decides the next execution step based on classified user intent.
    
    Returns:
        'create_ticket' if user wants to log an incident,
        'general_chat' if user gave a greeting or non-technical prompt,
        'retrieve_knowledge' if user is seeking factual/policy troubleshooting assistance.
    """
    intent = state.get("intent", "information_retrieval")
    if intent == "incident_creation":
        return "create_ticket"
    elif intent == "general_chat":
        return "general_chat"
    else:
        return "retrieve_knowledge"


# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================
def build_it_support_graph():
    """
    Assembles and compiles the LangGraph StateGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("classify_and_triage", classify_and_triage_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_resolution", generate_resolution_node)
    workflow.add_node("create_ticket", create_ticket_node)
    workflow.add_node("general_chat", handle_general_chat_node)

    # 2. Register Entry Edge
    workflow.add_edge(START, "classify_and_triage")

    # 3. Register Conditional Decision Edge
    workflow.add_conditional_edges(
        "classify_and_triage",
        route_after_triage,
        {
            "create_ticket": "create_ticket",
            "retrieve_knowledge": "retrieve_knowledge",
            "general_chat": "general_chat"
        }
    )

    # 4. Sequential Edges
    workflow.add_edge("retrieve_knowledge", "generate_resolution")

    # 5. Terminal Edges to END
    workflow.add_edge("generate_resolution", END)
    workflow.add_edge("create_ticket", END)
    workflow.add_edge("general_chat", END)

    # Compile into a runnable application
    return workflow.compile()


# Global compiled agent instance
it_support_agent = build_it_support_graph()


def run_it_support_agent(
    user_query: str,
    employee_name: str = "Employee",
    employee_email: str = "employee@company.com",
    department: str = "Engineering",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    requires_approval: bool = False,
    is_approved: bool = False
) -> AgentState:
    """
    High-level entry point to execute the LangGraph agent for a user query.
    
    Args:
        user_query: Employee's prompt in natural language.
        employee_name: Requester's name.
        employee_email: Requester's email.
        department: Requester's organizational unit.
        conversation_history: Prior chat turns.
        requires_approval: Whether human confirmation is required before creating a ticket.
        is_approved: Explicit user confirmation to log the drafted ticket.
        
    Returns:
        The final AgentState dictionary containing response, classifications, and citations.
    """
    initial_state: AgentState = {
        "user_query": user_query.strip(),
        "employee_name": employee_name.strip(),
        "employee_email": employee_email.strip(),
        "department": department.strip(),
        "conversation_history": conversation_history or [],
        "requires_approval": requires_approval,
        "is_approved": is_approved,
        "retrieved_chunks": [],
        "sources": [],
        "action_taken": "unknown"
    }

    try:
        final_state = it_support_agent.invoke(initial_state)
        return final_state
    except Exception as e:
        return {
            **initial_state,
            "ai_response": f"❌ An error occurred while executing the AI agent workflow: {str(e)}",
            "error": str(e),
            "action_taken": "error"
        }

