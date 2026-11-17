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


def route_after_triage(state: AgentState) -> str:
    intent = state.get("intent", "information_retrieval")
    if intent == "incident_creation":
        return "create_ticket"
    elif intent == "general_chat":
        return "general_chat"
    return "retrieve_knowledge"


def build_it_support_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_and_triage", classify_and_triage_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("generate_resolution", generate_resolution_node)
    workflow.add_node("create_ticket", create_ticket_node)
    workflow.add_node("general_chat", handle_general_chat_node)

    workflow.add_edge(START, "classify_and_triage")

    workflow.add_conditional_edges(
        "classify_and_triage",
        route_after_triage,
        {
            "create_ticket": "create_ticket",
            "retrieve_knowledge": "retrieve_knowledge",
            "general_chat": "general_chat"
        }
    )

    workflow.add_edge("retrieve_knowledge", "generate_resolution")
    workflow.add_edge("generate_resolution", END)
    workflow.add_edge("create_ticket", END)
    workflow.add_edge("general_chat", END)

    return workflow.compile()


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
        return it_support_agent.invoke(initial_state)
    except Exception as e:
        return {
            **initial_state,
            "ai_response": f"An error occurred while executing the AI agent workflow: {str(e)}",
            "error": str(e),
            "action_taken": "error"
        }
