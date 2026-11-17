"""
Agent State Schema for LangGraph IT Service Workflows.

Defines the shared state dictionary passed across all nodes in the graph.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Central state object for the LangGraph IT Support Agent.
    
    Fields:
        user_query: The natural language prompt or problem statement from the employee.
        employee_name: Requester's full name.
        employee_email: Requester's corporate email.
        department: Requester's organizational department.
        conversation_history: List of prior messages for multi-turn dialogue.
        intent: Detected intent ('information_retrieval', 'incident_creation', 'general_chat').
        category: Classified IT category ('Hardware', 'Software', 'Network', 'Email', 'Access').
        priority: Recommended priority ('Low', 'Medium', 'High', 'Critical').
        title: Short synthesized summary or subject for the ticket.
        retrieved_chunks: Documentation passages retrieved from ChromaDB.
        sources: Unique list of source documents and pages.
        ai_response: Generated troubleshooting answer or confirmation text.
        ticket_created: Dictionary of newly logged ticket if an incident was created.
        action_taken: Audit tag of primary action taken ('retrieved_knowledge', 'ticket_created', 'general_chat').
        error: Error description if a node encountered an exception.
    """
    user_query: str
    employee_name: str
    employee_email: str
    department: str
    conversation_history: List[Dict[str, str]]
    intent: str
    category: str
    priority: str
    title: str
    retrieved_chunks: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    ai_response: str
    requires_approval: bool
    is_approved: bool
    draft_ticket: Optional[Dict[str, Any]]
    ticket_created: Optional[Dict[str, Any]]
    action_taken: str
    error: Optional[str]

