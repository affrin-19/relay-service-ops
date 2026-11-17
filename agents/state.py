from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
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
