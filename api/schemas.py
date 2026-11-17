from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    employee_name: str = Field(..., min_length=2, max_length=100)
    employee_email: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )
    department: str
    category: str
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10)
    priority: str = Field(default="Medium")
    ticket_type: Optional[str] = Field(default="Incident")


class TicketUpdateRequest(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_technician: Optional[str] = None
    resolution_notes: Optional[str] = None
    auth_role: str = Field(default="technician")


class TicketResponse(BaseModel):
    id: int
    ticket_id: str
    ticket_type: Optional[str] = "Incident"
    title: str
    employee_name: str
    employee_email: str
    department: str
    category: str
    priority: str
    status: str
    description: str
    assigned_technician: str
    response_due_date: Optional[str] = None
    sla_due_date: Optional[str] = None
    is_sla_breached: bool = False
    resolution_notes: Optional[str] = None
    created_at: str
    updated_at: str


class TicketListResponse(BaseModel):
    total: int
    tickets: List[TicketResponse]


class CommentCreateRequest(BaseModel):
    author_name: str = Field(..., min_length=2, max_length=100)
    author_role: str = Field(default="requester")
    message: str = Field(..., min_length=1, max_length=2000)
    is_internal_note: bool = Field(default=False)


class CommentResponse(BaseModel):
    id: int
    ticket_id: str
    author_name: str
    author_role: str
    is_internal_note: bool
    message: str
    created_at: str


class AuditLogResponse(BaseModel):
    id: int
    ticket_id: str
    action: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    performed_by: str
    timestamp: str


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    employee_name: str = Field(default="Employee")
    employee_email: str = Field(default="employee@company.com")
    department: str = Field(default="Engineering")
    requires_approval: bool = Field(default=False)
    is_approved: bool = Field(default=False)


class AgentChatResponse(BaseModel):
    query: str
    intent: str
    category: str
    priority: str
    action_taken: str
    ai_response: str
    sources: List[Dict[str, Any]]
    ticket_created: Optional[Dict[str, Any]] = None
    draft_ticket: Optional[Dict[str, Any]] = None


class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class KnowledgePassage(BaseModel):
    source: str
    page: int
    similarity_score: float
    text: str


class KnowledgeQueryResponse(BaseModel):
    query: str
    total_retrieved: int
    passages: List[KnowledgePassage]


class AnalyticsMetricsResponse(BaseModel):
    total_tickets: int
    open_tickets: int
    resolved_tickets: int
    critical_incidents: int
    sla_breached: Optional[int] = 0
    resolution_rate_pct: float
    department_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    severity_breakdown: Dict[str, int]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    knowledge_base: str
    agent_engine: str
    timestamp: str
