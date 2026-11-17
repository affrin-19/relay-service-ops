"""
FastAPI REST Route Handlers for USM System.

Implements all CRUD endpoints, comments, audit logs,
LangGraph agent chat proxy, ChromaDB vector knowledge retrieval, and analytics.
"""

from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from database import get_db
from models import Ticket
from crud import (
    create_ticket as db_create_ticket,
    get_ticket_by_ticket_id,
    search_and_filter_tickets,
    update_ticket as db_update_ticket,
    delete_ticket as db_delete_ticket,
    add_ticket_comment as db_add_comment,
    get_ticket_comments as db_get_comments,
    get_ticket_audit_logs as db_get_audit_logs
)
from rag.vector_store import USMVectorStore
from agents.graph import run_it_support_agent
from api.analytics import get_detailed_analytics
from api.schemas import (
    TicketCreateRequest,
    TicketUpdateRequest,
    TicketResponse,
    TicketListResponse,
    CommentCreateRequest,
    CommentResponse,
    AuditLogResponse,
    AgentChatRequest,
    AgentChatResponse,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgePassage,
    AnalyticsMetricsResponse,
    HealthResponse
)

router = APIRouter()


# ==============================================================================
# 1. HEALTH CHECK ENDPOINT
# ==============================================================================
@router.get("/health", response_model=HealthResponse, tags=["System Health"])
def health_check(db: Session = Depends(get_db)):
    """
    Returns operational health status across SQLite, ChromaDB, and LangGraph components.
    """
    try:
        db.query(Ticket).first()
        db_status = "Connected (Healthy)"
    except Exception as e:
        db_status = f"Degraded: {str(e)}"

    try:
        vs = USMVectorStore()
        kb_chunks = vs.count()
        kb_status = f"Online ({kb_chunks} chunks stored)"
    except Exception as e:
        kb_status = f"Degraded: {str(e)}"

    return {
        "status": "healthy" if "Healthy" in db_status else "degraded",
        "version": "1.5.0 (Stage 5)",
        "database": db_status,
        "knowledge_base": kb_status,
        "agent_engine": "LangGraph StateGraph (Active)",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==============================================================================
# 2. ANALYTICS TELEMETRY ENDPOINT
# ==============================================================================
@router.get("/api/v1/analytics/metrics", response_model=AnalyticsMetricsResponse, tags=["Analytics & Telemetry"])
def get_analytics(db: Session = Depends(get_db)):
    """
    Calculates real-time incident resolution KPIs, department volume, and severity distributions.
    """
    data = get_detailed_analytics(db)
    return data


# ==============================================================================
# 3. TICKET CRUD ENDPOINTS
# ==============================================================================
@router.post("/api/v1/tickets", response_model=TicketResponse, status_code=status.HTTP_201_CREATED, tags=["Tickets"])
def create_ticket_endpoint(payload: TicketCreateRequest, db: Session = Depends(get_db)):
    """
    Creates a new IT incident or service request ticket with sequential ID (e.g., USM1001).
    """
    ticket_dict = {
        "employee_name": payload.employee_name,
        "employee_email": payload.employee_email,
        "department": payload.department,
        "category": payload.category,
        "title": payload.title,
        "description": payload.description,
        "priority": payload.priority,
        "ticket_type": payload.ticket_type or "Incident",
        "status": "Open",
        "assigned_technician": "Unassigned"
    }

    ticket, err = db_create_ticket(db, ticket_dict)
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    return ticket.to_api_dict()


@router.get("/api/v1/tickets", response_model=TicketListResponse, tags=["Tickets"])
def list_tickets_endpoint(
    search: Optional[str] = Query(None, description="Keyword search across title, description, requester"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    priority_filter: Optional[str] = Query(None, alias="priority", description="Filter by priority"),
    category_filter: Optional[str] = Query(None, alias="category", description="Filter by category"),
    type_filter: Optional[str] = Query(None, alias="type", description="Filter by ticket type (Incident / Service Request)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Searches and filters tickets with pagination.
    """
    matched = search_and_filter_tickets(
        db=db,
        search_query=search or "",
        status_filter=status_filter or "All",
        priority_filter=priority_filter or "All",
        category_filter=category_filter or "All",
        type_filter=type_filter or "All"
    )

    total = len(matched)
    paginated = matched[skip : skip + limit]

    return {
        "total": total,
        "tickets": [t.to_api_dict() for t in paginated]
    }


@router.get("/api/v1/tickets/{ticket_id}", response_model=TicketResponse, tags=["Tickets"])
def get_ticket_endpoint(ticket_id: str, db: Session = Depends(get_db)):
    """
    Retrieves a single ticket by its unique identifier (e.g. USM1001).
    """
    ticket = get_ticket_by_ticket_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")
    return ticket.to_api_dict()


@router.patch("/api/v1/tickets/{ticket_id}", response_model=TicketResponse, tags=["Tickets"])
def update_ticket_endpoint(ticket_id: str, payload: TicketUpdateRequest, db: Session = Depends(get_db)):
    """
    Updates status, priority, or technician with role authorization checks.
    """
    ALLOWED_ROLES = {"admin", "lead_technician", "technician", "it_manager"}
    if payload.auth_role.lower() not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{payload.auth_role}' is not authorized to update tickets. Allowed: {list(ALLOWED_ROLES)}"
        )

    ticket, err = db_update_ticket(
        db=db,
        ticket_id=ticket_id,
        status=payload.status,
        priority=payload.priority,
        assigned_technician=payload.assigned_technician,
        resolution_notes=payload.resolution_notes,
        performed_by=payload.auth_role
    )
    if err:
        if "not found" in err.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    return ticket.to_api_dict()


@router.delete("/api/v1/tickets/{ticket_id}", tags=["Tickets"])
def delete_ticket_endpoint(ticket_id: str, db: Session = Depends(get_db)):
    """
    Permanently deletes a ticket from the database.
    """
    deleted, err = db_delete_ticket(db, ticket_id)
    if not deleted:
        if err and "not found" in err.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err or "Deletion failed.")

    return {"status": "success", "message": f"Ticket '{ticket_id}' deleted successfully."}


# ==============================================================================
# 4. COMMENTS & AUDIT TRAIL ENDPOINTS
# ==============================================================================
@router.post("/api/v1/tickets/{ticket_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED, tags=["Ticket Collaboration"])
def add_comment_endpoint(ticket_id: str, payload: CommentCreateRequest, db: Session = Depends(get_db)):
    """
    Appends a new user comment or internal technician work note to a ticket thread.
    """
    ticket = get_ticket_by_ticket_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")

    comment, err = db_add_comment(
        db=db,
        ticket_id=ticket_id,
        author_name=payload.author_name,
        author_role=payload.author_role,
        message=payload.message,
        is_internal_note=payload.is_internal_note
    )
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    return comment.to_dict()


@router.get("/api/v1/tickets/{ticket_id}/comments", response_model=List[CommentResponse], tags=["Ticket Collaboration"])
def list_comments_endpoint(
    ticket_id: str,
    include_internal: bool = Query(True, description="Include technician internal notes"),
    db: Session = Depends(get_db)
):
    """
    Retrieves the chronological message history for a ticket.
    """
    ticket = get_ticket_by_ticket_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")

    comments = db_get_comments(db, ticket_id, include_internal=include_internal)
    return [c.to_dict() for c in comments]


@router.get("/api/v1/tickets/{ticket_id}/audit-logs", response_model=List[AuditLogResponse], tags=["Ticket Collaboration"])
def list_audit_logs_endpoint(ticket_id: str, db: Session = Depends(get_db)):
    """
    Retrieves immutable audit timeline for ticket status, assignment, and priority changes.
    """
    ticket = get_ticket_by_ticket_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")

    logs = db_get_audit_logs(db, ticket_id)
    return [l.to_dict() for l in logs]


# ==============================================================================
# 5. LANGGRAPH AGENT CHAT ENDPOINT
# ==============================================================================
@router.post("/api/v1/agent/chat", response_model=AgentChatResponse, tags=["AI Agent Workflows"])
def agent_chat_endpoint(payload: AgentChatRequest):
    """
    Executes the multi-agent LangGraph triage, RAG retrieval, and ticket generation pipeline.
    """
    try:
        result = run_it_support_agent(
            user_query=payload.query,
            employee_name=payload.employee_name,
            employee_email=payload.employee_email,
            department=payload.department,
            conversation_history=[],
            requires_approval=payload.requires_approval,
            is_approved=payload.is_approved
        )
        return {
            "query": payload.query,
            "intent": result.get("intent", "general_chat"),
            "category": result.get("category", "General"),
            "priority": result.get("priority", "Medium"),
            "action_taken": result.get("action_taken", "Consultation"),
            "ai_response": result.get("ai_response", ""),
            "sources": result.get("sources", []),
            "ticket_created": result.get("ticket_created"),
            "draft_ticket": result.get("draft_ticket")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Agent execution failed: {str(e)}"
        )


# ==============================================================================
# 6. KNOWLEDGE BASE & RAG ENDPOINTS
# ==============================================================================
@router.post("/api/v1/knowledge/query", response_model=KnowledgeQueryResponse, tags=["Knowledge Base (RAG)"])
def query_knowledge_base_endpoint(payload: KnowledgeQueryRequest):
    """
    Executes dense vector similarity search against ChromaDB IT support documents.
    """
    try:
        vs = USMVectorStore()
        chunks = vs.similarity_search(query=payload.query, top_k=payload.top_k)

        passages = [
            KnowledgePassage(
                source=ch.get("source", "Unknown"),
                page=ch.get("page", 1),
                similarity_score=float(ch.get("similarity", 0.0)),
                text=ch.get("text", "")
            )
            for ch in chunks
        ]

        return {
            "query": payload.query,
            "total_retrieved": len(passages),
            "passages": passages
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector knowledge search failed: {str(e)}"
        )


@router.get("/api/v1/knowledge/documents", tags=["Knowledge Base (RAG)"])
def list_knowledge_documents_endpoint():
    """
    Lists all indexed PDF manuals, policies, and SOPs currently stored in ChromaDB.
    """
    try:
        vs = USMVectorStore()
        docs = vs.get_indexed_documents()
        return {"total_documents": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


# ==============================================================================
# 8. AI COPILOT & ESM TELEMETRY ENDPOINTS (HELIXGPT & ATASSIAN ESM)
# ==============================================================================
@router.get("/api/v1/tickets/{ticket_id}/copilot-brief", tags=["AI Copilot"])
def get_ticket_copilot_briefing(ticket_id: str, db: Session = Depends(get_db)):
    """
    Synthesizes executive briefing, similar historical incidents, and draft resolution
    for technician incident investigation.
    """
    from ai_copilot import (
        summarize_incident_thread,
        correlate_similar_incidents,
        draft_smart_resolution,
        analyze_sentiment_and_governance
    )

    ticket = get_ticket_by_ticket_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket '{ticket_id}' not found.")

    comments = db_get_comments(db, ticket_id, include_internal=True)
    summary = summarize_incident_thread(ticket, comments)
    similar = correlate_similar_incidents(ticket, db, top_k=3)
    best_similar = similar[0] if similar else None
    resol_draft = draft_smart_resolution(ticket, similar_incident=best_similar)
    sentiment = analyze_sentiment_and_governance(ticket)

    return {
        "ticket_id": ticket.ticket_id,
        "summary": summary,
        "similar_incidents": similar,
        "draft_resolution": resol_draft,
        "sentiment_governance": sentiment
    }


@router.get("/api/v1/telemetry/esm-kpis", tags=["Telemetry & ESM Analytics"])
def get_esm_kpis_endpoint(db: Session = Depends(get_db)):
    """
    Computes Enterprise Service Management (ESM) KPIs: Deflection Rate, Hours Saved,
    CSAT index, and Proactive SLA Breach Radar.
    """
    from ai_copilot import get_esm_business_value_kpis
    return get_esm_business_value_kpis(db)
