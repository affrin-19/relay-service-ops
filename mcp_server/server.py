import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer
from database import get_db, init_db
from crud import (
    create_ticket as db_create_ticket,
    get_ticket_by_ticket_id,
    search_and_filter_tickets,
    update_ticket as db_update_ticket,
    get_dashboard_metrics
)
from rag.vector_store import USMVectorStore

init_db()

server = MCPServer("usm-mcp-server")


@server.tool()
def create_ticket(
    employee_name: str,
    employee_email: str,
    department: str,
    category: str,
    title: str,
    description: str,
    priority: str = "Medium"
) -> str:
    """Create a new IT incident or request ticket."""
    db = next(get_db())
    try:
        data = {
            "employee_name": employee_name,
            "employee_email": employee_email,
            "department": department,
            "category": category,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "Open",
            "assigned_technician": "Unassigned"
        }
        ticket, err = db_create_ticket(db, data)
        if err:
            return json.dumps({"status": "error", "message": err})
        return json.dumps({
            "status": "success",
            "ticket": ticket.to_dict(),
            "message": f"Ticket {ticket.ticket_id} created successfully."
        }, indent=2)
    finally:
        db.close()


@server.tool()
def get_ticket(ticket_id: str) -> str:
    """Retrieve details for a single ticket by ID."""
    db = next(get_db())
    try:
        ticket = get_ticket_by_ticket_id(db, ticket_id)
        if not ticket:
            return json.dumps({"status": "error", "message": f"Ticket '{ticket_id}' not found."})
        return json.dumps({"status": "success", "ticket": ticket.to_dict()}, indent=2)
    finally:
        db.close()


@server.tool()
def search_tickets(
    search_query: str = "",
    status: str = "All",
    priority: str = "All",
    category: str = "All"
) -> str:
    """Search and filter tickets by query, status, priority, or category."""
    db = next(get_db())
    try:
        matched = search_and_filter_tickets(
            db=db,
            search_query=search_query,
            status_filter=status,
            priority_filter=priority,
            category_filter=category
        )
        results = [t.to_dict() for t in matched]
        return json.dumps({
            "status": "success",
            "total_matches": len(results),
            "tickets": results
        }, indent=2)
    finally:
        db.close()


@server.tool()
def update_ticket_status(
    ticket_id: str,
    new_status: str,
    auth_role: str = "technician",
    assigned_technician: Optional[str] = None,
    technician: Optional[str] = None,
    priority: Optional[str] = None
) -> str:
    """Update lifecycle status and technician assignment for a ticket."""
    target_tech = assigned_technician or technician
    allowed_roles = {"admin", "lead_technician", "technician", "it_manager"}
    if auth_role.lower() not in allowed_roles:
        return json.dumps({
            "status": "forbidden",
            "message": f"Authorization Error: Role '{auth_role}' is not authorized to update tickets. Required roles: {list(allowed_roles)}"
        })

    valid_statuses = {"Open", "In Progress", "Pending Approval", "Resolved", "Closed"}
    if new_status not in valid_statuses:
        return json.dumps({
            "status": "error",
            "message": f"Invalid status '{new_status}'. Allowed statuses: {list(valid_statuses)}"
        })

    db = next(get_db())
    try:
        ticket, err = db_update_ticket(
            db=db,
            ticket_id=ticket_id,
            status=new_status,
            priority=priority,
            assigned_technician=target_tech
        )
        if err:
            return json.dumps({"status": "error", "message": err})
        return json.dumps({
            "status": "success",
            "ticket": ticket.to_dict(),
            "message": f"Ticket '{ticket_id}' updated to status '{new_status}' by {auth_role}."
        }, indent=2)
    finally:
        db.close()


@server.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search knowledge base chunks for SOPs and IT documentation."""
    try:
        vs = USMVectorStore()
        results = vs.similarity_search(query=query, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "source": r.get("source"),
                "page": r.get("page"),
                "similarity_score": r.get("similarity_score"),
                "text_snippet": r.get("text")
            })
        return json.dumps({
            "status": "success",
            "query": query,
            "total_retrieved": len(formatted),
            "passages": formatted
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Knowledge base search failed: {str(e)}"})


def run_mcp_server():
    server.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
