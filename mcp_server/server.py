"""
Official MCP Server for Unified Service Management (USM) System.

Exposes IT Service Management capabilities as Model Context Protocol (MCP) tools
compliant with the official Python MCP SDK (mcp 2.x MCPServer).
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to sys.path so modules can be imported when run standalone
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

# Ensure database is initialized
init_db()

# Initialize the MCP Server instance
server = MCPServer("usm-mcp-server")


# ==============================================================================
# TOOL 1: CREATE TICKET
# ==============================================================================
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
    """
    Creates a new IT support incident or request ticket in the USM database.
    
    Args:
        employee_name: Requester's full name.
        employee_email: Requester's corporate email address.
        department: Requester's department (e.g., Engineering, Sales, HR, Finance).
        category: Technical category (Hardware, Software, Network, Access & Security, Infrastructure, Email).
        title: Short summary of the incident.
        description: Detailed reproduction steps and symptoms.
        priority: Urgency level (Low, Medium, High, Critical). Default is Medium.
        
    Returns:
        JSON string containing the created ticket details and assigned Ticket ID (e.g., USM1006).
    """
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


# ==============================================================================
# TOOL 2: GET TICKET
# ==============================================================================
@server.tool()
def get_ticket(ticket_id: str) -> str:
    """
    Retrieves full details of an existing ticket using its unique business ID.
    
    Args:
        ticket_id: The unique ticket identifier (e.g., 'USM1001').
        
    Returns:
        JSON string with ticket attributes, or an error if not found.
    """
    db = next(get_db())
    try:
        ticket = get_ticket_by_ticket_id(db, ticket_id)
        if not ticket:
            return json.dumps({"status": "error", "message": f"Ticket '{ticket_id}' not found."})
        return json.dumps({"status": "success", "ticket": ticket.to_dict()}, indent=2)
    finally:
        db.close()


# ==============================================================================
# TOOL 3: SEARCH TICKETS
# ==============================================================================
@server.tool()
def search_tickets(
    search_query: str = "",
    status: str = "All",
    priority: str = "All",
    category: str = "All"
) -> str:
    """
    Searches and filters tickets by keywords, status, priority, or technical category.
    
    Args:
        search_query: Keyword search across title, description, employee name, or ticket ID.
        status: Filter by lifecycle status (Open, In Progress, Pending Approval, Resolved, Closed, or All).
        priority: Filter by urgency (Low, Medium, High, Critical, or All).
        category: Filter by category (Network, Hardware, Software, etc., or All).
        
    Returns:
        JSON string containing list of matching ticket summaries.
    """
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


# ==============================================================================
# TOOL 4: UPDATE TICKET STATUS (WITH AUTHORIZATION CHECKS)
# ==============================================================================
@server.tool()
def update_ticket_status(
    ticket_id: str,
    new_status: str,
    auth_role: str = "technician",
    assigned_technician: Optional[str] = None,
    technician: Optional[str] = None,
    priority: Optional[str] = None
) -> str:
    """
    Updates the lifecycle status and assignment of an existing ticket.
    Requires authorization check against caller's role.
    
    Args:
        ticket_id: Target ticket identifier (e.g., 'USM1001').
        new_status: New status (Open, In Progress, Pending Approval, Resolved, Closed).
        auth_role: Role of the actor attempting update (admin, lead_technician, technician, or viewer).
        assigned_technician: Optional new technician name to reassign ticket to.
        technician: Alias for assigned_technician.
        priority: Optional priority override.
        
    Returns:
        JSON string confirming update or returning permission denied error.
    """
    # Normalize technician argument
    target_tech = assigned_technician or technician
    # Authorization Check: only authorized roles can modify status
    ALLOWED_ROLES = {"admin", "lead_technician", "technician", "it_manager"}
    if auth_role.lower() not in ALLOWED_ROLES:
        return json.dumps({
            "status": "forbidden",
            "message": f"Authorization Error: Role '{auth_role}' is not authorized to update tickets. Required roles: {list(ALLOWED_ROLES)}"
        })

    VALID_STATUSES = {"Open", "In Progress", "Pending Approval", "Resolved", "Closed"}
    if new_status not in VALID_STATUSES:
        return json.dumps({
            "status": "error",
            "message": f"Invalid status '{new_status}'. Allowed statuses: {list(VALID_STATUSES)}"
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


# ==============================================================================
# TOOL 5: SEARCH KNOWLEDGE BASE
# ==============================================================================
@server.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Searches the persistent ChromaDB vector store for IT support documentation and SOPs.
    
    Args:
        query: Natural language query (e.g., 'How do I connect to VPN?', 'Password complexity rules').
        top_k: Number of relevant passages to retrieve (default 3).
        
    Returns:
        JSON string with retrieved document passages, source names, page numbers, and similarity scores.
    """
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
    """Starts the MCP server on stdio transport for IDE / Agent integration."""
    server.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
