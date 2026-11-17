from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Ticket


def get_detailed_analytics(db: Session) -> Dict[str, Any]:
    total = db.query(Ticket).count()

    open_count = db.query(Ticket).filter(
        Ticket.status.in_(["Open", "In Progress", "Pending Approval"])
    ).count()

    resolved_count = db.query(Ticket).filter(
        Ticket.status.in_(["Resolved", "Closed"])
    ).count()

    critical_count = db.query(Ticket).filter(
        Ticket.priority == "Critical",
        Ticket.status.notin_(["Resolved", "Closed"])
    ).count()

    now = datetime.utcnow()
    sla_breached_count = db.query(Ticket).filter(
        Ticket.status.notin_(["Resolved", "Closed"]),
        Ticket.sla_due_date.isnot(None),
        Ticket.sla_due_date < now
    ).count()

    resolution_rate = round((resolved_count / total * 100), 1) if total > 0 else 0.0

    severity_breakdown = {
        "Critical": db.query(Ticket).filter(Ticket.priority == "Critical").count(),
        "High": db.query(Ticket).filter(Ticket.priority == "High").count(),
        "Medium": db.query(Ticket).filter(Ticket.priority == "Medium").count(),
        "Low": db.query(Ticket).filter(Ticket.priority == "Low").count(),
    }

    categories = ["Network", "Hardware", "Software", "Access", "Email", "Infrastructure"]
    category_breakdown = {
        cat: db.query(Ticket).filter(Ticket.category.ilike(f"%{cat}%")).count()
        for cat in categories
    }

    dept_rows = db.query(Ticket.department, func.count(Ticket.id)).group_by(Ticket.department).all()
    department_breakdown = {dept: count for dept, count in dept_rows} if dept_rows else {"Engineering": 0}

    status_rows = db.query(Ticket.status, func.count(Ticket.id)).group_by(Ticket.status).all()
    status_breakdown = {st: count for st, count in status_rows} if status_rows else {"Open": 0}

    type_rows = db.query(Ticket.ticket_type, func.count(Ticket.id)).group_by(Ticket.ticket_type).all()
    type_breakdown = {t: count for t, count in type_rows} if type_rows else {"Incident": total}

    return {
        "total_tickets": total,
        "open_tickets": open_count,
        "resolved_tickets": resolved_count,
        "critical_incidents": critical_count,
        "sla_breached": sla_breached_count,
        "resolution_rate_pct": resolution_rate,
        "severity_breakdown": severity_breakdown,
        "category_breakdown": category_breakdown,
        "department_breakdown": department_breakdown,
        "status_breakdown": status_breakdown,
        "type_breakdown": type_breakdown,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S")
    }
