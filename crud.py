import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from models import Ticket, TicketComment, TicketAuditLog
from notifications import dispatch_incident_alert


def validate_ticket_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    required_fields = {
        "employee_name": "Employee Name",
        "employee_email": "Work Email",
        "department": "Department",
        "category": "Category",
        "title": "Incident Title",
        "description": "Description"
    }

    for key, label in required_fields.items():
        val = str(data.get(key, "")).strip()
        if not val:
            return False, f"Field '{label}' is required and cannot be empty."

    email = data.get("employee_email", "").strip()
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        return False, "Please provide a valid work email address (e.g., employee@company.com)."

    if len(data.get("title", "").strip()) < 5:
        return False, "Title is too short. Please provide at least 5 characters describing the issue."

    if len(data.get("description", "").strip()) < 10:
        return False, "Description is too short. Please provide at least 10 characters detailing the problem."

    return True, ""


def calculate_sla_deadlines(priority: str, base_time: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    base = base_time or datetime.utcnow()
    prio_norm = priority.strip().capitalize() if priority else "Medium"

    if prio_norm == "Critical":
        return base + timedelta(minutes=15), base + timedelta(hours=4)
    if prio_norm == "High":
        return base + timedelta(hours=1), base + timedelta(hours=8)
    if prio_norm == "Low":
        return base + timedelta(hours=8), base + timedelta(hours=72)
    return base + timedelta(hours=4), base + timedelta(hours=24)


def generate_ticket_id(db: Session) -> str:
    last_ticket = db.query(Ticket).order_by(Ticket.id.desc()).first()
    if not last_ticket:
        return "USM1001"

    ticket_id_str = str(last_ticket.ticket_id)
    if ticket_id_str.startswith("USM"):
        try:
            num = int(ticket_id_str.replace("USM", ""))
            return f"USM{num + 1}"
        except ValueError:
            pass

    return f"USM{1000 + last_ticket.id + 1}"


def create_ticket(db: Session, data: Dict[str, Any]) -> Tuple[Optional[Ticket], Optional[str]]:
    is_valid, validation_error = validate_ticket_input(data)
    if not is_valid:
        return None, validation_error

    try:
        new_ticket_id = generate_ticket_id(db)
        priority = data.get("priority", "Medium")
        ticket_type = data.get("ticket_type", "Incident")

        now = datetime.utcnow()
        resp_due, resol_due = calculate_sla_deadlines(priority, now)

        ticket = Ticket(
            ticket_id=new_ticket_id,
            ticket_type=ticket_type,
            title=data["title"].strip(),
            employee_name=data["employee_name"].strip(),
            employee_email=data["employee_email"].strip().lower(),
            department=data["department"],
            category=data["category"],
            priority=priority,
            status=data.get("status", "Open"),
            description=data["description"].strip(),
            assigned_technician=data.get("assigned_technician", "Unassigned"),
            created_at=now,
            updated_at=now,
            response_due_date=resp_due,
            sla_due_date=resol_due,
            is_sla_breached=False,
            resolution_notes=data.get("resolution_notes")
        )

        db.add(ticket)
        db.flush()

        audit = TicketAuditLog(
            ticket_id=new_ticket_id,
            action="ticket_created",
            old_value=None,
            new_value=f"Type={ticket_type}, Status={ticket.status}, Priority={priority}",
            performed_by=data.get("employee_name", "System"),
            timestamp=now
        )
        db.add(audit)

        db.commit()
        db.refresh(ticket)

        if priority in ["Critical", "High"]:
            dispatch_incident_alert(ticket.to_api_dict())

        return ticket, None

    except SQLAlchemyError as e:
        db.rollback()
        return None, f"Database transaction failed: {str(e)}"


def get_all_tickets(db: Session, limit: int = 500) -> List[Ticket]:
    try:
        return db.query(Ticket).order_by(Ticket.id.desc()).limit(limit).all()
    except SQLAlchemyError:
        return []


def get_ticket_by_ticket_id(db: Session, ticket_id: str) -> Optional[Ticket]:
    try:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id.strip()).first()
        if ticket and ticket.sla_due_date:
            if ticket.status not in ["Resolved", "Closed"] and datetime.utcnow() > ticket.sla_due_date:
                if not ticket.is_sla_breached:
                    ticket.is_sla_breached = True
                    db.commit()
        return ticket
    except SQLAlchemyError:
        return None


def search_and_filter_tickets(
    db: Session,
    search_query: str = "",
    status_filter: str = "All",
    priority_filter: str = "All",
    category_filter: str = "All",
    type_filter: str = "All"
) -> List[Ticket]:
    try:
        query = db.query(Ticket)

        if status_filter and status_filter != "All":
            query = query.filter(Ticket.status == status_filter)

        if priority_filter and priority_filter != "All":
            query = query.filter(Ticket.priority == priority_filter)

        if category_filter and category_filter != "All":
            query = query.filter(Ticket.category == category_filter)

        if type_filter and type_filter != "All":
            query = query.filter(Ticket.ticket_type == type_filter)

        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            query = query.filter(
                (Ticket.ticket_id.ilike(term)) |
                (Ticket.title.ilike(term)) |
                (Ticket.employee_name.ilike(term)) |
                (Ticket.employee_email.ilike(term)) |
                (Ticket.department.ilike(term)) |
                (Ticket.description.ilike(term))
            )

        return query.order_by(Ticket.id.desc()).all()

    except SQLAlchemyError:
        return []


def get_dashboard_metrics(db: Session) -> Dict[str, Any]:
    try:
        total_tickets = db.query(Ticket).count()

        open_tickets = db.query(Ticket).filter(
            Ticket.status.in_(["Open", "In Progress", "Pending Approval"])
        ).count()

        resolved_tickets = db.query(Ticket).filter(
            Ticket.status.in_(["Resolved", "Closed"])
        ).count()

        critical_incidents = db.query(Ticket).filter(
            Ticket.priority == "Critical",
            Ticket.status.notin_(["Resolved", "Closed"])
        ).count()

        now = datetime.utcnow()
        sla_breached_count = db.query(Ticket).filter(
            Ticket.status.notin_(["Resolved", "Closed"]),
            Ticket.sla_due_date.isnot(None),
            Ticket.sla_due_date < now
        ).count()

        severity_counts = {
            "Critical": db.query(Ticket).filter(Ticket.priority == "Critical").count(),
            "High": db.query(Ticket).filter(Ticket.priority == "High").count(),
            "Medium": db.query(Ticket).filter(Ticket.priority == "Medium").count(),
            "Low": db.query(Ticket).filter(Ticket.priority == "Low").count(),
        }

        type_counts = {
            "Incident": db.query(Ticket).filter(Ticket.ticket_type == "Incident").count(),
            "Service Request": db.query(Ticket).filter(Ticket.ticket_type == "Service Request").count(),
        }

        return {
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "resolved_tickets": resolved_tickets,
            "critical_incidents": critical_incidents,
            "sla_breached": sla_breached_count,
            "severity_counts": severity_counts,
            "type_counts": type_counts
        }

    except SQLAlchemyError:
        return {
            "total_tickets": 0,
            "open_tickets": 0,
            "resolved_tickets": 0,
            "critical_incidents": 0,
            "sla_breached": 0,
            "severity_counts": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            "type_counts": {"Incident": 0, "Service Request": 0}
        }


def update_ticket(
    db: Session,
    ticket_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_technician: Optional[str] = None,
    resolution_notes: Optional[str] = None,
    performed_by: str = "IT Staff"
) -> Tuple[Optional[Ticket], Optional[str]]:
    try:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id.strip()).first()
        if not ticket:
            return None, f"Ticket '{ticket_id}' not found in database."

        now = datetime.utcnow()

        if status and status != ticket.status:
            old_s = ticket.status
            ticket.status = status
            db.add(TicketAuditLog(
                ticket_id=ticket.ticket_id,
                action="status_change",
                old_value=old_s,
                new_value=status,
                performed_by=performed_by,
                timestamp=now
            ))

        if priority and priority != ticket.priority:
            old_p = ticket.priority
            ticket.priority = priority
            if ticket.status not in ["Resolved", "Closed"]:
                _, resol_due = calculate_sla_deadlines(priority, ticket.created_at)
                ticket.sla_due_date = resol_due
            db.add(TicketAuditLog(
                ticket_id=ticket.ticket_id,
                action="priority_change",
                old_value=old_p,
                new_value=priority,
                performed_by=performed_by,
                timestamp=now
            ))

        if assigned_technician and assigned_technician != ticket.assigned_technician:
            old_tech = ticket.assigned_technician
            ticket.assigned_technician = assigned_technician
            db.add(TicketAuditLog(
                ticket_id=ticket.ticket_id,
                action="assignment",
                old_value=old_tech,
                new_value=assigned_technician,
                performed_by=performed_by,
                timestamp=now
            ))

        if resolution_notes:
            ticket.resolution_notes = resolution_notes

        ticket.updated_at = now

        db.commit()
        db.refresh(ticket)
        return ticket, None

    except SQLAlchemyError as e:
        db.rollback()
        return None, f"Failed to update ticket: {str(e)}"


def delete_ticket(db: Session, ticket_id: str) -> Tuple[bool, Optional[str]]:
    try:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id.strip()).first()
        if not ticket:
            return False, f"Ticket '{ticket_id}' not found in database."

        db.query(TicketComment).filter(TicketComment.ticket_id == ticket_id.strip()).delete()
        db.query(TicketAuditLog).filter(TicketAuditLog.ticket_id == ticket_id.strip()).delete()

        db.delete(ticket)
        db.commit()
        return True, None

    except SQLAlchemyError as e:
        db.rollback()
        return False, f"Failed to delete ticket: {str(e)}"


def add_ticket_comment(
    db: Session,
    ticket_id: str,
    author_name: str,
    message: str,
    author_role: str = "requester",
    is_internal_note: bool = False
) -> Tuple[Optional[TicketComment], Optional[str]]:
    if not message or not message.strip():
        return None, "Comment message cannot be empty."

    try:
        comment = TicketComment(
            ticket_id=ticket_id.strip(),
            author_name=author_name.strip(),
            author_role=author_role.strip(),
            is_internal_note=is_internal_note,
            message=message.strip(),
            created_at=datetime.utcnow()
        )
        db.add(comment)

        note_type = "internal_note" if is_internal_note else "public_comment"
        db.add(TicketAuditLog(
            ticket_id=ticket_id.strip(),
            action=note_type,
            old_value=None,
            new_value=f"{author_name}: {message[:40]}...",
            performed_by=author_name.strip(),
            timestamp=datetime.utcnow()
        ))

        db.commit()
        db.refresh(comment)
        return comment, None

    except SQLAlchemyError as e:
        db.rollback()
        return None, f"Failed to add comment: {str(e)}"


def get_ticket_comments(db: Session, ticket_id: str, include_internal: bool = True) -> List[TicketComment]:
    try:
        query = db.query(TicketComment).filter(TicketComment.ticket_id == ticket_id.strip())
        if not include_internal:
            query = query.filter(TicketComment.is_internal_note == False)
        return query.order_by(TicketComment.created_at.asc()).all()
    except SQLAlchemyError:
        return []


def get_ticket_audit_logs(db: Session, ticket_id: str) -> List[TicketAuditLog]:
    try:
        return db.query(TicketAuditLog).filter(
            TicketAuditLog.ticket_id == ticket_id.strip()
        ).order_by(TicketAuditLog.timestamp.asc()).all()
    except SQLAlchemyError:
        return []


def seed_initial_tickets(db: Session) -> int:
    if db.query(Ticket).count() > 0:
        return 0

    seeds = [
        {
            "ticket_type": "Incident",
            "employee_name": "Jordan Hayes",
            "employee_email": "jordan.hayes@company.com",
            "department": "Engineering",
            "category": "Network",
            "title": "VPN connection timeout for remote engineering staff",
            "priority": "Critical",
            "status": "In Progress",
            "description": "Engineers connecting through the West Coast gateway report frequent 60-second timeouts when pulling git repositories or accessing internal staging clusters.",
            "assigned_technician": "Alex Chen (Network)"
        },
        {
            "ticket_type": "Service Request",
            "employee_name": "Sarah Miller",
            "employee_email": "sarah.m@company.com",
            "department": "Engineering",
            "category": "Software",
            "title": "Docker Desktop Enterprise License Renewal",
            "priority": "Medium",
            "status": "Pending Approval",
            "description": "Annual enterprise license key expired yesterday. Need new commercial license provisioned to run local containerized microservices.",
            "assigned_technician": "Devon Miller (Cloud)"
        },
        {
            "ticket_type": "Incident",
            "employee_name": "Marcus Vance",
            "employee_email": "marcus.v@company.com",
            "department": "IT & Infrastructure",
            "category": "Access",
            "title": "Single Sign-On (SSO) gateway latency spike",
            "priority": "High",
            "status": "Open",
            "description": "Okta SSO authentication takes between 8 and 14 seconds to respond, intermittently triggering 504 Gateway Timeouts across corporate web applications.",
            "assigned_technician": "Sarah Jenkins (Security)"
        },
        {
            "ticket_type": "Incident",
            "employee_name": "Priya Sharma",
            "employee_email": "priya.s@company.com",
            "department": "Marketing",
            "category": "Email",
            "title": "Email synchronization failure on Outlook iOS clients",
            "priority": "Medium",
            "status": "Resolved",
            "description": "Mobile Exchange sync stalled after multi-factor authentication token renewal. Re-authenticated app password to restore sync.",
            "assigned_technician": "Liam Patel (Tier 1 Support)",
            "resolution_notes": "User guided through MFA session re-authentication. Outlook mobile cache invalidated."
        },
        {
            "ticket_type": "Service Request",
            "employee_name": "David Ross",
            "employee_email": "david.ross@company.com",
            "department": "Sales",
            "category": "Hardware",
            "title": "Standard Developer Workstation Provisioning",
            "priority": "Low",
            "status": "Open",
            "description": "MacBook Pro 16-inch M3 Max, dual 27-inch 4K monitors, and USB-C hub required for incoming Account Executive starting next Monday.",
            "assigned_technician": "Unassigned"
        }
    ]

    count = 0
    for item in seeds:
        ticket, err = create_ticket(db, item)
        if ticket:
            count += 1
            if ticket.ticket_id == "USM1001":
                add_ticket_comment(
                    db,
                    ticket.ticket_id,
                    author_name="Alex Chen (Network)",
                    author_role="technician",
                    is_internal_note=True,
                    message="Inspecting West Coast VPN gateway IPSEC routing tables. CPU load is normal, checking transit ISP packet loss."
                )
                add_ticket_comment(
                    db,
                    ticket.ticket_id,
                    author_name="Jordan Hayes",
                    author_role="requester",
                    is_internal_note=False,
                    message="Thanks Alex. The timeouts are specifically affecting port 443 git clone operations from GitHub Enterprise."
                )

    return count
