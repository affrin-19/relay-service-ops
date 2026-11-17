"""
Data Models Module for AI-Powered Unified Service Management (USM) System.

Defines the SQLAlchemy ORM models representing IT Service Management entities.
This module translates Python classes into relational database tables.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from database import Base


class Ticket(Base):
    """
    SQLAlchemy ORM Model representing an IT Incident or Service Request Ticket.
    
    Table Name: 'tickets'
    """
    __tablename__ = "tickets"

    # Primary Key: Unique internal integer ID auto-incremented by the database
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Business Ticket Identifier: User-facing unique string code (e.g., USM1001)
    ticket_id = Column(String(20), unique=True, index=True, nullable=False)

    # Classification: Incident (break-fix) vs Service Request (catalog fulfillment)
    ticket_type = Column(String(30), nullable=False, default="Incident")

    # Ticket Summary / Title
    title = Column(String(200), nullable=False)

    # Employee / Requester Details
    employee_name = Column(String(100), nullable=False)
    employee_email = Column(String(120), nullable=False)
    department = Column(String(100), nullable=False)

    # Incident Categorization & Classification
    category = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False, default="Medium")  # Low, Medium, High, Critical
    status = Column(String(30), nullable=False, default="Open")      # Open, In Progress, Pending Approval, Resolved, Closed

    # Detailed Incident Description
    description = Column(Text, nullable=False)

    # Assignment & Lifecycle Tracking
    assigned_technician = Column(String(100), nullable=False, default="Unassigned")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Service Level Agreement (SLA) Tracking
    response_due_date = Column(DateTime, nullable=True)
    sla_due_date = Column(DateTime, nullable=True)
    is_sla_breached = Column(Boolean, default=False, nullable=False)
    resolution_notes = Column(Text, nullable=True)

    def to_dict(self):
        """
        Converts the ORM object into a standard Python dictionary for UI tables and metrics.
        """
        return {
            "ID": self.id,
            "Ticket ID": self.ticket_id,
            "Type": self.ticket_type or "Incident",
            "Title": self.title,
            "Employee": self.employee_name,
            "Email": self.employee_email,
            "Department": self.department,
            "Category": self.category,
            "Priority": self.priority,
            "Status": self.status,
            "Description": self.description,
            "Assigned To": self.assigned_technician,
            "SLA Due": self.sla_due_date.strftime("%Y-%m-%d %H:%M") if self.sla_due_date else "N/A",
            "SLA Breached": "Yes" if self.is_sla_breached else "No",
            "Resolution Notes": self.resolution_notes or "",
            "Created At": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
            "Updated At": self.updated_at.strftime("%Y-%m-%d %H:%M") if self.updated_at else ""
        }

    def to_api_dict(self):
        """Returns programmatic snake_case dictionary matching REST API schemas."""
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "ticket_type": self.ticket_type or "Incident",
            "title": self.title,
            "employee_name": self.employee_name,
            "employee_email": self.employee_email,
            "department": self.department,
            "category": self.category,
            "priority": self.priority,
            "status": self.status,
            "description": self.description,
            "assigned_technician": self.assigned_technician or "Unassigned",
            "response_due_date": self.response_due_date.strftime("%Y-%m-%d %H:%M:%S") if self.response_due_date else None,
            "sla_due_date": self.sla_due_date.strftime("%Y-%m-%d %H:%M:%S") if self.sla_due_date else None,
            "is_sla_breached": bool(self.is_sla_breached),
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else ""
        }

    def __repr__(self):
        return f"<Ticket(id={self.id}, ticket_id='{self.ticket_id}', type='{self.ticket_type}', status='{self.status}')>"


class TicketComment(Base):
    """
    SQLAlchemy ORM Model representing a comment or internal work note on a ticket.
    Enables bi-directional communication between requesters and IT technicians.
    """
    __tablename__ = "ticket_comments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(20), index=True, nullable=False)
    author_name = Column(String(100), nullable=False)
    author_role = Column(String(50), nullable=False, default="requester")  # requester, technician, system
    is_internal_note = Column(Boolean, default=False, nullable=False)       # True for technician-only work notes
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "author_name": self.author_name,
            "author_role": self.author_role,
            "is_internal_note": bool(self.is_internal_note),
            "message": self.message,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else ""
        }


class TicketAuditLog(Base):
    """
    SQLAlchemy ORM Model representing an immutable audit log entry for ticket lifecycle changes.
    Tracks state transitions, assignment changes, and priority modifications.
    """
    __tablename__ = "ticket_audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(20), index=True, nullable=False)
    action = Column(String(50), nullable=False)  # ticket_created, status_change, priority_change, assignment, note_added
    old_value = Column(String(200), nullable=True)
    new_value = Column(String(200), nullable=True)
    performed_by = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "performed_by": self.performed_by,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else ""
        }
