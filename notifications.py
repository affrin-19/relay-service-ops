import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("USM_NOTIFICATIONS")
logger.setLevel(logging.INFO)


def dispatch_incident_alert(ticket_data: Dict[str, Any], webhook_url: Optional[str] = None) -> Dict[str, Any]:
    priority = ticket_data.get("priority", "Medium")
    ticket_id = ticket_data.get("ticket_id", "N/A")
    title = ticket_data.get("title", "Untitled Incident")
    category = ticket_data.get("category", "General")
    requester = ticket_data.get("employee_name", "Anonymous")
    department = ticket_data.get("department", "General")

    target_url = webhook_url or os.getenv("INCIDENT_WEBHOOK_URL", "")

    labels = {
        "Critical": "[P1 CRITICAL]",
        "High": "[P2 HIGH]",
        "Medium": "[P3 MEDIUM]",
        "Low": "[P4 LOW]"
    }
    header_text = labels.get(priority, "[INCIDENT ALERT]")

    payload = {
        "text": f"{header_text}: {ticket_id} - {title}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{header_text}: {ticket_id}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Title:*\n{title}"},
                    {"type": "mrkdwn", "text": f"*Priority:*\n`{priority}`"},
                    {"type": "mrkdwn", "text": f"*Category:*\n{category}"},
                    {"type": "mrkdwn", "text": f"*Requester:*\n{requester} ({department})"},
                ]
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Logged at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
                ]
            }
        ]
    }

    if target_url and target_url.startswith("http"):
        try:
            import httpx
            response = httpx.post(target_url, json=payload, timeout=4.0)
            logger.info(f"Webhook alert dispatched to {target_url[:25]}... (Status: {response.status_code})")
            return {"dispatched": True, "channel": "webhook", "status": response.status_code}
        except Exception as e:
            logger.warning(f"Webhook dispatch failed: {e}")
            return {"dispatched": False, "channel": "webhook", "error": str(e)}

    logger.info(f"[simulated alert] {header_text} {ticket_id}: '{title}' by {requester}")
    return {
        "dispatched": True,
        "channel": "simulated_local",
        "message": f"Alert queued for {requester} regarding {ticket_id}",
        "timestamp": datetime.utcnow().isoformat()
    }


def send_ticket_status_email(ticket_id: str, recipient_email: str, new_status: str, message: str) -> Dict[str, Any]:
    logger.info(f"[email] To: {recipient_email} | Ticket: {ticket_id} | Status: {new_status}")
    return {
        "sent": True,
        "recipient": recipient_email,
        "ticket_id": ticket_id,
        "status": new_status,
        "timestamp": datetime.utcnow().isoformat()
    }
