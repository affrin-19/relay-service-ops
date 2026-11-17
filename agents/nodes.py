import os
import json
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI

from agents.state import AgentState

try:
    import importlib
    import config
    importlib.reload(config)
    from config import (
        OPENAI_API_KEY,
        LLM_MODEL_NAME,
        get_openai_api_key,
        is_openai_configured,
        get_masked_api_key
    )
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    def get_openai_api_key():
        return os.getenv("OPENAI_API_KEY", "")

    def is_openai_configured():
        k = get_openai_api_key()
        return bool(k and k.startswith("sk-") and len(k) > 20)

    def get_masked_api_key():
        k = get_openai_api_key()
        return f"{k[:7]}...{k[-4:]}" if len(k) > 12 else ("Configured" if k else "Not Configured")

from database import get_db
from crud import create_ticket
from rag.vector_store import USMVectorStore
from rag.rag_chain import generate_rag_answer

logger = logging.getLogger("USM_AGENT")
logger.setLevel(logging.INFO)


def get_chat_llm(temperature: float = 0.0) -> Optional[ChatOpenAI]:
    key = get_openai_api_key()
    if key and key.startswith("sk-") and len(key) > 20:
        return ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=temperature,
            api_key=key
        )
    return None


TRIAGE_SYSTEM_PROMPT = """You are an ITIL Service Desk triage assistant.
Analyze the employee request and classify it into JSON format.

JSON Schema:
{
    "intent": "information_retrieval" | "incident_creation" | "general_chat",
    "category": "Hardware" | "Software" | "Network" | "Email" | "Access",
    "priority": "Low" | "Medium" | "High" | "Critical",
    "title": "Clear 5-10 word summary of the incident or request"
}

Classification Guidelines:
- intent:
    - "incident_creation": user asks to create/log a ticket, reports an outage, severe error, or broken equipment.
    - "information_retrieval": user asks how to do something, asks for troubleshooting guidance, policies, passwords, VPN, or software config.
    - "general_chat": greetings or questions about what you can do without technical problems.
- category:
    - "Network": VPN, Wi-Fi, DNS, gateway, internet connection timeouts.
    - "Hardware": Laptop, monitor, keyboard, docking station, broken physical components.
    - "Software": Application crashes, Docker, development tools, licenses.
    - "Email": Outlook, Exchange, inbox sync, distribution lists, email authentication.
    - "Access": Passwords, account lockout, Okta, SSO, permissions, MFA tokens, logins.
- priority:
    - "Critical": Complete outage, production failure, security breach.
    - "High": Major blocker impacting employee core work without workaround.
    - "Medium": Standard technical issue with partial workaround.
    - "Low": Minor questions, peripheral requests, cosmetic issues.

Return ONLY valid JSON."""


def _heuristic_triage(query: str) -> Dict[str, Any]:
    q = query.lower().strip()

    it_indicators = [
        "vpn", "wifi", "wi-fi", "network", "dns", "gateway", "firewall", "latency", "ping", "internet", "connection",
        "laptop", "macbook", "monitor", "keyboard", "mouse", "usb", "dock", "hardware", "screen", "cable", "display", "charger",
        "email", "outlook", "exchange", "inbox", "mail", "calendar", "invite", "mailbox", "send or receive",
        "password", "lockout", "locked", "sso", "okta", "mfa", "login", "access", "permission", "credentials", "2fa", "authenticate", "authentication",
        "software", "app", "application", "crash", "error", "bug", "docker", "git", "install", "update", "license",
        "troubleshoot", "troubleshooting", "setup", "configure", "how to", "policy", "guideline", "steps", "requirement"
    ]
    has_it_indicator = any(k in q for k in it_indicators)

    ticket_triggers = [
        "create ticket", "create a ticket", "log ticket", "log a ticket",
        "open ticket", "open a ticket", "report incident", "submit ticket",
        "file a ticket", "please create", "please log", "raise a ticket",
        "raise ticket", "report an incident", "submit a ticket", "open an incident"
    ]

    is_greeting = (
        q in ["help", "help me", "can you help me", "i need help", "hello", "hi", "hey"]
        or any(phrase in q for phrase in ["who are you", "what can you do", "introduce yourself", "how do you work", "tell me about yourself"])
        or (any(g in q for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "greetings"]) and len(q.split()) <= 4 and not has_it_indicator)
    )

    if any(k in q for k in ticket_triggers):
        intent = "incident_creation"
    elif is_greeting and not has_it_indicator:
        intent = "general_chat"
    else:
        intent = "information_retrieval"

    if any(k in q for k in ["vpn", "wifi", "wi-fi", "network", "dns", "gateway", "ping", "firewall", "ip address", "latency", "internet", "connection"]):
        category = "Network"
    elif any(k in q for k in ["email", "outlook", "exchange", "inbox", "mail", "calendar", "invite", "send or receive", "mailbox"]):
        category = "Email"
    elif any(k in q for k in ["password", "lockout", "locked", "sso", "okta", "mfa", "login", "access", "permission", "credentials", "2fa", "authentication", "authenticate"]):
        category = "Access"
    elif any(k in q for k in ["laptop", "macbook", "monitor", "keyboard", "mouse", "usb", "dock", "hardware", "screen", "cable", "display", "charger"]):
        category = "Hardware"
    else:
        category = "Software"

    if any(k in q for k in ["down", "outage", "emergency", "production", "crash", "critical", "blocked all", "dropping all"]):
        priority = "Critical"
    elif any(k in q for k in ["urgent", "cannot work", "cannot send", "broken", "failed", "error", "timeout", "high", "authentication error", "locked out"]):
        priority = "High"
    elif any(k in q for k in ["question", "how to", "policy", "guideline", "info", "low"]):
        priority = "Low"
    else:
        priority = "Medium"

    words = query.strip().split()
    title = " ".join(words[:8]) if len(words) > 8 else query.strip()

    logger.info(f"Triage fallback: '{query}' -> intent={intent}, category={category}, priority={priority}")
    return {
        "intent": intent,
        "category": category,
        "priority": priority,
        "title": title
    }


def classify_and_triage_node(state: AgentState) -> Dict[str, Any]:
    user_query = state.get("user_query", "").strip()
    configured = is_openai_configured()

    if configured:
        try:
            llm = get_chat_llm(temperature=0.0)
            if llm:
                response = llm.invoke([
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ])
                raw_text = response.content if hasattr(response, "content") else str(response)
                clean_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(clean_text or "{}")
                return {
                    "intent": parsed.get("intent", "information_retrieval"),
                    "category": parsed.get("category", "Software"),
                    "priority": parsed.get("priority", "Medium"),
                    "title": parsed.get("title", user_query[:50])
                }
        except Exception as e:
            logger.warning(f"OpenAI triage failed ({e}). Using heuristic.")

    return _heuristic_triage(user_query)


def retrieve_knowledge_node(state: AgentState) -> Dict[str, Any]:
    user_query = state.get("user_query", "").strip()

    try:
        vs = USMVectorStore()
        chunks = vs.similarity_search(query=user_query, top_k=3)

        sources = []
        seen = set()
        for c in chunks:
            key = (c.get("source"), c.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": c.get("source", "Unknown"),
                    "page": c.get("page", 1),
                    "similarity": c.get("similarity_score", 0.0)
                })

        return {
            "retrieved_chunks": chunks,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        return {
            "retrieved_chunks": [],
            "sources": [],
            "error": f"Vector retrieval error: {str(e)}"
        }


def generate_resolution_node(state: AgentState) -> Dict[str, Any]:
    user_query = state.get("user_query", "").strip()
    chunks = state.get("retrieved_chunks", [])
    category = state.get("category", "General")
    priority = state.get("priority", "Medium")

    rag_out = generate_rag_answer(query=user_query, retrieved_chunks=chunks)
    answer = rag_out["answer"]

    if "not contain" in answer.lower() or not chunks:
        answer += (
            f"\n\n---\n**Need to escalate?**  \n"
            f"If these steps do not resolve your problem, you can type **'Please create a ticket'** or click **Report New Incident** "
            f"to log this directly under Category **{category}** ({priority} Priority)."
        )

    return {
        "ai_response": answer,
        "action_taken": "retrieved_knowledge"
    }


def create_ticket_node(state: AgentState) -> Dict[str, Any]:
    title = state.get("title") or state.get("user_query", "IT Incident")[:60]
    requires_approval = state.get("requires_approval", False)
    is_approved = state.get("is_approved", False)

    if requires_approval and not is_approved:
        draft = {
            "employee_name": state.get("employee_name", "AI Assistant User"),
            "employee_email": state.get("employee_email", "employee@company.com"),
            "department": state.get("department", "Engineering"),
            "category": state.get("category", "Software"),
            "priority": state.get("priority", "Medium"),
            "title": title,
            "description": state.get("user_query", "Logged via AI Agent"),
            "status": "Pending Human Approval",
            "assigned_technician": "Unassigned"
        }
        draft_msg = (
            f"**Drafted Incident Ticket (Approval Required)**\n\n"
            f"- **Proposed Title:** {draft['title']}\n"
            f"- **Classified Category:** {draft['category']}\n"
            f"- **Recommended Priority:** {draft['priority']}\n"
            f"- **Requester:** {draft['employee_name']} ({draft['employee_email']})\n"
            f"- **Department:** {draft['department']}\n\n"
            f"**Approval Gate:** Please review the proposed details above. "
            f"Confirm approval to write this record to the corporate database."
        )
        return {
            "draft_ticket": draft,
            "ai_response": draft_msg,
            "action_taken": "awaiting_human_approval",
            "requires_approval": True,
            "is_approved": False
        }

    db = next(get_db())
    try:
        payload = {
            "employee_name": state.get("employee_name", "AI Assistant User"),
            "employee_email": state.get("employee_email", "employee@company.com"),
            "department": state.get("department", "Engineering"),
            "category": state.get("category", "Software"),
            "priority": state.get("priority", "Medium"),
            "ticket_type": "Incident",
            "title": title,
            "description": state.get("user_query", "Logged via AI Agent"),
            "status": "Open",
            "assigned_technician": "Unassigned"
        }
        ticket, err = create_ticket(db, payload)
        if err:
            logger.error(f"Database insertion failed: {err}")
            return {
                "ai_response": f"Failed to create ticket: {err}",
                "action_taken": "ticket_creation_failed",
                "error": err
            }

        sla_str = ticket.sla_due_date.strftime("%Y-%m-%d %H:%M UTC") if ticket.sla_due_date else "Standard"
        response_msg = (
            f"**Support Ticket Logged: `{ticket.ticket_id}`**\n\n"
            f"- **Title:** {ticket.title}\n"
            f"- **Category:** {ticket.category}\n"
            f"- **Priority:** {ticket.priority}\n"
            f"- **SLA Target:** {sla_str}\n"
            f"- **Status:** {ticket.status}\n\n"
            f"An IT specialist in the **{ticket.category} Queue** has been notified. You can track progress in the Incidents view."
        )

        return {
            "ticket_created": ticket.to_dict(),
            "ai_response": response_msg,
            "action_taken": "ticket_created",
            "is_approved": True
        }
    finally:
        db.close()


def handle_general_chat_node(state: AgentState) -> Dict[str, Any]:
    user_query = state.get("user_query", "").strip()
    employee_name = state.get("employee_name", "there")
    configured = is_openai_configured()

    if configured:
        try:
            llm = get_chat_llm(temperature=0.3)
            if llm:
                response = llm.invoke([
                    {
                        "role": "system",
                        "content": (
                            "You are the AI Assistant for the Unified Service Management (USM) System. "
                            "Respond concisely and professionally. Explain capabilities such as answering IT SOP questions, "
                            "retrieving knowledge base documents, and drafting or creating incident tickets if asked."
                        )
                    },
                    {"role": "user", "content": user_query}
                ])
                chat_answer = response.content if hasattr(response, "content") else str(response)
                return {
                    "ai_response": chat_answer,
                    "action_taken": "general_chat"
                }
        except Exception as e:
            logger.warning(f"ChatOpenAI call failed ({e}). Using offline greeting.")

    q_low = user_query.lower()
    if any(k in q_low for k in ["who are you", "what can you do", "introduce yourself", "how do you work", "capabilities"]):
        message = (
            f"**Hello {employee_name}. I am the IT Service Management Assistant.**\n\n"
            f"Capabilities:\n"
            f"- **Answer IT questions:** Ask about VPN configuration, password lockout policies, Wi-Fi, or software guidelines.\n"
            f"- **Log incident tickets:** Report technical issues or equipment malfunctions (e.g., 'Please create a ticket for my broken monitor').\n"
            f"- **Intelligent triage:** Categorizes requests and assigns severity.\n\n"
            f"How can I help you today?"
        )
    elif any(k in q_low for k in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"]):
        message = (
            f"**Hello {employee_name}. Welcome to the IT Service Desk.**\n\n"
            f"You can ask about IT policies, get troubleshooting guidance, or report an incident ticket."
        )
    else:
        message = (
            f"**Hello {employee_name}.**\n\n"
            f"You can ask technical questions (such as VPN setup, password policies, or email troubleshooting) "
            f"or request to log an IT support ticket."
        )

    return {
        "ai_response": message,
        "action_taken": "general_chat"
    }
