"""
LangGraph Workflow Nodes Module for IT Service Management.

Implements the operational nodes for:
1. Intent classification & triage (Category + Priority)
2. ChromaDB RAG Knowledge retrieval
3. Grounded resolution generation with source citations
4. Automated SQLite incident ticket creation
5. Contextual conversational guidance & greeting handling
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
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

# Dedicated logger for Agent workflows
logger = logging.getLogger("USM_AGENT")
logger.setLevel(logging.INFO)


def get_chat_llm(temperature: float = 0.0) -> Optional[ChatOpenAI]:
    """
    Initializes a LangChain ChatOpenAI instance using the dynamically loaded API key.
    Never prints or logs the raw secret.
    """
    key = get_openai_api_key()
    if key and key.startswith("sk-") and len(key) > 20:
        return ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=temperature,
            api_key=key
        )
    return None


# ==============================================================================
# NODE 1: CLASSIFY AND TRIAGE
# ==============================================================================
TRIAGE_SYSTEM_PROMPT = """You are an expert ITIL Service Desk Triage AI.
Analyze the employee's request and classify it accurately into JSON format.

JSON Schema:
{
    "intent": "information_retrieval" | "incident_creation" | "general_chat",
    "category": "Hardware" | "Software" | "Network" | "Email" | "Access",
    "priority": "Low" | "Medium" | "High" | "Critical",
    "title": "Clear 5-10 word summary of the incident or request"
}

Classification Guidelines:
- intent:
    - "incident_creation": when the user asks to "log a ticket", "create an incident", reports a system outage, severe error, or broken equipment requiring technician dispatch.
    - "information_retrieval": when the user asks how to do something, asks for troubleshooting guidance, questions IT policies, passwords, VPN setup, email authentication, or software configuration.
    - "general_chat": ONLY for greetings (hello, hi), pleasantries, or questions asking about who you are or what you can do without mentioning technical IT problems.
- category:
    - "Network": VPN, Wi-Fi, DNS, gateway, internet connection timeouts.
    - "Hardware": Laptop, monitor, keyboard, docking station, broken physical components.
    - "Software": Application crashes, Docker, development tools, licenses.
    - "Email": Outlook, Exchange, inbox sync, distribution lists, email authentication.
    - "Access": Passwords, account lockout, Okta, SSO, permissions, MFA tokens, logins.
- priority:
    - "Critical": Complete team/business outage, production failure, security breach.
    - "High": Major blocker impacting employee's core workday without workaround (e.g., cannot send/receive emails, account locked).
    - "Medium": Standard technical issue with partial workaround available.
    - "Low": Minor questions, peripheral requests, cosmetic issues.

Return ONLY valid JSON.
"""


def _heuristic_triage(query: str) -> Dict[str, Any]:
    """
    High-accuracy deterministic fallback classifier when no OpenAI API key is present.
    """
    q = query.lower().strip()

    # 1. Technical IT Indicators & Categories
    it_indicators = [
        "vpn", "wifi", "wi-fi", "network", "dns", "gateway", "firewall", "latency", "ping", "internet", "connection",
        "laptop", "macbook", "monitor", "keyboard", "mouse", "usb", "dock", "hardware", "screen", "cable", "display", "charger",
        "email", "outlook", "exchange", "inbox", "mail", "calendar", "invite", "mailbox", "send or receive",
        "password", "lockout", "locked", "sso", "okta", "mfa", "login", "access", "permission", "credentials", "2fa", "authenticate", "authentication",
        "software", "app", "application", "crash", "error", "bug", "docker", "git", "install", "update", "license",
        "troubleshoot", "troubleshooting", "setup", "configure", "how to", "policy", "guideline", "steps", "requirement"
    ]
    has_it_indicator = any(k in q for k in it_indicators)

    # 2. Intent Detection
    ticket_triggers = [
        "create ticket", "create a ticket", "log ticket", "log a ticket",
        "open ticket", "open a ticket", "report incident", "submit ticket",
        "file a ticket", "please create", "please log", "raise a ticket",
        "raise ticket", "report an incident", "submit a ticket", "open an incident"
    ]

    is_pure_greeting_or_capability = (
        q in ["help", "help me", "can you help me", "i need help", "hello", "hi", "hey"]
        or any(phrase in q for phrase in ["who are you", "what can you do", "introduce yourself", "how do you work", "tell me about yourself"])
        or (any(g in q for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "greetings"]) and len(q.split()) <= 4 and not has_it_indicator)
    )

    if any(k in q for k in ticket_triggers):
        intent = "incident_creation"
    elif is_pure_greeting_or_capability and not has_it_indicator:
        intent = "general_chat"
    else:
        # Technical inquiries, troubleshooting requests, policy questions
        intent = "information_retrieval"

    # 3. Category Classification
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

    # 4. Priority Determination
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

    logger.info(f"[TRIAGE_HEURISTIC] Query: '{query}' -> Intent={intent}, Category={category}, Priority={priority}")
    return {
        "intent": intent,
        "category": category,
        "priority": priority,
        "title": title
    }


def classify_and_triage_node(state: AgentState) -> Dict[str, Any]:
    """
    Workflow Node 1: Analyzes the employee prompt and determines intent, category, and priority.
    """
    user_query = state.get("user_query", "").strip()
    configured = is_openai_configured()
    masked = get_masked_api_key()
    logger.info(f"[CLASSIFY_NODE] Processing incoming query: '{user_query}' | OpenAI Configured={configured} ({masked})")

    if configured:
        try:
            logger.info(f"[CLASSIFY_NODE] Executing LangGraph triage via model '{LLM_MODEL_NAME}'")
            llm = get_chat_llm(temperature=0.0)
            if llm:
                response = llm.invoke([
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ])
                raw_text = response.content if hasattr(response, "content") else str(response)
                clean_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(clean_text or "{}")
                triage_res = {
                    "intent": parsed.get("intent", "information_retrieval"),
                    "category": parsed.get("category", "Software"),
                    "priority": parsed.get("priority", "Medium"),
                    "title": parsed.get("title", user_query[:50])
                }
                logger.info(f"[CLASSIFY_NODE] OpenAI classification: {triage_res}")
                return triage_res
        except Exception as e:
            logger.warning(f"[CLASSIFY_NODE] OpenAI triage failed ({type(e).__name__}: {str(e)}). Falling back to deterministic heuristic.")

    # Fallback heuristic
    return _heuristic_triage(user_query)


# ==============================================================================
# NODE 2: RETRIEVE KNOWLEDGE
# ==============================================================================
def retrieve_knowledge_node(state: AgentState) -> Dict[str, Any]:
    """
    Workflow Node 2: Searches ChromaDB for relevant SOPs and documentation chunks.
    """
    user_query = state.get("user_query", "").strip()
    logger.info(f"[RETRIEVE_NODE] Initiating ChromaDB vector similarity search for: '{user_query}'")

    try:
        vs = USMVectorStore()
        chunks = vs.similarity_search(query=user_query, top_k=3)
        logger.info(f"[RETRIEVE_NODE] ChromaDB search returned {len(chunks)} relevant chunks.")

        sources = []
        seen = set()
        for idx, c in enumerate(chunks, 1):
            key = (c.get("source"), c.get("page"))
            preview = c.get("text", "").replace("\n", " ")[:90]
            logger.info(f"[RETRIEVE_NODE] Chunk {idx}: Document={c.get('source')}, Page={c.get('page')}, Score={c.get('similarity_score', 0.0):.4f}, Preview='{preview}...'")
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
        logger.error(f"[RETRIEVE_NODE] Vector retrieval failed: {e}", exc_info=True)
        return {
            "retrieved_chunks": [],
            "sources": [],
            "error": f"Vector retrieval error: {str(e)}"
        }


# ==============================================================================
# NODE 3: GENERATE RESOLUTION
# ==============================================================================
def generate_resolution_node(state: AgentState) -> Dict[str, Any]:
    """
    Workflow Node 3: Synthesizes a factual resolution from retrieved documentation.
    Offers automatic ticket creation if documentation is unavailable.
    """
    user_query = state.get("user_query", "").strip()
    chunks = state.get("retrieved_chunks", [])
    category = state.get("category", "General")
    priority = state.get("priority", "Medium")
    logger.info(f"[RESOLUTION_NODE] Synthesizing resolution for Category='{category}', Priority='{priority}' with {len(chunks)} context passages.")

    rag_out = generate_rag_answer(query=user_query, retrieved_chunks=chunks)
    answer = rag_out["answer"]

    # Append actionable self-service guidance if answer indicates missing info or escalation option
    if "not contain" in answer.lower() or not chunks:
        answer += (
            f"\n\n---\n💡 **Need to escalate?**  \n"
            f"If these steps do not resolve your problem, you can type **'Please create a ticket'** or click **Report New Incident** "
            f"to log this directly under Category **{category}** ({priority} Priority)."
        )

    logger.info(f"[RESOLUTION_NODE] Generated resolution successfully (length: {len(answer)} chars).")
    return {
        "ai_response": answer,
        "action_taken": "retrieved_knowledge"
    }


# ==============================================================================
# NODE 4: CREATE TICKET (WITH HUMAN-IN-THE-LOOP APPROVAL)
# ==============================================================================
def create_ticket_node(state: AgentState) -> Dict[str, Any]:
    """
    Workflow Node 4: Creates a persistent incident ticket in SQLite, supporting
    Human-in-the-Loop (HITL) approval gates before database commitment.
    """
    title = state.get("title") or state.get("user_query", "IT Incident")[:60]
    requires_approval = state.get("requires_approval", False)
    is_approved = state.get("is_approved", False)
    logger.info(f"[TICKET_NODE] Invoked: requires_approval={requires_approval}, is_approved={is_approved}, Title='{title}'")

    # --- Human Approval Gate ---
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
            f"📋 **Drafted Incident Ticket (Human Approval Required)**\n\n"
            f"- **Proposed Title:** {draft['title']}\n"
            f"- **Classified Category:** {draft['category']}\n"
            f"- **Recommended Priority:** {draft['priority']}\n"
            f"- **Requester:** {draft['employee_name']} ({draft['employee_email']})\n"
            f"- **Department:** {draft['department']}\n\n"
            f"⚠️ **Human Approval Gate:** Please review the proposed details above. "
            f"Confirm approval to write this record to the corporate database."
        )
        logger.info(f"[TICKET_NODE] Draft ticket prepared. Awaiting human confirmation.")
        return {
            "draft_ticket": draft,
            "ai_response": draft_msg,
            "action_taken": "awaiting_human_approval",
            "requires_approval": True,
            "is_approved": False
        }

    # --- Approved or Direct Ticket Creation ---
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
            logger.error(f"[TICKET_NODE] Database insertion failed: {err}")
            return {
                "ai_response": f"❌ Failed to automatically create ticket: {err}",
                "action_taken": "ticket_creation_failed",
                "error": err
            }

        sla_str = ticket.sla_due_date.strftime("%Y-%m-%d %H:%M UTC") if ticket.sla_due_date else "Standard"
        response_msg = (
            f"✅ **Support Ticket Logged: `{ticket.ticket_id}`**\n\n"
            f"- **Title:** {ticket.title}\n"
            f"- **Category:** {ticket.category}\n"
            f"- **Priority:** {ticket.priority}\n"
            f"- **SLA Target:** {sla_str}\n"
            f"- **Status:** {ticket.status}\n\n"
            f"An IT specialist in the **{ticket.category} Queue** has been notified. You can track progress and add comments directly in the **Incidents** view."
        )

        logger.info(f"[TICKET_NODE] Ticket created in SQLite: Ticket ID='{ticket.ticket_id}'")
        return {
            "ticket_created": ticket.to_dict(),
            "ai_response": response_msg,
            "action_taken": "ticket_created",
            "is_approved": True
        }
    finally:
        db.close()


# ==============================================================================
# NODE 5: GENERAL CHAT
# ==============================================================================
def handle_general_chat_node(state: AgentState) -> Dict[str, Any]:
    """
    Workflow Node 5: Provides welcoming guidance or intelligent conversational answers
    for greetings and general non-technical interactions.
    """
    user_query = state.get("user_query", "").strip()
    employee_name = state.get("employee_name", "there")
    api_key = get_openai_api_key()
    configured = is_openai_configured()
    logger.info(f"[GENERAL_CHAT_NODE] Handling general conversation for user '{employee_name}': '{user_query}' | OpenAI Configured={configured}")

    # If OpenAI API is configured, generate a conversational response
    if configured:
        try:
            logger.info(f"[GENERAL_CHAT_NODE] Invoking ChatOpenAI for general dialogue.")
            llm = get_chat_llm(temperature=0.3)
            if llm:
                response = llm.invoke([
                    {
                        "role": "system",
                        "content": (
                            "You are the AI Assistant for the Unified Service Management (USM) System. "
                            "Respond warmly, concisely, and professionally. Explain your capabilities (answering IT SOP questions, "
                            "retrieving knowledge base documents, and automatically drafting or creating incident tickets) if asked."
                        )
                    },
                    {"role": "user", "content": user_query}
                ])
                chat_answer = response.content if hasattr(response, "content") else str(response)
                logger.info(f"[GENERAL_CHAT_NODE] ChatOpenAI conversational response generated successfully.")
                return {
                    "ai_response": chat_answer,
                    "action_taken": "general_chat"
                }
        except Exception as e:
            logger.warning(f"[GENERAL_CHAT_NODE] ChatOpenAI call failed ({type(e).__name__}: {str(e)}). Falling back to dynamic offline greeting.")

    # Dynamic contextual offline fallback
    q_low = user_query.lower()
    if any(k in q_low for k in ["who are you", "what can you do", "introduce yourself", "how do you work", "capabilities"]):
        message = (
            f"👋 **Hello {employee_name}! I am your AI-Powered Unified Service Management (USM) Copilot.**\n\n"
            f"Here is what I can do for you:\n"
            f"- 🔍 **Answer IT Questions:** Ask about VPN configuration, password lockout policies, Wi-Fi, or software guidelines.\n"
            f"- 🚨 **Log Incident Tickets:** Tell me about technical issues or equipment malfunctions (e.g., *'Please create a ticket for my broken monitor'*).\n"
            f"- 🏷️ **Intelligent Triage:** I automatically categorize your issue (Hardware, Software, Network, Email, Access) and recommend appropriate urgency.\n\n"
            f"How can I assist you with your IT needs today?"
        )
    elif any(k in q_low for k in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"]):
        message = (
            f"👋 **Hello {employee_name}! Welcome to the USM Service Desk.**\n\n"
            f"I'm here to help you resolve technical issues, lookup IT documentation, or create support tickets. "
            f"Feel free to describe what you need assistance with!"
        )
    else:
        message = (
            f"👋 **Hello {employee_name}!**\n\n"
            f"I am ready to help. You can ask me technical questions (such as VPN setup, password policies, or email troubleshooting) "
            f"or request to log an IT support ticket."
        )

    return {
        "ai_response": message,
        "action_taken": "general_chat"
    }
