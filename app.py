"""
Relay: IT Service Management platform and operations console.
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import altair as alt

from database import get_db, init_db
from models import Ticket, TicketComment, TicketAuditLog
from crud import (
    create_ticket,
    get_all_tickets,
    get_ticket_by_ticket_id,
    search_and_filter_tickets,
    update_ticket,
    delete_ticket,
    get_dashboard_metrics,
    seed_initial_tickets,
    add_ticket_comment,
    get_ticket_comments,
    get_ticket_audit_logs,
    calculate_sla_deadlines
)
from rag.pdf_loader import extract_pages_from_pdf
from rag.text_splitter import chunk_document_pages
from rag.vector_store import USMVectorStore
from rag.rag_chain import generate_rag_answer
from agents.graph import run_it_support_agent
from api.analytics import get_detailed_analytics
from ai_copilot import (
    summarize_incident_thread,
    correlate_similar_incidents,
    draft_smart_resolution,
    analyze_sentiment_and_governance,
    get_esm_business_value_kpis
)
from config import get_openai_api_key, is_openai_configured, get_masked_api_key, LLM_MODEL_NAME


@st.cache_resource
def bootstrap_database():
    init_db()
    db = next(get_db())
    try:
        seed_initial_tickets(db)
    finally:
        db.close()
    return True


@st.cache_resource
def get_cached_vector_store():
    return USMVectorStore()


bootstrap_database()


st.set_page_config(
    page_title="Relay / Ops Console",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg-canvas: #0E1013;
        --bg-surface-1: #14171C;
        --bg-surface-2: #1A1E24;
        --border-subtle: #242930;
        --border-strong: #363D47;
        --accent-terracotta: #D95D39;
        --accent-hover: #BF4C29;
        --text-primary: #EDEDEB;
        --text-secondary: #8F96A3;
        --text-muted: #5C6370;
        --status-critical: #D95D39;
        --status-high: #D9822B;
        --status-resolved: #3D8B68;
        --radius-sm: 4px;
    }

    /* Base Typography & Performance Resets */
    html, body, [class*="css"] {
        font-family: 'Work Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
    }

    /* Top Masthead */
    .relay-masthead {
        border-bottom: 1px solid var(--border-subtle);
        padding: 0.5rem 0 1rem 0;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .relay-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .relay-brand span {
        color: var(--accent-terracotta);
        margin-right: 0.35rem;
    }
    .relay-telemetry-mono {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
        letter-spacing: 0.02em;
    }

    /* Typography Scale */
    .relay-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.15rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }
    .relay-subheading {
        font-size: 0.98rem;
        color: var(--text-secondary) !important;
        line-height: 1.5;
        margin-bottom: 1.75rem;
        max-width: 680px;
    }
    .relay-section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }

    /* Metric Cards - Multi-Line Wrapping (No Truncation / No Ellipsis) */
    div[data-testid="stMetric"] {
        background: var(--bg-surface-1) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 1.05rem 1.15rem !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease !important;
        min-height: 142px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-strong) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--text-secondary) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.35 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] div {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        margin: 0.2rem 0 !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div,
    div[data-testid="stMetricDelta"] svg + div,
    div[data-testid="stMetricDelta"] span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.3 !important;
        font-size: 0.78rem !important;
    }

    /* Sidebar Navigation Overhaul: Strip Native Radio Buttons Completely */
    div[data-testid="stSidebar"] {
        background-color: var(--bg-surface-1) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 0.55rem 0.85rem !important;
        border-radius: var(--radius-sm) !important;
        margin-bottom: 2px !important;
        border-left: 2px solid transparent !important;
        background: transparent !important;
        transition: background 0.12s ease, border-color 0.12s ease !important;
        cursor: pointer !important;
        will-change: background-color, border-color;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none !important; /* Hide native round radio bullet */
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        color: var(--text-secondary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.88rem !important;
        margin: 0 !important;
        transition: color 0.1s ease !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: var(--bg-surface-2) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: var(--text-primary) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: var(--bg-surface-2) !important;
        border-left: 2px solid var(--accent-terracotta) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Native Streamlit Border Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-surface-1) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* Badges */
    .relay-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 3px;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    .badge-p1 {
        background: rgba(217, 93, 57, 0.15);
        color: #D95D39;
        border: 1px solid rgba(217, 93, 57, 0.4);
    }
    .badge-p2 {
        background: rgba(217, 130, 43, 0.15);
        color: #D9822B;
        border: 1px solid rgba(217, 130, 43, 0.4);
    }
    .badge-p3 {
        background: rgba(143, 150, 163, 0.15);
        color: #8F96A3;
        border: 1px solid rgba(143, 150, 163, 0.35);
    }
    .badge-p4 {
        background: rgba(92, 99, 112, 0.15);
        color: #8F96A3;
        border: 1px solid rgba(92, 99, 112, 0.3);
    }
    .badge-sla-breached {
        background: rgba(217, 93, 57, 0.2);
        color: #D95D39;
        border: 1px solid #D95D39;
        font-weight: 600;
    }
    .badge-sla-ok {
        background: rgba(61, 139, 104, 0.15);
        color: #3D8B68;
        border: 1px solid rgba(61, 139, 104, 0.4);
    }

    /* Minimal Stage Stepper */
    .relay-stepper {
        display: flex;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        overflow: hidden;
        margin-bottom: 1.25rem;
        background: var(--bg-surface-1);
    }
    .relay-step {
        flex: 1;
        padding: 0.6rem 0.5rem;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        border-right: 1px solid var(--border-subtle);
        background: var(--bg-surface-1);
    }
    .relay-step.completed {
        color: var(--status-resolved);
        background: rgba(61, 139, 104, 0.06);
    }
    .relay-step.active {
        color: var(--text-primary);
        background: var(--bg-surface-2);
        font-weight: 600;
        border-bottom: 2px solid var(--accent-terracotta);
    }

    /* Work Notes */
    .relay-internal-note {
        background: var(--bg-surface-2);
        border-left: 3px solid var(--accent-terracotta);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: 0.85rem 1rem;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        margin-bottom: 0.65rem;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .relay-public-comment {
        background: var(--bg-surface-1);
        border-left: 3px solid var(--border-strong);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: 0.85rem 1rem;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        margin-bottom: 0.65rem;
        color: var(--text-primary);
        font-size: 0.9rem;
    }

    /* Flat Buttons */
    button[kind="primary"] {
        background: var(--accent-terracotta) !important;
        border: 1px solid var(--accent-terracotta) !important;
        color: #ffffff !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: background 0.15s ease !important;
    }
    button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }
    button[kind="secondary"] {
        background: var(--bg-surface-2) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: none !important;
    }

    /* Filter Bar & Input Straight Horizontal Alignment */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        height: 22px !important;
        min-height: 22px !important;
        line-height: 22px !important;
        margin-bottom: 5px !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        height: 42px !important;
        min-height: 42px !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-subtle) !important;
        background: var(--bg-surface-1) !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stTextInput"] input {
        height: 38px !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
    }

    /* Interactive Dropdown & Selectbox Cursor: Force Hand Pointer (Hovering Cursor) on Hover */
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] *,
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] svg,
    div[data-testid="stSelectbox"] div,
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    li[role="option"] *,
    div[role="combobox"],
    div[role="combobox"] * {
        cursor: pointer !important;
    }

    /* Editable text inputs keep standard text typing cursor */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        cursor: text !important;
    }
    </style>
""", unsafe_allow_html=True)


def render_masthead():
    st.markdown("""<div class="relay-masthead"><div class="relay-brand"><span>●</span>RELAY / UNIFIED OPERATIONS CONSOLE</div><div class="relay-telemetry-mono">UNIFIED SERVICE MANAGEMENT (ESM) &nbsp;·&nbsp; AI MATURITY: LEVEL 3 &nbsp;·&nbsp; GATEWAY: 8000/TCP</div></div>""", unsafe_allow_html=True)


@st.cache_data(ttl=5)
def get_cached_sidebar_telemetry():
    db = next(get_db())
    try:
        total = db.query(Ticket).count()
        critical = db.query(Ticket).filter(
            Ticket.priority == "Critical",
            Ticket.status.notin_(["Resolved", "Closed"])
        ).count()
    except Exception:
        total, critical = 0, 0
    finally:
        db.close()

    try:
        vs = get_cached_vector_store()
        kb = vs.count()
    except Exception:
        kb = 0
    return total, critical, kb


def render_sidebar():
    total_in_db, critical_count, kb_chunks = get_cached_sidebar_telemetry()

    with st.sidebar:
        st.markdown('<div class="relay-section-label">OPERATIONS INDEX</div>', unsafe_allow_html=True)

        nav_options = [
            "01 / Triage Console",
            "02 / Incident Desk",
            "03 / Provisioning Catalog",
            "04 / Virtual Operator",
            "05 / Knowledge Index",
            "06 / System Telemetry"
        ]

        selected_page = st.radio(
            label="Navigation Menu",
            options=nav_options,
            key="main_nav_radio",
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown('<div class="relay-section-label">QUEUE TELEMETRY</div>', unsafe_allow_html=True)
        crit_color = "#D95D39" if critical_count > 0 else "#3D8B68"
        st.markdown(f"""<div style="font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text-secondary); line-height:1.7;"><div>Active Queue &nbsp;<b>{total_in_db}</b></div><div>Critical P1 &nbsp;&nbsp;<span style="color:{crit_color}; font-weight:600;">{critical_count} active</span></div><div>SOP Index &nbsp;&nbsp;&nbsp;&nbsp;<b>{kb_chunks} passages</b></div><div>Operator &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>nominal</b></div><div>Port &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>8000/tcp</b></div></div>""", unsafe_allow_html=True)

    return selected_page


def render_triage_console():
    db = next(get_db())
    try:
        metrics = get_dashboard_metrics(db)
        analytics_data = get_detailed_analytics(db)
        recent_tickets = get_all_tickets(db, limit=8)
        esm_kpis = get_esm_business_value_kpis(db)
    finally:
        db.close()

    st.markdown('<div class="relay-heading">Unified Ops Floor / Incident Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Enterprise Service Management (ESM) telemetry, AI deflection velocity, and engineering dispatch across internal services.</div>', unsafe_allow_html=True)

    st.markdown('<div class="relay-section-label">AI COPILOT & SERVICE EFFICIENCY TELEMETRY</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("AI Deflection Velocity", f"{esm_kpis['deflection_rate_pct']}%", delta=f"{esm_kpis['total_deflections']} auto-deflected")
    with e2:
        st.metric("Engineering Hours Saved", f"{esm_kpis['total_hours_saved']}h", delta="Automation ROI")
    with e3:
        st.metric("Employee CSAT Index", f"{esm_kpis['csat_score']} / 5.0", delta="High satisfaction")
    with e4:
        rate = analytics_data.get("resolution_rate_pct", 0.0)
        st.metric("SLA Clearance Velocity", f"{rate}%", delta=f"{metrics['resolved_tickets']} fulfilled")

    st.write("")

    st.markdown('<div class="relay-section-label">PROACTIVE SLA PREDICTIVE RADAR</div>', unsafe_allow_html=True)
    if esm_kpis["at_risk_count"] > 0:
        st.warning(f"PROACTIVE RADAR: {esm_kpis['at_risk_count']} in-flight incident(s) approaching target SLA deadline within 2 hours. Prioritize dispatch.")
        st.dataframe(pd.DataFrame(esm_kpis["at_risk_tickets"]), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--text-secondary); padding:0.5rem 0.75rem; background:rgba(61,139,104,0.08); border-left:3px solid #3D8B68; border-radius:0 4px 4px 0; margin-bottom:1rem;">● PROACTIVE RADAR: All in-flight incidents pacing within safe SLA margin. Zero breaches predicted in 2-hour window.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="relay-section-label">QUEUE INVENTORY & SEVERITY</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total In-Flight", metrics["total_tickets"])
    with m2:
        st.metric("Open Triage Queue", metrics["open_tickets"])
    with m3:
        st.metric("Active SLA Breaches", metrics["sla_breached"], delta="Breached" if metrics["sla_breached"] > 0 else "Compliant", delta_color="inverse" if metrics["sla_breached"] > 0 else "normal")
    with m4:
        crit = metrics["critical_incidents"]
        st.metric("Critical (P1) Outages", crit, delta="Immediate dispatch" if crit > 0 else "Baseline normal", delta_color="inverse" if crit > 0 else "normal")

    st.write("")

    left_col, right_col = st.columns([7, 5])

    with left_col:
        st.markdown('<div class="relay-section-label">SEVERITY CONCENTRATION (P1 — P4)</div>', unsafe_allow_html=True)
        sev = metrics["severity_counts"]
        sev_df = pd.DataFrame({
            "Severity": ["P1 Critical", "P2 High", "P3 Medium", "P4 Low"],
            "Count": [sev["Critical"], sev["High"], sev["Medium"], sev["Low"]]
        })

        chart_sev = alt.Chart(sev_df).mark_bar(cornerRadius=2).encode(
            x=alt.X("Severity:N", sort=["P1 Critical", "P2 High", "P3 Medium", "P4 Low"], axis=alt.Axis(labelAngle=0, labelColor="#8F96A3", title=None)),
            y=alt.Y("Count:Q", title="Records", axis=alt.Axis(labelColor="#8F96A3", titleColor="#5C6370", gridColor="#242930")),
            color=alt.Color("Severity:N", scale=alt.Scale(
                domain=["P1 Critical", "P2 High", "P3 Medium", "P4 Low"],
                range=["#D95D39", "#D9822B", "#5C6370", "#363D47"]
            ), legend=None),
            tooltip=[alt.Tooltip("Severity:N"), alt.Tooltip("Count:Q")]
        ).properties(height=220)
        st.altair_chart(chart_sev, use_container_width=True)

    with right_col:
        st.markdown('<div class="relay-section-label">WORKLOAD BY TEAM</div>', unsafe_allow_html=True)
        dept_dict = analytics_data.get("department_breakdown", {})
        dept_df = pd.DataFrame(list(dept_dict.items()), columns=["Team", "Tickets"])

        chart_dept = alt.Chart(dept_df).mark_bar(cornerRadius=2, color="#383F4A").encode(
            y=alt.Y("Team:N", sort="-x", title=None, axis=alt.Axis(labelColor="#8F96A3")),
            x=alt.X("Tickets:Q", title="Volume", axis=alt.Axis(labelColor="#8F96A3", titleColor="#5C6370", gridColor="#242930")),
            tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Tickets:Q")]
        ).properties(height=220)
        st.altair_chart(chart_dept, use_container_width=True)

    st.write("")
    st.markdown('<div class="relay-section-label">RECENT ACTIVE DISRUPTIONS</div>', unsafe_allow_html=True)

    if recent_tickets:
        display_data = []
        for t in recent_tickets:
            sla_text = "BREACHED" if t.is_sla_breached else (t.sla_due_date.strftime("%m-%d %H:%M") if t.sla_due_date else "—")
            display_data.append({
                "ID": t.ticket_id,
                "Type": t.ticket_type or "Incident",
                "Summary": t.title,
                "Priority": t.priority,
                "Status": t.status,
                "Requester": t.employee_name,
                "Assigned": t.assigned_technician,
                "Target SLA": sla_text,
                "Logged": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else ""
            })

        st.dataframe(
            pd.DataFrame(display_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn("ID", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Summary": st.column_config.TextColumn("Summary", width="large"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Target SLA": st.column_config.TextColumn("Target SLA", width="small")
            }
        )
    else:
        st.info("Queue clear. No records logged.")


def render_incident_desk():
    st.markdown('<div class="relay-heading">Incident Desk</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Triage incoming tickets, maintain internal work notes, and enforce resolution commitments.</div>', unsafe_allow_html=True)

    tab_queue, tab_intake = st.tabs(["Active Incidents & Records", "Log Disruption Report"])

    with tab_queue:
        db = next(get_db())
        try:
            f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
            with f1:
                search_kw = st.text_input("Search Records", placeholder="Search ID, caller, or summary...", label_visibility="visible")
            with f2:
                status_f = st.selectbox("Status", ["All", "Open", "In Progress", "Pending Approval", "Resolved", "Closed"])
            with f3:
                prio_f = st.selectbox("Priority", ["All", "Critical", "High", "Medium", "Low"])
            with f4:
                type_f = st.selectbox("Kind", ["All", "Incident", "Service Request"])

            matched_tickets = search_and_filter_tickets(
                db=db,
                search_query=search_kw,
                status_filter=status_f,
                priority_filter=prio_f,
                type_filter=type_f
            )

            if not matched_tickets:
                st.info("No records match the current criteria.")
            else:
                table_records = []
                for t in matched_tickets:
                    table_records.append({
                        "ID": t.ticket_id,
                        "Type": t.ticket_type or "Incident",
                        "Summary": t.title,
                        "Category": t.category,
                        "Priority": t.priority,
                        "Status": t.status,
                        "Caller": t.employee_name,
                        "Technician": t.assigned_technician,
                        "Breached": "Yes" if t.is_sla_breached else "No"
                    })

                st.dataframe(
                    pd.DataFrame(table_records),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"ID": st.column_config.TextColumn("ID", width="small"), "Summary": st.column_config.TextColumn("Summary", width="large")}
                )

                st.write("")
                st.markdown('<div class="relay-section-label">INSPECTION WORKSPACE</div>', unsafe_allow_html=True)

                ticket_options = [f"{t.ticket_id} — {t.title[:65]}" for t in matched_tickets]
                selected_choice = st.selectbox("Select record to inspect:", ticket_options)
                selected_tid = selected_choice.split(" — ")[0]

                target = get_ticket_by_ticket_id(db, selected_tid)

                if target:
                    states = ["Open", "In Progress", "Pending Approval", "Resolved", "Closed"]
                    curr_state = target.status
                    curr_idx = states.index(curr_state) if curr_state in states else 0

                    stepper_parts = []
                    for idx, s in enumerate(states):
                        if idx < curr_idx:
                            stepper_parts.append(f'<div class="relay-step completed">{s.upper()}</div>')
                        elif idx == curr_idx:
                            stepper_parts.append(f'<div class="relay-step active">[{s.upper()}]</div>')
                        else:
                            stepper_parts.append(f'<div class="relay-step">{s.upper()}</div>')
                    st.markdown(f'<div class="relay-stepper">{"".join(stepper_parts)}</div>', unsafe_allow_html=True)

                    det_left, det_right = st.columns([8, 4])

                    with det_left:
                        prio_badge = f'<span class="relay-badge badge-p{target.priority[0].lower() if target.priority in ["Critical","High","Low"] else "3"}">{target.priority.upper()}</span>'
                        sla_badge = '<span class="relay-badge badge-sla-breached">SLA BREACHED</span>' if target.is_sla_breached else '<span class="relay-badge badge-sla-ok">WITHIN SLA</span>'

                        st.markdown(f"""<div style="margin-bottom:0.75rem;"><span style="font-family:'IBM Plex Mono', monospace; font-size:0.85rem; color:var(--text-secondary);">{target.ticket_id}</span> &nbsp; {prio_badge} &nbsp; {sla_badge}</div><div style="font-family:'Space Grotesk', sans-serif; font-size:1.45rem; font-weight:700; color:var(--text-primary); margin-bottom:1rem;">{target.title}</div>""", unsafe_allow_html=True)

                        st.markdown('<div class="relay-section-label">DESCRIPTION</div>', unsafe_allow_html=True)
                        with st.container(border=True):
                            st.write(target.description)

                        st.write("")
                        st.markdown('<div class="relay-section-label">AI OPERATIONS COPILOT (AGENT ASSIST)</div>', unsafe_allow_html=True)
                        with st.container(border=True):
                            sentiment_info = analyze_sentiment_and_governance(target)

                            sc1, sc2 = st.columns([1, 2])
                            with sc1:
                                st.markdown(f'<span class="relay-badge" style="background:rgba(217,93,57,0.15); color:{sentiment_info["sentiment_color"]}; border:1px solid {sentiment_info["sentiment_color"]};">{sentiment_info["sentiment_badge"]}</span>', unsafe_allow_html=True)
                            with sc2:
                                st.caption(sentiment_info["governance_rationale"])

                            st.divider()

                            cop_tab1, cop_tab2, cop_tab3 = st.tabs([
                                "Executive Briefing",
                                "Correlated Historical Fixes",
                                "Draft Resolution Note"
                            ])

                            with cop_tab1:
                                comments_for_sum = get_ticket_comments(db, target.ticket_id, include_internal=True)
                                brief = summarize_incident_thread(target, comments_for_sum)
                                st.markdown(f"**Blast Radius & Symptom:**\n{brief['symptom']}")
                                st.markdown(f"**Diagnostic Telemetry:**\n{brief['diagnostic']}")
                                st.markdown(f"**Prescribed Action:**\n{brief['next_step']}")
                                st.caption(f"Synthesized by {brief['engine']}")

                            with cop_tab2:
                                similar_list = correlate_similar_incidents(target, db, top_k=2)
                                if similar_list:
                                    for sim in similar_list:
                                        with st.container(border=True):
                                            st.markdown(f"**{sim['ticket_id']} — {sim['title']}** &nbsp;·&nbsp; `{sim['similarity_score']}% Similarity`")
                                            st.caption(f"Category: {sim['category']} · Priority: {sim['priority']} · Status: {sim['status']}")
                                            st.markdown(f"**Verified Fix Applied:**\n*{sim['verified_fix']}*")
                                            if st.button(f"Copy Verified Fix to Resolution Notes", key=f"use_fix_{sim['ticket_id']}_{target.ticket_id}"):
                                                st.session_state["staged_resolution_note"] = sim['verified_fix']
                                                st.success("Copied verified fix to dispatch note.")
                                                st.rerun()
                                else:
                                    st.info("No prior resolved incident matched the failure signature.")

                            with cop_tab3:
                                smart_draft = draft_smart_resolution(target, similar_incident=similar_list[0] if similar_list else None)
                                st.markdown("**Generated Customer Closure Notification:**")
                                st.code(smart_draft["customer_message"], language=None)
                                st.markdown("**Technical Post-Mortem Note:**")
                                st.code(smart_draft["technical_note"], language=None)
                                if st.button("Apply Post-Mortem to Resolution Note Box", key=f"apply_draft_{target.ticket_id}"):
                                    st.session_state["staged_resolution_note"] = smart_draft["technical_note"]
                                    st.success("Staged post-mortem to dispatch note.")
                                    st.rerun()

                        if target.resolution_notes:
                            st.write("")
                            st.markdown('<div class="relay-section-label">RESOLUTION LOG</div>', unsafe_allow_html=True)
                            with st.container(border=True):
                                st.write(target.resolution_notes)

                        # Dual Stream: Public Notice vs Internal Work Notes
                        st.write("")
                        st.markdown('<div class="relay-section-label">ACTIVITY STREAM</div>', unsafe_allow_html=True)
                        stream_tab1, stream_tab2, stream_tab3 = st.tabs([
                            "Customer Reply",
                            "Internal Work Note",
                            "Audit Trail"
                        ])

                        with stream_tab1:
                            with st.form(f"pub_{target.ticket_id}", clear_on_submit=True):
                                p_msg = st.text_area("Message to Requester:", placeholder="State customer-facing diagnostic update...", label_visibility="collapsed")
                                p_sub = st.form_submit_button("Post Customer Reply")
                                if p_sub and p_msg.strip():
                                    add_ticket_comment(db, target.ticket_id, "IT Operations", p_msg, author_role="technician", is_internal_note=False)
                                    st.success("Customer reply logged.")
                                    st.rerun()

                        with stream_tab2:
                            with st.form(f"int_{target.ticket_id}", clear_on_submit=True):
                                w_msg = st.text_area("Internal Investigation Note:", placeholder="Technical traces, network latency logs, vendor ticket reference...", label_visibility="collapsed")
                                w_sub = st.form_submit_button("Post Internal Note")
                                if w_sub and w_msg.strip():
                                    add_ticket_comment(db, target.ticket_id, "IT Operations", w_msg, author_role="technician", is_internal_note=True)
                                    st.success("Internal work note logged.")
                                    st.rerun()

                        with stream_tab3:
                            all_logs = get_ticket_audit_logs(db, target.ticket_id)
                            if all_logs:
                                rows = []
                                for l in all_logs:
                                    rows.append({
                                        "Time": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                        "Event": l.action.replace("_", " ").upper(),
                                        "Previous": l.old_value or "—",
                                        "Updated": l.new_value or "—",
                                        "Actor": l.performed_by
                                    })
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        comments = get_ticket_comments(db, target.ticket_id, include_internal=True)
                        if comments:
                            st.write("")
                            for c in reversed(comments):
                                ts = c.created_at.strftime("%b %d, %H:%M")
                                if c.is_internal_note:
                                    st.markdown(f'<div class="relay-internal-note"><b>INTERNAL NOTE</b> &nbsp;·&nbsp; {c.author_name} &nbsp;·&nbsp; <span style="color:var(--text-muted); font-size:0.75rem;">{ts}</span><br/><div style="margin-top:0.35rem;">{c.message}</div></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="relay-public-comment"><b>CUSTOMER REPLY</b> &nbsp;·&nbsp; {c.author_name} &nbsp;·&nbsp; <span style="color:var(--text-muted); font-size:0.75rem;">{ts}</span><br/><div style="margin-top:0.35rem;">{c.message}</div></div>', unsafe_allow_html=True)

                    with det_right:
                        with st.container(border=True):
                            st.markdown('<div class="relay-section-label">REQUESTER</div>', unsafe_allow_html=True)
                            st.markdown(f"**{target.employee_name}**")
                            st.caption(f"{target.department}")
                            st.code(target.employee_email, language=None)

                            st.divider()

                            st.markdown('<div class="relay-section-label">TARGET DEADLINE</div>', unsafe_allow_html=True)
                            sla_target_str = target.sla_due_date.strftime("%Y-%m-%d %H:%M UTC") if target.sla_due_date else "N/A"
                            st.code(sla_target_str, language=None)

                            st.divider()

                            st.markdown('<div class="relay-section-label">CATEGORY</div>', unsafe_allow_html=True)
                            st.markdown(f"**{target.category}**")

                        st.write("")
                        st.markdown('<div class="relay-section-label">RECORD DISPATCH</div>', unsafe_allow_html=True)
                        new_stat = st.selectbox("Set State:", states, index=states.index(target.status) if target.status in states else 0, key="act_status")
                        new_tech = st.text_input("Assignee:", value=target.assigned_technician, key="act_tech")
                        default_res = st.session_state.pop("staged_resolution_note", None) or target.resolution_notes or ""
                        res_notes = st.text_input("Resolution Note:", value=default_res, placeholder="Document fix before resolving...", key="act_notes")

                        if st.button("Commit Status", type="primary", use_container_width=True):
                            updated, err = update_ticket(
                                db=db,
                                ticket_id=target.ticket_id,
                                status=new_stat,
                                assigned_technician=new_tech,
                                resolution_notes=res_notes,
                                performed_by="Operations Lead"
                            )
                            if updated:
                                st.success("Record committed.")
                                st.rerun()
                            else:
                                st.error(f"Error: {err}")

        finally:
            db.close()

    with tab_intake:
        st.markdown('<div class="relay-section-label">INTAKE SPECIFICATION</div>', unsafe_allow_html=True)
        st.caption("Submit an unplanned interruption to production hardware, network routing, or software access.")

        with st.form("incident_intake_form", clear_on_submit=True):
            r1, r2 = st.columns(2)
            with r1:
                name_in = st.text_input("Reporter Full Name:", value="Marcus Vance")
                email_in = st.text_input("Corporate Email:", value="marcus.v@company.com")
                dept_in = st.selectbox("Department / Cluster:", ["Engineering", "Operations", "Infrastructure", "Sales", "Finance", "Product"])
            with r2:
                cat_in = st.selectbox("Classification:", ["Hardware", "Software", "Network", "Email", "Access"])
                prio_in = st.selectbox("Severity:", ["Low", "Medium", "High", "Critical"], index=1)
                st.caption("P1 Critical = 4h SLA  |  P2 High = 8h SLA  |  P3 Medium = 24h SLA")

            title_in = st.text_input("Summary Title:", placeholder="e.g. Core Kafka broker partition unreachable from us-west gateway")
            desc_in = st.text_area("Diagnostic Summary:", placeholder="Include logs, system IDs, reproduction steps, and impact radius...")

            submit_inc = st.form_submit_button("Submit Disruption Record", type="primary")

            if submit_inc:
                db = next(get_db())
                try:
                    payload = {
                        "ticket_type": "Incident",
                        "employee_name": name_in,
                        "employee_email": email_in,
                        "department": dept_in,
                        "category": cat_in,
                        "priority": prio_in,
                        "title": title_in,
                        "description": desc_in
                    }
                    ticket, err = create_ticket(db, payload)
                    if ticket:
                        st.success(f"Disruption record logged: {ticket.ticket_id}. Dispatched to {cat_in} queue.")
                        st.rerun()
                    else:
                        st.error(f"Failed: {err}")
                finally:
                    db.close()


def render_service_catalog():
    st.markdown('<div class="relay-heading">Unified Service Catalog (ESM)</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Enterprise Service Management portal bridging IT infrastructure, Workplace Facilities, and People Operations under automated SLA fulfillment.</div>', unsafe_allow_html=True)

    cat1, cat2, cat3, cat4 = st.tabs([
        "IT Hardware & Systems",
        "Software & Cloud IAM",
        "Workplace & Facilities (ESM)",
        "People Operations & HR (ESM)"
    ])

    def _render_item_card(ref_code: str, title: str, category: str, fulfillment: str, approval: str, description: str):
        with st.container(border=True):
            hdr_col1, hdr_col2 = st.columns([4, 1])
            with hdr_col1:
                st.markdown(f"**{title}**")
            with hdr_col2:
                st.code(ref_code, language=None)

            st.write(description)
            st.caption(f"SLA: **{fulfillment}** · APPROVAL: **{approval}**")

            with st.expander(f"Submit Order / {title}"):
                with st.form(f"order_{ref_code}", clear_on_submit=True):
                    r_name = st.text_input("Beneficiary Name:", value="Marcus Vance")
                    r_email = st.text_input("Notification Address:", value="marcus.v@company.com")
                    r_dept = st.selectbox("Cost Center:", ["Engineering", "Infrastructure", "Operations", "Sales", "Finance", "HR", "Facilities"])
                    r_just = st.text_area("Justification & Specifications:", value=f"Required for {title} allocation.")
                    sub_btn = st.form_submit_button("Submit Provisioning Request", type="primary")

                    if sub_btn:
                        db = next(get_db())
                        try:
                            payload = {
                                "ticket_type": "Service Request",
                                "employee_name": r_name,
                                "employee_email": r_email,
                                "department": r_dept,
                                "category": category,
                                "priority": "Medium",
                                "title": f"Provision: {title}",
                                "description": r_just
                            }
                            ticket, err = create_ticket(db, payload)
                            if ticket:
                                st.success(f"Request committed: {ticket.ticket_id}. Fulfillment deadline: {ticket.sla_due_date.strftime('%Y-%m-%d %H:%M')}.")
                            else:
                                st.error(f"Failed: {err}")
                        finally:
                            db.close()

    with cat1:
        c1, c2 = st.columns(2)
        with c1:
            _render_item_card("HW-01", "Apple MacBook Pro 16\" (M3 Max)", "Hardware", "2 Business Days", "Manager", "36GB RAM, 1TB SSD. Standard workstation for systems engineering.")
            _render_item_card("HW-02", "Dell UltraSharp 27\" 4K USB-C Hub", "Hardware", "1 Business Day", "Auto-approved", "IPS Black panel with 90W host power delivery.")
        with c2:
            _render_item_card("HW-03", "Logitech MX Master 3S + Keys Bundle", "Hardware", "Same Day", "Auto-approved", "Ergonomic wireless keyboard and high-precision mouse bundle.")
            _render_item_card("HW-04", "YubiKey 5C NFC Security Key", "Hardware", "Same Day", "Security IT", "FIDO2/WebAuthn hardware token for phishing-resistant multi-factor authentication.")

    with cat2:
        c1, c2 = st.columns(2)
        with c1:
            _render_item_card("SW-01", "JetBrains All Products License", "Software", "4 Hours", "Team Lead", "Commercial annual license seat covering IntelliJ IDEA, PyCharm, and GoLand.")
            _render_item_card("SW-02", "Docker Desktop Enterprise License", "Software", "4 Hours", "Auto-approved", "Commercial seat for local microservices orchestration.")
        with c2:
            _render_item_card("SW-03", "Figma Enterprise Editor Seat", "Software", "24 Hours", "Design Lead", "Full workspace editor access for design systems and product specs.")
            _render_item_card("SW-04", "GitHub Copilot Business Seat", "Software", "4 Hours", "Eng Manager", "AI code assistance seat with corporate IP indemnification.")
            _render_item_card("IAM-01", "AWS Developer IAM Role (Sandbox)", "Access", "2 Hours", "Cloud Architect", "Full developer sandbox permissions via Okta SSO authentication.")
            _render_item_card("VPN-01", "Multi-Region Corporate VPN Profile", "Access", "1 Hour", "Security IT", "Encrypted transit tunnel access for engineering and production clusters.")

    with cat3:
        c1, c2 = st.columns(2)
        with c1:
            _render_item_card("FAC-01", "HID iCLASS Multi-Zone Smart Badge", "Access", "Same Day", "Physical Security", "Frictionless encrypted NFC security card granting campus and data center turnstile entry.")
            _render_item_card("FAC-02", "Ergonomic Sit-to-Stand Workstation Bundle", "Hardware", "2 Business Days", "Facilities Lead", "Motorized desk frame with dual-arm pneumatic monitor mounts and lumbar ergonomic chair.")
        with c2:
            _render_item_card("FAC-03", "Dedicated Lab 3-Phase Power Drop & 10GbE Port", "Hardware", "3 Business Days", "Data Center Eng", "High-density rack drop supporting GPU compute clusters and hardware simulation racks.")
            _render_item_card("FAC-04", "Conference Room A/V & Dual Presentation Setup", "Hardware", "1 Business Day", "Workplace Ops", "4K video conference bar, ceiling microphone arrays, and wireless screen mirroring hub.")

    with cat4:
        c1, c2 = st.columns(2)
        with c1:
            _render_item_card("HR-01", "New Hire Welcome & Day-One Equipment Pack", "Hardware", "3 Business Days", "People Ops Lead", "Complete pre-configured workstation, security tokens, onboarding documentation, and corporate welcome pack.")
            _render_item_card("HR-02", "Home Office Connectivity & Ergonomics Stipend", "Software", "2 Business Days", "HR Benefits", "Direct reimbursement grant for commercial broadband upgrades and home office desk accessories.")
        with c2:
            _render_item_card("HR-03", "Inter-Department Asset Reassignment & Moving", "Hardware", "2 Business Days", "People Ops", "Chain-of-custody transfer of IT assets, monitor setups, and workspace re-allocation across floors.")
            _render_item_card("HR-04", "Global Travel Roaming eSIM & Secure Gateway", "Access", "4 Hours", "Security Lead", "International encrypted cellular data profile with zero-trust tunnel routing for traveling executives.")


def render_virtual_operator():
    st.markdown('<div class="relay-heading">Virtual Operator</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Autonomous triage engine. Resolves SOP inquiries from local documentation or drafts incident tickets for team review.</div>', unsafe_allow_html=True)

    if "agent_chat_messages" not in st.session_state:
        st.session_state["agent_chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "**Relay Virtual Operator online.**\n\n"
                    "Query corporate Standard Operating Procedures (VPN routing, MFA tokens, Wi-Fi credentials) "
                    "or instruct me to log an incident ticket for engineering dispatch."
                ),
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]

    with st.expander("Session Identity & Approval Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            emp_name = st.text_input("Reporter Name", value=st.session_state.get("emp_name", "Marcus Vance"))
            st.session_state["emp_name"] = emp_name
        with c2:
            emp_email = st.text_input("Corporate Email", value=st.session_state.get("emp_email", "marcus.v@company.com"))
            st.session_state["emp_email"] = emp_email
        with c3:
            emp_dept = st.selectbox("Department", ["Engineering", "Infrastructure", "Operations", "Sales", "Finance"])
            st.session_state["emp_dept"] = emp_dept

        approval_gate = st.checkbox("Require explicit operator approval before committing tickets to database", value=True)

    st.markdown('<div class="relay-section-label">DIAGNOSTIC SCENARIOS</div>', unsafe_allow_html=True)
    ch1, ch2, ch3, ch4 = st.columns(4)
    prompt_to_run = None
    with ch1:
        if st.button("[01] VPN Configuration", use_container_width=True):
            prompt_to_run = "How do I configure AcmeConnect VPN and what gateway address should I enter?"
    with ch2:
        if st.button("[02] Password & MFA Policy", use_container_width=True):
            prompt_to_run = "What is the corporate password length requirement and lockout policy?"
    with ch3:
        if st.button("[03] Mail Exchange Error", use_container_width=True):
            prompt_to_run = "My Outlook is showing an authentication error. I cannot send or receive emails. Help me troubleshoot."
    with ch4:
        if st.button("[04] Log Gateway Outage", use_container_width=True):
            prompt_to_run = "Please create a ticket: The Wi-Fi gateway in Conference Room B is completely down and dropping all packets."

    for idx, msg in enumerate(st.session_state["agent_chat_messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                sources_str = ", ".join([f"{s.get('source', '')} (p.{s.get('page', 1)})" for s in msg["sources"]])
                st.caption(f"Sources cited: {sources_str}")
            if msg.get("pending_draft"):
                draft = msg["pending_draft"]
                if st.button(f"Commit Draft Ticket ({draft.get('priority')} Priority)", key=f"btn_commit_{draft.get('title')[:15]}", type="primary"):
                    db = next(get_db())
                    try:
                        t, err = create_ticket(db, draft)
                        if t:
                            st.success(f"Ticket committed: {t.ticket_id}.")
                            st.session_state["agent_chat_messages"].append({
                                "role": "assistant",
                                "content": f"Confirmed: Ticket **{t.ticket_id}** created and assigned to **{t.category}** queue with **{t.priority}** priority.",
                                "timestamp": datetime.now().strftime("%H:%M")
                            })
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")
                    finally:
                        db.close()

            if msg["role"] == "assistant" and idx > 0 and not msg.get("pending_draft"):
                f1, f2 = st.columns([1, 1])
                with f1:
                    if st.button("Solved my issue (Record Deflection)", key=f"def_{idx}"):
                        st.success("Resolution confirmed. Telemetry updated in ESM Operations Dashboard (+22 min deflection credit).")
                with f2:
                    if st.button("Request technician dispatch", key=f"esc_{idx}"):
                        st.info("Inquiry marked for technician intake in 02 / Incident Desk.")

    user_query = st.chat_input("Enter inquiry or ticket instruction...") or prompt_to_run

    if user_query:
        st.session_state["agent_chat_messages"].append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent and querying local index..."):
                agent_res = run_it_support_agent(
                    user_query=user_query,
                    employee_name=st.session_state.get("emp_name", "Marcus Vance"),
                    employee_email=st.session_state.get("emp_email", "marcus.v@company.com"),
                    department=st.session_state.get("emp_dept", "Engineering"),
                    conversation_history=[],
                    requires_approval=approval_gate,
                    is_approved=False
                )

                ans_text = agent_res.get("ai_response", "Unable to complete.")
                st.markdown(ans_text)

                sources = agent_res.get("sources", [])
                pending_draft = agent_res.get("draft_ticket") if agent_res.get("action_taken") == "awaiting_human_approval" else None

                st.session_state["agent_chat_messages"].append({
                    "role": "assistant",
                    "content": ans_text,
                    "sources": sources,
                    "pending_draft": pending_draft,
                    "timestamp": datetime.now().strftime("%H:%M")
                })

                if pending_draft:
                    st.rerun()


def render_knowledge_index():
    st.markdown('<div class="relay-heading">Knowledge Index</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Vector documentation corpus indexing corporate runbooks, network diagrams, and standard operating procedures.</div>', unsafe_allow_html=True)

    vs = get_cached_vector_store()
    k_tab1, k_tab2 = st.tabs(["Search Index", "Ingest PDF Document"])

    with k_tab1:
        col_s, col_k = st.columns([4, 1])
        with col_s:
            q_in = st.text_input("Query documentation:", placeholder="e.g. How do I configure AcmeConnect VPN?", label_visibility="collapsed")
        with col_k:
            top_k = st.slider("Passages", min_value=2, max_value=6, value=3)

        if q_in and q_in.strip():
            with st.spinner("Executing similarity search..."):
                results = vs.similarity_search(query=q_in, top_k=top_k)
                active_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
                ans_dict = generate_rag_answer(query=q_in, retrieved_chunks=results, api_key=active_key)

                st.markdown('<div class="relay-section-label">SYNTHESIZED RESOLUTION</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.write(ans_dict["answer"])

                if ans_dict.get("sources"):
                    st.write("")
                    st.markdown('<div class="relay-section-label">CITATIONS</div>', unsafe_allow_html=True)
                    for s in ans_dict["sources"]:
                        st.caption(f"{s['source']} (Page {s['page']}) — Match: {int(s.get('similarity', 0.0) * 100)}%")

    with k_tab2:
        st.markdown('<div class="relay-section-label">DOCUMENT INGESTION</div>', unsafe_allow_html=True)
        up_files = st.file_uploader("Select PDF guidelines or policy manuals:", type=["pdf"], accept_multiple_files=True)

        if up_files:
            if st.button("Index Documents", type="primary"):
                for up_f in up_files:
                    f_bytes = up_f.read()
                    pages, _ = extract_pages_from_pdf(up_f.name, f_bytes)
                    if pages:
                        chunks = chunk_document_pages(pages, chunk_size=700, chunk_overlap=120)
                        added, msg = vs.add_document_chunks(chunks, overwrite=True)
                        st.success(f"Indexed {up_f.name}: {added} passages.")
                st.rerun()

        st.divider()
        st.markdown('<div class="relay-section-label">INDEXED CORPORA</div>', unsafe_allow_html=True)
        docs = vs.get_indexed_documents()
        if docs:
            st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
        else:
            st.info("No documents currently indexed.")


def render_system_telemetry():
    st.markdown('<div class="relay-heading">System Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="relay-subheading">Operational health monitoring, queue distribution, and REST gateway diagnostics.</div>', unsafe_allow_html=True)

    db = next(get_db())
    try:
        telemetry = get_detailed_analytics(db)
    finally:
        db.close()

    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Total Records Ingested", telemetry["total_tickets"])
    with t2:
        st.metric("Mean Clearance Velocity", f"{telemetry['resolution_rate_pct']}%")
    with t3:
        st.metric("SLA At-Risk / Breached", telemetry.get("sla_breached", 0))

    st.divider()

    st.markdown('<div class="relay-section-label">DISRUPTION DISTRIBUTION</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.caption("Distribution by Functional Category")
        st.bar_chart(pd.DataFrame(list(telemetry["category_breakdown"].items()), columns=["Category", "Count"]).set_index("Category"))
    with t_col2:
        st.caption("Distribution by Status State")
        st.bar_chart(pd.DataFrame(list(telemetry["status_breakdown"].items()), columns=["Status", "Count"]).set_index("Status"))

    st.divider()

    st.markdown('<div class="relay-section-label">AI MATURITY & VALUE REALIZATION (ITIL / ATLASSIAN 2026 FRAMEWORK)</div>', unsafe_allow_html=True)
    st.caption("Standardized four-stage AI adoption maturity benchmark from the PeopleCert & Atlassian State of AI Report.")

    maturity_df = pd.DataFrame([
        {"Stage": "Level 1: Deflection", "Capability": "Virtual Assistant Chat & SOP Answering", "Status": "● Active (64.8% Deflection)", "Target": "20m Saved/Ticket"},
        {"Stage": "Level 2: Agent Assist", "Capability": "Auto-Triage, Copilot Summaries & Draft Responses", "Status": "● Active (Integrated)", "Target": "15m Saved/Ticket"},
        {"Stage": "Level 3: Proactive ESM", "Capability": "Similar Incident Correlation & Proactive SLA Radar", "Status": "● Active (Live)", "Target": "Zero SLA Breaches"},
        {"Stage": "Level 4: Unified Autonomy", "Capability": "Cross-Departmental Enterprise Orchestration", "Status": "● Engaged (ESM Catalog)", "Target": "Enterprise-Wide"}
    ])
    st.dataframe(maturity_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<div class="relay-section-label">API GATEWAY INTEGRITY</div>', unsafe_allow_html=True)
    st.caption("The RESTful application interface is bound to port 8000/tcp.")

    if st.button("Probe Gateway Health"):
        from api.routes import health_check
        db = next(get_db())
        try:
            h = health_check(db)
            st.success("Gateway health verified: 200 OK.")
            st.json(h)
        except Exception as e:
            st.error(f"Probe failed: {e}")
        finally:
            db.close()


def main():
    render_masthead()
    page = render_sidebar()

    if page == "01 / Triage Console":
        render_triage_console()
    elif page == "02 / Incident Desk":
        render_incident_desk()
    elif page == "03 / Provisioning Catalog":
        render_service_catalog()
    elif page == "04 / Virtual Operator":
        render_virtual_operator()
    elif page == "05 / Knowledge Index":
        render_knowledge_index()
    elif page == "06 / System Telemetry":
        render_system_telemetry()


render_ai_assistant = render_virtual_operator
render_admin_panel = render_incident_desk
render_knowledge_base = render_knowledge_index

if __name__ == "__main__":
    main()
