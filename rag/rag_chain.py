"""
RAG Generation Chain Module using OpenAI.

Synthesizes answers to user queries grounded on retrieved
knowledge base document chunks, citing source documents and page numbers.
Supports both OpenAI LLM synthesis and robust local extraction fallback.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    import importlib
    import config
    importlib.reload(config)
    from config import (
        LLM_MODEL_NAME,
        OPENAI_API_KEY,
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

logger = logging.getLogger("USM_RAG")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """You are the AI IT Knowledge Assistant for the Unified Service Management (USM) System.
Your job is to answer employee questions accurately and professionally, based on the provided technical documentation context.

STRICT OPERATIONAL RULES:
1. Base your answer primarily on the provided context passages below.
2. If the context does not contain the answer, explicitly state:
   "The provided internal IT documentation does not specifically cover this topic. Here is general guidance based on enterprise IT standards: [provide guidance]. Please submit an Incident Ticket if you require a technician."
3. Cite the source document and page number in parentheses (e.g., [acme_it_sop.pdf, Page 2]).
4. Be concise, professional, and use bullet points for multi-step procedures.
"""


def format_context_passages(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved vector chunks into structured context text with citations.
    """
    if not retrieved_chunks:
        return "No relevant documentation found."

    formatted_passages = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "Unknown Document")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        formatted_passages.append(
            f"--- Context Passage {idx} [Document: {source} | Page: {page}] ---\n{text}"
        )

    return "\n\n".join(formatted_passages)


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the Generation step of the RAG pipeline.
    
    Args:
        query: Employee's IT support question.
        retrieved_chunks: Top-k matching passages retrieved from ChromaDB.
        api_key: OpenAI API key (falls back to environment variable).
        model_name: Target OpenAI chat model (default: gpt-4o-mini).
        
    Returns:
        Dictionary containing 'answer', 'sources', and 'context_passages'.
    """
    effective_key = api_key or get_openai_api_key()
    target_model = model_name or os.getenv("LLM_MODEL_NAME") or LLM_MODEL_NAME

    logger.info(f"[RAG_CHAIN] Generating answer for query: '{query}' with {len(retrieved_chunks)} context chunks. OpenAI configured: {is_openai_configured()} ({get_masked_api_key()})")

    # Extract unique sources for UI display
    unique_sources = []
    seen = set()
    for chunk in retrieved_chunks:
        source_key = (chunk.get("source"), chunk.get("page"))
        if source_key not in seen:
            seen.add(source_key)
            unique_sources.append({
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", 1),
                "similarity": chunk.get("similarity_score", 0.0)
            })

    # Case 1: No documents retrieved
    if not retrieved_chunks:
        logger.info("[RAG_CHAIN] No matching vector chunks found in ChromaDB.")
        # Provide helpful domain troubleshooting baseline if specific keywords match
        q_low = query.lower()
        if "outlook" in q_low or "email" in q_low:
            fallback = (
                "### 📧 Standard Email & Outlook Troubleshooting Steps\n\n"
                "1. **Check SSO & MFA:** Verify that your corporate SSO login is active and approve any pending Okta MFA prompts.\n"
                "2. **Clear Cached Credentials:** Open Windows Credential Manager or macOS Keychain and delete stale Microsoft Exchange login tokens.\n"
                "3. **Verify Webmail:** Test your account at https://outlook.office.com. If webmail fails, your account may be locked or password expired.\n"
                "4. **Repair Connection:** In Outlook desktop, go to `File > Account Settings > Repair`.\n"
                "5. **Escalate:** If authentication continues to fail, submit an IT Incident ticket under Category **Email**."
            )
        elif "vpn" in q_low:
            fallback = (
                "### 🌐 Corporate VPN Troubleshooting Steps\n\n"
                "1. Verify internet connectivity on your local workstation.\n"
                "2. Open the AcmeConnect VPN client and confirm gateway `gw-west.acme-it.corp` or `gw-east.acme-it.corp`.\n"
                "3. Complete Okta MFA push verification.\n"
                "4. If Error 403 occurs (Certificate Expired), log an IT Incident under Category **Network**."
            )
        else:
            fallback = "The internal Knowledge Base does not contain documentation covering this specific question. Please contact the IT Service Desk or log an Incident Ticket."

        return {
            "answer": fallback,
            "sources": [],
            "has_error": False,
            "used_ai": False
        }

    # Case 2: API Key missing or placeholder (Local Extraction Fallback)
    if not effective_key or not effective_key.strip().startswith("sk-"):
        logger.info("[RAG_CHAIN] OpenAI API key not configured. Using verified passage extraction engine.")
        passage_blocks = []
        for c in retrieved_chunks:
            src = c.get("source", "Technical Documentation")
            pg = c.get("page", 1)
            raw_text = c.get("text", "").strip()
            if raw_text:
                passage_blocks.append(f"📄 **From `{src}` (Page {pg}):**\n{raw_text}")

        extracted_body = "\n\n".join(passage_blocks)
        fallback_answer = (
            f"### 📋 Verified IT Documentation Instructions\n\n"
            f"{extracted_body}\n\n"
            f"---\n"
            f"ℹ️ *Extracted directly from verified corporate SOP documentation.*"
        )
        return {
            "answer": fallback_answer,
            "sources": unique_sources,
            "retrieved_chunks": retrieved_chunks,
            "has_error": False,
            "used_ai": False
        }

    # Case 3: Live OpenAI LLM Inference
    try:
        logger.info(f"[RAG_CHAIN] Executing OpenAI completion using model '{target_model}'.")
        client = OpenAI(api_key=effective_key.strip())
        context_str = format_context_passages(retrieved_chunks)

        user_content = (
            f"CONTEXT PASSAGES FROM IT KNOWLEDGE BASE:\n"
            f"{context_str}\n\n"
            f"EMPLOYEE QUESTION:\n"
            f"{query.strip()}"
        )

        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,  # Low temperature to minimize hallucination
            max_tokens=800
        )

        answer_text = response.choices[0].message.content or ""
        logger.info(f"[RAG_CHAIN] OpenAI response generated successfully (length: {len(answer_text)} chars).")

        return {
            "answer": answer_text,
            "sources": unique_sources,
            "retrieved_chunks": retrieved_chunks,
            "has_error": False,
            "used_ai": True
        }

    except Exception as e:
        logger.error(f"[RAG_CHAIN] OpenAI API generation failed: {e}", exc_info=True)
        # Fall back gracefully to extracted text if API call fails
        passage_blocks = [
            f"📄 **From `{c.get('source')}` (Page {c.get('page')}):**\n{c.get('text', '').strip()}"
            for c in retrieved_chunks if c.get("text")
        ]
        fallback_answer = (
            f"⚠️ *OpenAI API request encountered an error ({str(e)}). Displaying verified documentation excerpts below:*\n\n"
            + "\n\n".join(passage_blocks)
        )
        return {
            "answer": fallback_answer,
            "sources": unique_sources,
            "retrieved_chunks": retrieved_chunks,
            "has_error": True,
            "used_ai": False
        }
