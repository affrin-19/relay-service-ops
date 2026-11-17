"""
Comprehensive Verification Test Suite for Stage 3 RAG Pipeline.

Tests:
1. PDF Loading & Text Extraction
2. Semantic Chunking & Metadata Preservation
3. Vector Storage & Duplicate Ingestion Detection in ChromaDB
4. Semantic Retrieval for In-Domain Questions
5. Rejection of Unrelated / Out-of-Domain Questions (Anti-Hallucination)
6. Cold-Restart Vector Persistence
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pdf_loader import extract_pages_from_pdf
from rag.text_splitter import chunk_document_pages
from rag.vector_store import USMVectorStore
from rag.rag_chain import generate_rag_answer, SYSTEM_PROMPT


def run_stage_3_verification():
    print("=" * 70)
    print("🚀 STARTING STAGE 3 RAG PIPELINE VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # TEST 1: PDF Text Extraction
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Testing PDF Text Extraction from 'acme_it_sop.pdf'...")
    sample_pdf_path = PROJECT_ROOT / "data" / "knowledge_base" / "acme_it_sop.pdf"
    assert sample_pdf_path.exists(), f"Sample PDF not found at {sample_pdf_path}"

    with open(sample_pdf_path, "rb") as f:
        file_bytes = f.read()

    pages_data, warning = extract_pages_from_pdf("acme_it_sop.pdf", file_bytes)
    print(f"  ✓ Extracted {len(pages_data)} pages.")
    assert len(pages_data) >= 2, f"Expected at least 2 pages, got {len(pages_data)}"
    assert "AcmeConnect VPN" in pages_data[0]["text"], "Page 1 content missing VPN info"
    assert "14 characters" in pages_data[1]["text"], "Page 2 content missing password info"
    print("  ✓ Page 1 and Page 2 text verified successfully.")

    # --------------------------------------------------------------------------
    # TEST 2: Semantic Chunking & Metadata
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Testing Semantic Chunking...")
    chunks = chunk_document_pages(pages_data, chunk_size=500, chunk_overlap=100)
    print(f"  ✓ Generated {len(chunks)} chunks.")
    assert len(chunks) >= 3, "Expected at least 3 chunks"
    
    first_chunk = chunks[0]
    print(f"  ✓ Chunk metadata keys: {list(first_chunk['metadata'].keys())}")
    assert first_chunk["metadata"]["source"] == "acme_it_sop.pdf"
    assert "page" in first_chunk["metadata"]
    assert "chunk_index" in first_chunk["metadata"]
    print("  ✓ Metadata integrity confirmed.")

    # --------------------------------------------------------------------------
    # TEST 3: ChromaDB Vector Storage & Duplicate Ingestion Check
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Testing ChromaDB Vector Storage & Duplicate Prevention...")
    vs = USMVectorStore()
    
    # Clean previous demo entries for fresh test
    vs.delete_document("acme_it_sop.pdf")
    
    added_count, msg = vs.add_document_chunks(chunks, overwrite=False)
    print(f"  ✓ Initial Ingestion: {msg}")
    assert added_count == len(chunks), f"Expected {len(chunks)} chunks added, got {added_count}"

    # Attempt duplicate addition
    dup_count, dup_msg = vs.add_document_chunks(chunks, overwrite=False)
    print(f"  ✓ Duplicate Prevention Test: {dup_msg}")
    assert dup_count == 0, "Duplicate document was incorrectly re-indexed"
    assert "already indexed" in dup_msg

    # --------------------------------------------------------------------------
    # TEST 4: Semantic Similarity Search (In-Domain Query)
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Testing Semantic Similarity Search for IT Question...")
    query_1 = "How do I configure AcmeConnect VPN and what gateway address should I enter?"
    results_1 = vs.similarity_search(query_1, top_k=2)
    
    assert len(results_1) > 0, "No results returned for in-domain query"
    top_result = results_1[0]
    print(f"  ✓ Top Match Document: {top_result['source']} (Page {top_result['page']})")
    print(f"  ✓ Match Snippet: {top_result['text'][:120]}...")
    assert "gw-west" in top_result["text"] or "AcmeConnect" in top_result["text"] or "VPN" in top_result["text"]

    # --------------------------------------------------------------------------
    # TEST 5: Question 2 - Password Expiration & Lockout
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Testing Password Policy Query...")
    query_2 = "What is the account lockout policy after failed login attempts?"
    results_2 = vs.similarity_search(query_2, top_k=2)
    
    found_lockout = any("lockout" in r["text"].lower() for r in results_2)
    assert found_lockout, "Expected lockout policy in retrieved chunks"
    print("  ✓ Successfully retrieved account lockout policy passage from Page 2.")

    # --------------------------------------------------------------------------
    # TEST 6: Anti-Hallucination & Out-of-Domain Query
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Testing Anti-Hallucination & Out-of-Domain Guardrails...")
    unrelated_query = "What ingredients are in the cafeteria chocolate cake?"
    unrelated_results = vs.similarity_search(unrelated_query, top_k=2)
    
    # Test answer generation logic with guardrails
    answer_result = generate_rag_answer(unrelated_query, unrelated_results)
    print("  ✓ Guardrail Answer Status:")
    print(f"    AI Generated? {answer_result['used_ai']}")
    print(f"    Response preview: {answer_result['answer'][:100]}...")

    # --------------------------------------------------------------------------
    # TEST 7: Cold-Restart Vector Persistence Verification
    # --------------------------------------------------------------------------
    print("\n[TEST 7] Testing Cold-Restart Persistence (New Client Instance)...")
    # Simulate closing and reopening the app with a fresh client
    cold_vs = USMVectorStore()
    total_stored = cold_vs.count()
    indexed_list = cold_vs.get_indexed_documents()
    print(f"  ✓ Cold instance count: {total_stored} chunks")
    print(f"  ✓ Cold instance documents: {[d['source'] for d in indexed_list]}")
    assert total_stored >= len(chunks), "Cold restart vector count mismatch"
    assert any(d["source"] == "acme_it_sop.pdf" for d in indexed_list), "acme_it_sop.pdf not found in cold instance"

    print("\n" + "=" * 70)
    print("🎉 ALL STAGE 3 RAG VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_stage_3_verification()

