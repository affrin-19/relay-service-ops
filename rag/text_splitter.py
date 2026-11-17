"""
Document Chunking and Segmentation Module.

Splits extracted PDF text into semantically cohesive chunks while preserving
document source, page numbers, and chunk sequence indices.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Divides page-level documents into smaller, overlapping text chunks.
    
    Why chunking is essential in RAG architectures:
    1. Precision: Vector embeddings created from a focused paragraph capture specific meanings
       much better than an entire 20-page document averaged together.
    2. Context Window Efficiency: LLM prompt context is limited; chunking allows sending only
       the exact relevant passages rather than the entire manual.
    3. Overlap continuity: Chunk overlap ensures that a sentence or procedure split at a chunk
       boundary does not lose its semantic meaning.
       
    Args:
        pages: List of dictionaries with 'text' and 'metadata' (from pdf_loader).
        chunk_size: Maximum character count per chunk (default 700 chars ~ 150 tokens).
        chunk_overlap: Shared characters between consecutive chunks (default 120 chars).
        
    Returns:
        List of chunk dictionaries with 'text', 'metadata', and 'chunk_id'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_item in pages:
        page_text = page_item["text"]
        page_meta = page_item["metadata"]

        raw_splits = splitter.split_text(page_text)

        for split_idx, split_text in enumerate(raw_splits):
            if split_text.strip():
                chunk_counter += 1
                chunk_metadata = {
                    "source": page_meta.get("source", "unknown"),
                    "page": page_meta.get("page", 1),
                    "total_pages": page_meta.get("total_pages", 1),
                    "chunk_index": split_idx + 1,
                    "global_chunk_id": f"{page_meta.get('source', 'doc')}_p{page_meta.get('page', 1)}_c{split_idx + 1}"
                }

                all_chunks.append({
                    "text": split_text.strip(),
                    "metadata": chunk_metadata
                })

    return all_chunks

