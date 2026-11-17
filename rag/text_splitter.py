from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks: List[Dict[str, Any]] = []

    for page_item in pages:
        page_text = page_item["text"]
        page_meta = page_item["metadata"]

        splits = splitter.split_text(page_text)
        for idx, split in enumerate(splits):
            cleaned = split.strip()
            if cleaned:
                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "source": page_meta.get("source", "unknown"),
                        "page": page_meta.get("page", 1),
                        "total_pages": page_meta.get("total_pages", 1),
                        "chunk_index": idx + 1,
                        "global_chunk_id": f"{page_meta.get('source', 'doc')}_p{page_meta.get('page', 1)}_c{idx + 1}"
                    }
                })

    return chunks
