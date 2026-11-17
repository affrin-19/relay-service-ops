import io
from typing import List, Dict, Any, Tuple
import pypdf

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


def validate_pdf_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    if not filename.lower().endswith(".pdf"):
        return False, f"Invalid format: '{filename}' is not a PDF file. Only PDF files are supported."

    if len(file_bytes) == 0:
        return False, f"File '{filename}' is empty (0 bytes)."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return False, f"File '{filename}' exceeds maximum allowed size (15 MB). Current size: {size_mb:.2f} MB."

    return True, ""


def extract_pages_from_pdf(filename: str, file_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
    is_valid, error_msg = validate_pdf_file(filename, file_bytes)
    if not is_valid:
        raise ValueError(error_msg)

    pages: List[Dict[str, Any]] = []
    total_chars = 0

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)

        if num_pages == 0:
            return [], f"Document '{filename}' contains 0 pages."

        for idx in range(num_pages):
            page_text = (reader.pages[idx].extract_text() or "").strip()
            total_chars += len(page_text)

            if page_text:
                pages.append({
                    "text": page_text,
                    "metadata": {
                        "source": filename,
                        "page": idx + 1,
                        "total_pages": num_pages
                    }
                })

        warning = ""
        if total_chars < 30:
            warning = f"'{filename}' has {num_pages} page(s) but very little extractable text ({total_chars} chars)."

        return pages, warning

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{filename}': {str(e)}")
