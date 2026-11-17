"""
PDF Document Loader and Text Extraction Module.

Extracts text from uploaded PDF files page by page using PyPDF,
validating file constraints and preserving page metadata.
"""

import io
from typing import List, Dict, Any, Tuple
import pypdf

# Maximum file size constraint (15 Megabytes)
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


def validate_pdf_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """
    Validates the uploaded document filename and payload size.
    
    Args:
        filename: Name of the uploaded file.
        file_bytes: Raw bytes of the uploaded file.
        
    Returns:
        (True, "") if valid, or (False, "error message") if invalid.
    """
    if not filename.lower().endswith(".pdf"):
        return False, f"Invalid format: '{filename}' is not a PDF file. Only PDF files are supported."

    if len(file_bytes) == 0:
        return False, f"File '{filename}' is empty (0 bytes)."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return False, f"File '{filename}' exceeds maximum allowed size (15 MB). Current size: {size_mb:.2f} MB."

    return True, ""


def extract_pages_from_pdf(filename: str, file_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parses a PDF byte stream and extracts readable text per page.
    
    Args:
        filename: Original name of the document (used as metadata source).
        file_bytes: Raw PDF binary content.
        
    Returns:
        (pages_list, warning_message):
            pages_list is a list of dictionaries with text and metadata.
            warning_message indicates if text extraction was partially or completely degraded.
    """
    is_valid, error_msg = validate_pdf_file(filename, file_bytes)
    if not is_valid:
        raise ValueError(error_msg)

    pages_data: List[Dict[str, Any]] = []
    total_characters = 0

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        num_pages = len(reader.pages)

        if num_pages == 0:
            return [], f"Document '{filename}' contains 0 pages."

        for page_idx in range(num_pages):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            clean_text = page_text.strip()
            total_characters += len(clean_text)

            # Store page if text is present
            if clean_text:
                pages_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": filename,
                        "page": page_idx + 1,
                        "total_pages": num_pages
                    }
                })

        # Graceful handling of scanned images or empty PDFs
        warning = ""
        if total_characters < 30:
            warning = (
                f"⚠️ '{filename}' has {num_pages} page(s) but very little extractable text ({total_characters} chars). "
                "It may be a scanned image without an OCR text layer."
            )

        return pages_data, warning

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{filename}': {str(e)}")

