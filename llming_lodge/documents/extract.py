"""Document text extraction dispatcher and token budget management."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ~4 chars per token is a safe approximation for English text
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 80_000  # total budget across all documents

# Registry mapping MIME types to extractor functions
_EXTRACTORS: dict[str, callable] = {}


def _register_defaults() -> None:
    """Lazily register built-in extractors on first use."""
    if _EXTRACTORS:
        return

    from llming_lodge.documents.pdf import extract_pdf
    from llming_lodge.documents.docx import extract_docx
    from llming_lodge.documents.xlsx import extract_xlsx

    _EXTRACTORS["application/pdf"] = extract_pdf
    _EXTRACTORS["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] = extract_docx
    _EXTRACTORS["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] = extract_xlsx


def extract_text(path: Path, mime_type: str) -> str:
    """Extract text content from a document file.

    Supports PDF, DOCX, and XLSX out of the box.
    Returns extracted text or an error message.
    """
    _register_defaults()
    extractor = _EXTRACTORS.get(mime_type)
    if extractor is None:
        return f"[Unsupported file type: {mime_type}]"
    try:
        return extractor(path)
    except Exception as e:
        logger.error(f"[DOC] Failed to extract text from {path}: {e}")
        return f"[Error extracting text: {e}]"


def truncate_to_token_budget(
    documents: List[Tuple[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[Tuple[str, str]]:
    """Truncate document texts so their combined size stays within a token budget.

    Args:
        documents: list of (filename, extracted_text) tuples
        max_tokens: total token budget across all documents

    Returns:
        list of (filename, possibly-truncated text) tuples
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    remaining = max_chars
    result = []

    for name, text in documents:
        if remaining <= 0:
            result.append((name, "[Skipped — token budget exhausted]"))
            continue

        if len(text) <= remaining:
            result.append((name, text))
            remaining -= len(text)
        else:
            truncated = text[:remaining]
            # Try to cut at a line boundary
            last_nl = truncated.rfind('\n')
            if last_nl > remaining * 0.8:
                truncated = truncated[:last_nl]
            result.append((name, truncated + "\n\n[... truncated — document exceeded remaining token budget]"))
            remaining = 0

    return result
