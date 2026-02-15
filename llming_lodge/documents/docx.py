"""DOCX text extraction using python-docx."""
from pathlib import Path


def extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs) if paragraphs else "[No text content found in document]"
