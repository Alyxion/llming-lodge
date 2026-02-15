"""PDF text extraction using pdfplumber."""
from pathlib import Path


def extract_pdf(path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages) if pages else "[No text content found in PDF]"
