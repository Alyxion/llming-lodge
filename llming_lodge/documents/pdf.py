"""PDF text extraction using pdfplumber, preserving hyperlinks."""
import io
from pathlib import Path
from typing import Union


def extract_pdf(source: Union[Path, bytes]) -> str:
    import pdfplumber

    fp = io.BytesIO(source) if isinstance(source, bytes) else source
    pages = []
    with pdfplumber.open(fp) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            # Collect hyperlinks and append as reference list
            try:
                links = page.hyperlinks
            except Exception:
                links = []

            if links:
                # Deduplicate URLs preserving order
                seen = set()
                unique_urls = []
                for link in links:
                    url = link.get("uri", "")
                    if url and url not in seen:
                        seen.add(url)
                        unique_urls.append(url)
                if unique_urls:
                    ref_lines = "\n".join(f"  - {url}" for url in unique_urls)
                    text = f"{text}\n\nLinks on this page:\n{ref_lines}"

            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages) if pages else "[No text content found in PDF]"
