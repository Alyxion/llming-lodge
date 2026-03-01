"""PdfViewerMCP — in-process MCP server for visual PDF page inspection.

Renders PDF pages as JPEG images for LLM visual analysis. Images are
injected into the LLM context via a callback (not returned as base64
in the tool result to avoid wasting tokens).

Uses pypdfium2 (BSD-3/Apache-2.0) for rendering — no AGPL dependencies.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llming_lodge.documents.image_registry import ImageEntry, ImageRegistry
from llming_lodge.tools.mcp import InProcessMCPServer

logger = logging.getLogger(__name__)

# Max pages per single tool call to prevent context bloat
MAX_PAGES_PER_CALL = 5
RENDER_DPI = 150

# PDF source: either a file path or raw bytes (for nudge/project files in memory)
PdfSource = Union[Path, bytes]


def _open_pdf(source: PdfSource):
    """Open a PDF from a file path or raw bytes."""
    import pypdfium2 as pdfium

    if isinstance(source, bytes):
        return pdfium.PdfDocument(source)
    return pdfium.PdfDocument(str(source))


def _render_page(source: PdfSource, page_num: int, dpi: int = RENDER_DPI) -> bytes:
    """Render a single PDF page to JPEG bytes using pypdfium2."""
    pdf = _open_pdf(source)
    try:
        if page_num < 0 or page_num >= len(pdf):
            raise ValueError(f"Page {page_num + 1} out of range (1–{len(pdf)})")
        page = pdf[page_num]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    finally:
        pdf.close()


def _get_page_count(source: PdfSource) -> int:
    """Get the number of pages in a PDF."""
    pdf = _open_pdf(source)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    """Parse a page specification like '1,3-5,7' into 0-based page indices."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = max(1, int(start_s.strip()))
            end = min(max_page, int(end_s.strip()))
            pages.update(range(start - 1, end))
        else:
            p = int(part)
            if 1 <= p <= max_page:
                pages.add(p - 1)
    return sorted(pages)


class PdfViewerMCP(InProcessMCPServer):
    """MCP server for visual PDF page inspection.

    Renders PDF pages as images for the LLM to analyze charts, diagrams,
    layouts, and other visual content that text extraction misses.
    """

    def __init__(
        self,
        pdf_sources: dict[str, PdfSource],
        image_registry: ImageRegistry,
        image_callback: Callable[[list[str], bool, list[dict] | None], None],
    ) -> None:
        """
        Args:
            pdf_sources: mapping of file_id → Path or bytes for each PDF
            image_registry: shared image registry for the session
            image_callback: called with (list[base64_data_uris], show_to_user)
                           to inject images into the LLM context
        """
        self._pdf_sources = dict(pdf_sources)
        self._registry = image_registry
        self._image_callback = image_callback
        self._page_counts: dict[str, int] = {}

        # Pre-compute page counts and register images in registry
        for file_id, path in self._pdf_sources.items():
            try:
                count = _get_page_count(path)
                self._page_counts[file_id] = count
                # Register all pages as available images
                entries = []
                for i in range(count):
                    entry = ImageEntry(
                        id=f"pdf:{file_id}:page:{i + 1}",
                        name=f"{file_id} — Page {i + 1}",
                        source="pdf",
                        page=i + 1,
                        file_id=file_id,
                        _data_getter=lambda p=path, pg=i: _render_page(p, pg),
                    )
                    entries.append(entry)
                image_registry.register_many(entries)
                logger.info("[PDF_MCP] Registered %d pages for %s", count, file_id)
            except Exception as e:
                logger.warning("[PDF_MCP] Failed to index %s: %s", file_id, e)

    def _resolve_file_id(self, file_id: Optional[str]) -> tuple[str, PdfSource]:
        """Resolve file_id, defaulting to the only PDF if there's just one."""
        if file_id:
            if file_id not in self._pdf_sources:
                raise ValueError(f"Unknown file_id '{file_id}'. Available: {list(self._pdf_sources.keys())}")
            return file_id, self._pdf_sources[file_id]
        if len(self._pdf_sources) == 1:
            fid = next(iter(self._pdf_sources))
            return fid, self._pdf_sources[fid]
        raise ValueError(
            f"Multiple PDFs available — specify file_id. Options: {list(self._pdf_sources.keys())}"
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        file_id_schema: dict[str, Any] = {
            "type": "string",
            "description": "File identifier of the PDF. Optional if only one PDF is uploaded.",
        }
        return [
            {
                "name": "pdf_get_pages",
                "displayName": "View PDF Pages",
                "displayDescription": "Render PDF pages as images for visual inspection",
                "icon": "picture_as_pdf",
                "description": (
                    "Render one or more PDF pages as images so you can visually inspect "
                    "charts, diagrams, tables, and layout. Use page specs like '1', '1-3', "
                    "or '1,3,5'. Maximum 5 pages per call. The rendered images are injected "
                    "into your context — the tool result only contains metadata."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pages": {
                            "type": "string",
                            "description": "Page specification: '1', '2-4', '1,3,5-7'. 1-based.",
                        },
                        "file_id": file_id_schema,
                    },
                    "required": ["pages"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "pdf_get_pages":
            return await self._pdf_get_pages(arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _pdf_get_pages(self, args: Dict[str, Any]) -> str:
        """Render pages and inject images into LLM context."""
        pages_spec = args.get("pages", "1")
        file_id_arg = args.get("file_id")

        try:
            file_id, source = self._resolve_file_id(file_id_arg)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        max_page = self._page_counts.get(file_id, 0)
        if max_page == 0:
            return json.dumps({"error": f"Could not determine page count for {file_id}"})

        try:
            page_indices = _parse_page_spec(pages_spec, max_page)
        except (ValueError, TypeError):
            return json.dumps({"error": f"Invalid page spec: {pages_spec}"})

        if not page_indices:
            return json.dumps({"error": f"No valid pages in spec '{pages_spec}' (PDF has {max_page} pages)"})

        if len(page_indices) > MAX_PAGES_PER_CALL:
            page_indices = page_indices[:MAX_PAGES_PER_CALL]

        # Render pages
        rendered = []
        images_b64 = []
        for idx in page_indices:
            try:
                jpeg_bytes = _render_page(source, idx)
                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                data_uri = f"data:image/jpeg;base64,{b64}"
                images_b64.append(data_uri)
                rendered.append({"page": idx + 1, "size_kb": len(jpeg_bytes) // 1024})
            except Exception as e:
                rendered.append({"page": idx + 1, "error": str(e)})

        # Inject images into LLM context via callback (also shown to user)
        if images_b64:
            page_info = [{"file_id": file_id, "page": idx + 1, "total_pages": max_page} for idx in page_indices]
            self._image_callback(images_b64, True, page_info)

        return json.dumps({
            "file": file_id,
            "total_pages": max_page,
            "rendered": rendered,
            "note": "Page images are now visible to both you and the user. Describe what you see without mentioning tool names.",
        })

    async def get_prompt_hints(self) -> List[str]:
        """Tell the LLM about available PDFs and their visual inspection tools."""
        hints = []
        for file_id in self._pdf_sources:
            count = self._page_counts.get(file_id, 0)
            hints.append(
                f"PDF '{file_id}' has {count} page(s). "
                f"You can visually inspect any page to analyze charts, diagrams, tables, and layout. "
                f"The text content is already available in the system prompt above. "
                f"IMPORTANT: Never mention internal tool names in your responses to the user."
            )
        return hints

    def add_pdf(self, file_id: str, source: PdfSource) -> None:
        """Add a PDF after initial construction."""
        self._pdf_sources[file_id] = source
        try:
            count = _get_page_count(source)
            self._page_counts[file_id] = count
            entries = []
            for i in range(count):
                entry = ImageEntry(
                    id=f"pdf:{file_id}:page:{i + 1}",
                    name=f"{file_id} — Page {i + 1}",
                    source="pdf",
                    page=i + 1,
                    file_id=file_id,
                    _data_getter=lambda s=source, pg=i: _render_page(s, pg),
                )
                entries.append(entry)
            self._registry.register_many(entries)
        except Exception as e:
            logger.warning("[PDF_MCP] Failed to index added PDF %s: %s", file_id, e)

    def remove_pdf(self, file_id: str) -> None:
        """Remove a PDF and its registry entries."""
        self._pdf_sources.pop(file_id, None)
        self._page_counts.pop(file_id, None)
        self._registry.unregister_source("pdf", file_id=file_id)

    @property
    def has_pdfs(self) -> bool:
        return bool(self._pdf_sources)
