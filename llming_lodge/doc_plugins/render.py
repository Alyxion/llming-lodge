"""Unified document rendering and embedding system.

Provides a single ``render_to(doc_type, spec, target_format, context)`` API
that guarantees all documents are properly rendered to binary format for
downloads, email attachments, and cross-document embedding.

The ``EmbedBehavior`` registry declares how each document type behaves when
embedded inside another document (e.g. a Plotly chart inside a Word doc).
Host documents use a generic ``{"type": "embed", "$ref": "<id>"}`` section —
they never need to know *what* they are embedding.  The registry is extensible:
new document types just add an entry.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Embed behavior registry ──────────────────────────────────────────────


@dataclass(frozen=True)
class EmbedBehavior:
    """Declares how a document type behaves when embedded in another document.

    Attributes:
        mode: How the embedded object is represented in the host document.
            ``"graphic"`` — rasterised to an image (PNG).  Needs client-side
                rendering (Plotly.toImage, html2canvas, …).
            ``"table"``   — embedded as a native table (headers + rows).
            ``"text"``    — inlined as text paragraphs / sections.
        aspect: Width / height ratio for *graphic* embeds.
            ``None`` means the graphic has no preferred ratio (fixed/intrinsic
            size — e.g. a LaTeX formula).  Ignored for non-graphic modes.
    """

    mode: str  # "graphic" | "table" | "text"
    aspect: float | None = None


# Extensible registry — add new doc types here (or at runtime via
# ``register_embed_behavior``).  Host documents never reference this
# directly; the client and export pipeline look it up automatically.
EMBED_BEHAVIOR: dict[str, EmbedBehavior] = {
    "plotly":       EmbedBehavior(mode="graphic", aspect=1.6),
    "table":        EmbedBehavior(mode="table"),
    "text_doc":     EmbedBehavior(mode="text"),
    "html_sandbox": EmbedBehavior(mode="graphic", aspect=1.6),
    "html":         EmbedBehavior(mode="graphic", aspect=1.6),
    "presentation": EmbedBehavior(mode="graphic", aspect=16 / 9),
    "email_draft":  EmbedBehavior(mode="text"),
    "latex":        EmbedBehavior(mode="graphic", aspect=None),
    "rich_mcp":     EmbedBehavior(mode="graphic", aspect=1.6),
}


def register_embed_behavior(doc_type: str, behavior: EmbedBehavior) -> None:
    """Register (or override) the embed behavior for *doc_type*.

    Call this at import time from new doc-plugin modules to make them
    embeddable without touching this file.
    """
    EMBED_BEHAVIOR[doc_type] = behavior


def get_embed_behavior(doc_type: str) -> EmbedBehavior | None:
    """Look up the embed behavior for *doc_type*.

    Returns ``None`` if the type has no registered behavior (not embeddable).
    """
    return EMBED_BEHAVIOR.get(doc_type)


# ── Render capabilities & results ────────────────────────────────────────


@dataclass(frozen=True)
class RenderResult:
    """Result of rendering a document to a target format."""

    data: bytes
    content_type: str
    filename: str


# Which formats each doc type can be rendered to.
RENDER_CAPABILITIES: dict[str, set[str]] = {
    "plotly": {"png"},
    "table": {"xlsx", "csv"},
    "text_doc": {"docx"},
    "presentation": {"pptx"},
    "html": {"html"},
    "html_sandbox": {"html"},
}

# Embedding rules: source_type → {target_type → rendered_format}
# E.g. a plotly chart embedded into a PPTX is rendered as PNG first.
EMBED_RULES: dict[str, dict[str, str]] = {
    "plotly": {
        "pptx": "png",
        "docx": "png",
        "email": "png",
        "html": "iframe",
    },
    "table": {
        "pptx": "native",
        "docx": "native",
        "email": "xlsx",
        "html": "html",
    },
    "html_sandbox": {
        "pptx": "png",
        "docx": "png",
        "email": "html",
    },
    "text_doc": {
        "email": "docx",
    },
    "presentation": {
        "email": "pptx",
    },
    "email_draft": {},
}

# Content types for rendered formats.
_CONTENT_TYPES: dict[str, str] = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "html": "text/html",
    "png": "image/png",
}

_EXTENSIONS: dict[str, str] = {
    "docx": ".docx",
    "pptx": ".pptx",
    "xlsx": ".xlsx",
    "csv": ".csv",
    "html": ".html",
    "png": ".png",
}


def can_render(doc_type: str, target_format: str) -> bool:
    """Check whether *doc_type* can be rendered to *target_format*."""
    caps = RENDER_CAPABILITIES.get(doc_type, set())
    return target_format in caps


def can_embed(source_type: str, target_type: str) -> bool:
    """Check whether *source_type* can be embedded into *target_type*."""
    rules = EMBED_RULES.get(source_type, {})
    return target_type in rules


def get_embed_format(source_type: str, target_type: str) -> str | None:
    """Return the intermediate format needed to embed *source_type* into *target_type*.

    Returns ``None`` if embedding is not supported.
    """
    return EMBED_RULES.get(source_type, {}).get(target_type)


@dataclass
class RenderContext:
    """Context passed to render_to for template/session-specific data."""

    chart_images: dict[str, str] | None = None
    """Pre-rendered chart images (chart_id → base64 PNG data URI)."""

    template_path: str | None = None
    """Path to PPTX template file (for presentation export)."""

    template_config: dict | None = None
    """Serialized presentation template config (for presentation export)."""

    session_id: str | None = None
    """Active session ID (for session-scoped lookups)."""


def render_to(
    doc_type: str,
    spec: dict,
    target_format: str,
    context: RenderContext | None = None,
) -> RenderResult:
    """Render a document spec to a binary format.

    Args:
        doc_type: Document type (``plotly``, ``table``, ``text_doc``,
            ``presentation``, ``html``, ``html_sandbox``).
        spec: The document's JSON spec.
        target_format: Target format (``docx``, ``pptx``, ``xlsx``, ``csv``,
            ``html``, ``png``).
        context: Optional rendering context with chart images, template info, etc.

    Returns:
        A :class:`RenderResult` with binary data, content type, and filename.

    Raises:
        ValueError: If the doc_type/format combination is unsupported.
    """
    ctx = context or RenderContext()

    if not can_render(doc_type, target_format):
        raise ValueError(
            f"Cannot render '{doc_type}' to '{target_format}'. "
            f"Supported formats: {RENDER_CAPABILITIES.get(doc_type, set())}"
        )

    title = (spec.get("title") or "document").replace("/", "_").strip() or "document"
    content_type = _CONTENT_TYPES[target_format]
    ext = _EXTENSIONS[target_format]
    filename = f"{title}{ext}"

    data = _dispatch(doc_type, spec, target_format, ctx)

    return RenderResult(data=data, content_type=content_type, filename=filename)


def _dispatch(
    doc_type: str,
    spec: dict,
    target_format: str,
    ctx: RenderContext,
) -> bytes:
    """Dispatch to the appropriate exporter."""
    if doc_type == "text_doc" and target_format == "docx":
        from llming_lodge.doc_plugins.word_exporter import export_docx

        return export_docx(spec, chart_images=ctx.chart_images)

    if doc_type == "presentation" and target_format == "pptx":
        from llming_lodge.doc_plugins.pptx_exporter import export_pptx

        if not ctx.template_path:
            raise ValueError("Presentation export requires a template_path in context")
        return export_pptx(
            spec,
            template_path=ctx.template_path,
            chart_images=ctx.chart_images,
            template_config=ctx.template_config,
        )

    if doc_type == "table" and target_format == "xlsx":
        from llming_lodge.doc_plugins.table_exporter import export_xlsx

        return export_xlsx(spec)

    if doc_type == "table" and target_format == "csv":
        from llming_lodge.doc_plugins.table_exporter import export_csv

        return export_csv(spec)

    if doc_type in ("html", "html_sandbox") and target_format == "html":
        from llming_lodge.doc_plugins.html_exporter import export_html

        return export_html(spec)

    if doc_type == "plotly" and target_format == "png":
        # Plotly PNG is pre-rendered client-side — pass through from context
        images = ctx.chart_images or {}
        # Look for image by spec ID or first available
        img_id = spec.get("id") or spec.get("_chartImageId", "")
        img_data = images.get(img_id, "")
        if not img_data and images:
            # Fallback: use first available image
            img_data = next(iter(images.values()))
        if not img_data:
            raise ValueError("Plotly PNG export requires pre-rendered chart_images in context")
        # Strip data URI prefix if present
        import base64

        if img_data.startswith("data:"):
            img_data = img_data.split(",", 1)[1]
        return base64.b64decode(img_data)

    raise ValueError(f"No exporter for '{doc_type}' → '{target_format}'")
