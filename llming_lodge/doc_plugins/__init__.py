"""Document plugin system for llming-lodge chat.

Provides a plugin architecture for rich document creation and visualization
within chat conversations. Supports Plotly charts, LaTeX formulas, tables,
text documents, presentations, and HTML sandboxes.
"""

from llming_lodge.doc_plugins.document_store import Document, DocumentSessionStore
from llming_lodge.doc_plugins.manager import DocPluginManager, ALL_DOC_PLUGIN_TYPES
from llming_lodge.doc_plugins.render import (
    EmbedBehavior,
    EMBED_BEHAVIOR,
    RenderResult,
    RenderContext,
    RENDER_CAPABILITIES,
    EMBED_RULES,
    render_to,
    can_render,
    can_embed,
    get_embed_format,
    get_embed_behavior,
    register_embed_behavior,
)

__all__ = [
    "Document",
    "DocumentSessionStore",
    "DocPluginManager",
    "ALL_DOC_PLUGIN_TYPES",
    "EmbedBehavior",
    "EMBED_BEHAVIOR",
    "RenderResult",
    "RenderContext",
    "RENDER_CAPABILITIES",
    "EMBED_RULES",
    "render_to",
    "can_render",
    "can_embed",
    "get_embed_format",
    "get_embed_behavior",
    "register_embed_behavior",
]
