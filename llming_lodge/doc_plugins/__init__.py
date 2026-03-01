"""Document plugin system for llming-lodge chat.

Provides a plugin architecture for rich document creation and visualization
within chat conversations. Supports Plotly charts, LaTeX formulas, tables,
text documents, presentations, and HTML sandboxes.
"""

from llming_lodge.doc_plugins.document_store import Document, DocumentSessionStore
from llming_lodge.doc_plugins.manager import DocPluginManager, ALL_DOC_PLUGIN_TYPES

__all__ = ["Document", "DocumentSessionStore", "DocPluginManager", "ALL_DOC_PLUGIN_TYPES"]
