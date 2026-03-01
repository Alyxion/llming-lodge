"""DocumentCreatorMCP — in-process MCP server for creating and managing documents."""

import json
import logging
from typing import Any, Dict, List

from llming_lodge.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)

# Supported document types
DOC_TYPES = ["plotly", "latex", "table", "text_doc", "presentation", "html", "email_draft",
             "word", "powerpoint"]  # old names accepted for backward compat


class DocumentCreatorMCP(InProcessMCPServer):
    """MCP server that lets the LLM create, list, get, and delete documents."""

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "create_document",
                "displayName": "Create Document",
                "displayDescription": "Create a rich document",
                "icon": "description",
                "description": (
                    "Create a new document that will be rendered in the chat. "
                    "Supported types: plotly, latex, table, text_doc, presentation, html. "
                    "The data format depends on the type:\n"
                    "- plotly: {data: [...], layout: {...}}\n"
                    "- latex: {formula: '...'}\n"
                    "- table: {columns: [...], rows: [...]}  (for spreadsheets / data tables)\n"
                    "- text_doc: {sections: [{id, type, content, ...}]}  (for text documents / DOCX)\n"
                    "- presentation: {slides: [{id, title, elements: [...]}]}  (for presentations; optional \"template\" field for branded styling)\n"
                    "- html: {html: '', css: '', js: '', title: ''}\n"
                    "- email_draft: {subject: '', to: [...], cc: [...], bcc: [...], body_html: '', attachments: [{ref, name}]}\n"
                    "IMPORTANT: Instead of using this tool, you can also produce fenced "
                    "code blocks with the type as language (e.g. ```plotly ... ```) for "
                    "inline rendering. Use this tool when the user wants a persistent "
                    "document they can manage in the document panel."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": DOC_TYPES,
                            "description": "Document type",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable document name",
                        },
                        "data": {
                            "type": "string",
                            "description": "Document data as a JSON string (format depends on type)",
                        },
                    },
                    "required": ["type", "name", "data"],
                },
            },
            {
                "name": "list_documents",
                "displayName": "List Documents",
                "displayDescription": "List all documents",
                "icon": "folder_open",
                "description": (
                    "List all documents in the current conversation. "
                    "Optionally filter by type. Returns id, type, name, version."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": DOC_TYPES,
                            "description": "Filter by document type (optional)",
                        },
                    },
                },
            },
            {
                "name": "get_document",
                "displayName": "Get Document",
                "displayDescription": "Get document details",
                "icon": "article",
                "description": "Get the full data of a specific document by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "delete_document",
                "displayName": "Delete Document",
                "displayDescription": "Delete a document",
                "icon": "delete",
                "description": "Delete a document by ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID",
                        },
                    },
                    "required": ["document_id"],
                },
            },
        ]

    # Backward compat: old type names → new
    _TYPE_ALIASES = {"word": "text_doc", "powerpoint": "presentation"}

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "create_document":
            data = arguments["data"]
            if isinstance(data, str):
                data = json.loads(data)
            doc_type = self._TYPE_ALIASES.get(arguments["type"], arguments["type"])
            doc = self._store.create(
                type=doc_type,
                name=arguments["name"],
                data=data,
            )
            return json.dumps({
                "status": "created",
                "document_id": doc.id,
                "type": doc.type,
                "name": doc.name,
                "version": doc.version,
            })

        elif name == "list_documents":
            doc_type = arguments.get("type")
            docs = self._store.list_by_type(doc_type) if doc_type else self._store.list_all()
            return json.dumps([
                {"id": d.id, "type": d.type, "name": d.name, "version": d.version}
                for d in docs
            ])

        elif name == "get_document":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            return json.dumps(doc.model_dump())

        elif name == "delete_document":
            deleted = self._store.delete(arguments["document_id"])
            return json.dumps({
                "status": "deleted" if deleted else "not_found",
                "document_id": arguments["document_id"],
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
