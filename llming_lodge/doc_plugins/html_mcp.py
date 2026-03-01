"""HtmlDocumentMCP -- in-process MCP server for editing website documents."""

import json
import logging
from typing import Any, Dict, List

from llming_lodge.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)


class HtmlDocumentMCP(InProcessMCPServer):
    """MCP server that lets the LLM inspect and edit website / web app documents."""

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "html_get_source",
                "displayName": "Get Source",
                "displayDescription": "Get the full HTML document source",
                "icon": "code",
                "description": (
                    "Get the full source of a website document. "
                    "Returns an object with html, css, js, and title fields."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "html_get_html",
                "displayName": "Get HTML",
                "displayDescription": "Get just the HTML part",
                "icon": "html",
                "description": (
                    "Get only the HTML markup portion of a website document."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "html_get_css",
                "displayName": "Get CSS",
                "displayDescription": "Get just the CSS part",
                "icon": "css",
                "description": (
                    "Get only the CSS styles portion of a website document."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "html_get_js",
                "displayName": "Get JavaScript",
                "displayDescription": "Get just the JavaScript part",
                "icon": "javascript",
                "description": (
                    "Get only the JavaScript code portion of a website document."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "html_update_html",
                "displayName": "Update HTML",
                "displayDescription": "Update the HTML markup",
                "icon": "edit",
                "description": (
                    "Replace the HTML markup portion of a website document "
                    "with new content. The CSS and JS parts are preserved."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                        "html": {
                            "type": "string",
                            "description": "New HTML markup content",
                        },
                    },
                    "required": ["document_id", "html"],
                },
            },
            {
                "name": "html_update_css",
                "displayName": "Update CSS",
                "displayDescription": "Update the CSS styles",
                "icon": "palette",
                "description": (
                    "Replace the CSS styles portion of a website document "
                    "with new content. The HTML and JS parts are preserved."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                        "css": {
                            "type": "string",
                            "description": "New CSS styles content",
                        },
                    },
                    "required": ["document_id", "css"],
                },
            },
            {
                "name": "html_update_js",
                "displayName": "Update JavaScript",
                "displayDescription": "Update the JavaScript code",
                "icon": "terminal",
                "description": (
                    "Replace the JavaScript code portion of a website "
                    "document with new content. The HTML and CSS parts are preserved."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                        "js": {
                            "type": "string",
                            "description": "New JavaScript code content",
                        },
                    },
                    "required": ["document_id", "js"],
                },
            },
            {
                "name": "html_search",
                "displayName": "Search Source",
                "displayDescription": "Search HTML document source code",
                "icon": "search",
                "description": (
                    "Search the source code of a website document (HTML, "
                    "CSS, and JS) for text matching a query string. Returns "
                    "which parts contain matches and the matching lines."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the HTML document",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query string (case-insensitive)",
                        },
                    },
                    "required": ["document_id", "query"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "html_get_source":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            return json.dumps({
                "document_id": doc.id,
                "name": doc.name,
                "version": doc.version,
                "html": data.get("html", ""),
                "css": data.get("css", ""),
                "js": data.get("js", ""),
                "title": data.get("title", ""),
            })

        elif name == "html_get_html":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            return json.dumps({
                "document_id": doc.id,
                "html": data.get("html", ""),
            })

        elif name == "html_get_css":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            return json.dumps({
                "document_id": doc.id,
                "css": data.get("css", ""),
            })

        elif name == "html_get_js":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            return json.dumps({
                "document_id": doc.id,
                "js": data.get("js", ""),
            })

        elif name == "html_update_html":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            data["html"] = arguments["html"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "html_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "html_update_css":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            data["css"] = arguments["css"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "css_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "html_update_js":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            data["js"] = arguments["js"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "js_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "html_search":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "html":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'html'"})
            data = doc.data or {}
            query = arguments["query"].lower()
            results = []
            for part_name in ("html", "css", "js"):
                source = data.get(part_name, "")
                if not source:
                    continue
                lines = source.split("\n")
                matching_lines = []
                for i, line in enumerate(lines):
                    if query in line.lower():
                        matching_lines.append({
                            "line": i + 1,
                            "content": line.rstrip(),
                        })
                if matching_lines:
                    results.append({
                        "part": part_name,
                        "match_count": len(matching_lines),
                        "lines": matching_lines,
                    })
            # Also search title
            title = data.get("title", "")
            if title and query in title.lower():
                results.append({
                    "part": "title",
                    "match_count": 1,
                    "lines": [{"line": 0, "content": title}],
                })
            return json.dumps({
                "document_id": doc.id,
                "query": arguments["query"],
                "results": results,
                "total_matches": sum(r["match_count"] for r in results),
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
