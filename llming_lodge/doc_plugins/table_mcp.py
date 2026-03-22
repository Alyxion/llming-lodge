"""TableDocumentMCP -- in-process MCP server for editing table documents."""

import json
import logging
from typing import Any, Dict, List

from llming_models.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)


class TableDocumentMCP(InProcessMCPServer):
    """MCP server that lets the LLM inspect and edit table documents."""

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "table_get_rows",
                "displayName": "Get Rows",
                "displayDescription": "Get rows from a table document",
                "icon": "table_rows",
                "description": (
                    "Get rows from a table / spreadsheet document. Returns columns and "
                    "row data. Optionally specify a start index and count to paginate "
                    "through large tables. Use for tabular data, spreadsheets, and "
                    "Excel-style content."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "start": {
                            "type": "integer",
                            "description": "Zero-based starting row index (default: 0)",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of rows to return (default: all remaining)",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "table_update_cells",
                "displayName": "Update Cells",
                "displayDescription": "Update specific cells in a table",
                "icon": "edit",
                "description": (
                    "Update specific cells in a table document. Provide an array of "
                    "updates, each specifying the row index, column index, and new value."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "row": {"type": "integer", "description": "Zero-based row index"},
                                    "col": {"type": "integer", "description": "Zero-based column index"},
                                    "value": {"type": "string", "description": "New cell value"},
                                },
                                "required": ["row", "col", "value"],
                            },
                            "description": "Array of cell updates",
                        },
                    },
                    "required": ["document_id", "updates"],
                },
            },
            {
                "name": "table_add_rows",
                "displayName": "Add Rows",
                "displayDescription": "Add rows to the end of a table",
                "icon": "playlist_add",
                "description": (
                    "Add one or more rows to the end of a table document. "
                    "Each row is an array of cell values matching the column order."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "rows": {
                            "type": "array",
                            "items": {"type": "array"},
                            "description": "Array of rows to add, each row is an array of values",
                        },
                    },
                    "required": ["document_id", "rows"],
                },
            },
            {
                "name": "table_delete_rows",
                "displayName": "Delete Rows",
                "displayDescription": "Delete rows by indices",
                "icon": "delete_sweep",
                "description": (
                    "Delete rows from a table document by their indices. "
                    "Indices are zero-based. Rows are removed in reverse order "
                    "so that indices remain stable during deletion."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Array of zero-based row indices to delete",
                        },
                    },
                    "required": ["document_id", "indices"],
                },
            },
            {
                "name": "table_add_column",
                "displayName": "Add Column",
                "displayDescription": "Add a column to a table",
                "icon": "view_column",
                "description": (
                    "Add a new column to a table document. Optionally provide a "
                    "default value for existing rows and a position to insert at."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of the new column",
                        },
                        "default_value": {
                            "type": "string",
                            "description": "Default value for existing rows (default: empty string)",
                        },
                        "position": {
                            "type": "integer",
                            "description": "Zero-based position to insert the column (default: end)",
                        },
                    },
                    "required": ["document_id", "name"],
                },
            },
            {
                "name": "table_remove_column",
                "displayName": "Remove Column",
                "displayDescription": "Remove a column from a table",
                "icon": "remove_circle",
                "description": (
                    "Remove a column from a table document by its index or name. "
                    "Provide either 'index' (zero-based) or 'name'."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "index": {
                            "type": "integer",
                            "description": "Zero-based column index to remove",
                        },
                        "name": {
                            "type": "string",
                            "description": "Column name to remove",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "table_search",
                "displayName": "Search Table",
                "displayDescription": "Search table data for matching values",
                "icon": "search",
                "description": (
                    "Search a table document for cells matching a query string. "
                    "Returns matching rows with their indices and the matching cells."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query string (case-insensitive)",
                        },
                    },
                    "required": ["document_id", "query"],
                },
            },
            {
                "name": "table_sort",
                "displayName": "Sort Table",
                "displayDescription": "Sort table rows by a column",
                "icon": "sort",
                "description": (
                    "Sort a table document by a specific column. The column can be "
                    "specified by name or zero-based index. Sorts ascending by default."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the table",
                        },
                        "column": {
                            "description": "Column name (string) or zero-based index (integer) to sort by",
                        },
                        "descending": {
                            "type": "boolean",
                            "description": "Sort in descending order (default: false)",
                        },
                    },
                    "required": ["document_id", "column"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "table_get_rows":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            start = arguments.get("start", 0)
            count = arguments.get("count")
            end = start + count if count is not None else len(rows)
            return json.dumps({
                "document_id": doc.id,
                "columns": columns,
                "rows": rows[start:end],
                "total_rows": len(rows),
                "start": start,
                "returned": len(rows[start:end]),
            })

        elif name == "table_update_cells":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            rows = data.get("rows", [])
            updates = arguments["updates"]
            applied = 0
            for u in updates:
                r, c, v = u["row"], u["col"], u["value"]
                if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
                    rows[r][c] = v
                    applied += 1
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "cells_updated",
                "document_id": doc.id,
                "applied": applied,
                "total_updates": len(updates),
                "version": updated.version,
            })

        elif name == "table_add_rows":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            if "rows" not in data:
                data["rows"] = []
            new_rows = arguments["rows"]
            data["rows"].extend(new_rows)
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "rows_added",
                "document_id": doc.id,
                "added": len(new_rows),
                "total_rows": len(data["rows"]),
                "version": updated.version,
            })

        elif name == "table_delete_rows":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            rows = data.get("rows", [])
            indices = sorted(arguments["indices"], reverse=True)
            deleted = 0
            for idx in indices:
                if 0 <= idx < len(rows):
                    rows.pop(idx)
                    deleted += 1
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "rows_deleted",
                "document_id": doc.id,
                "deleted": deleted,
                "total_rows": len(rows),
                "version": updated.version,
            })

        elif name == "table_add_column":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            col_name = arguments["name"]
            default_value = arguments.get("default_value", "")
            position = arguments.get("position")
            if position is not None and 0 <= position <= len(columns):
                columns.insert(position, col_name)
                for row in rows:
                    row.insert(position, default_value)
            else:
                columns.append(col_name)
                for row in rows:
                    row.append(default_value)
            data["columns"] = columns
            data["rows"] = rows
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "column_added",
                "document_id": doc.id,
                "column_name": col_name,
                "column_count": len(columns),
                "version": updated.version,
            })

        elif name == "table_remove_column":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            col_idx = arguments.get("index")
            col_name = arguments.get("name")
            if col_idx is None and col_name is not None:
                if col_name in columns:
                    col_idx = columns.index(col_name)
                else:
                    return json.dumps({"error": f"Column '{col_name}' not found"})
            if col_idx is None:
                return json.dumps({"error": "Must provide either 'index' or 'name'"})
            if col_idx < 0 or col_idx >= len(columns):
                return json.dumps({"error": f"Column index {col_idx} out of range (0-{len(columns) - 1})"})
            removed_name = columns.pop(col_idx)
            for row in rows:
                if col_idx < len(row):
                    row.pop(col_idx)
            data["columns"] = columns
            data["rows"] = rows
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "column_removed",
                "document_id": doc.id,
                "removed_column": removed_name,
                "column_count": len(columns),
                "version": updated.version,
            })

        elif name == "table_search":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            query = arguments["query"].lower()
            matches = []
            for i, row in enumerate(rows):
                row_matches = []
                for j, cell in enumerate(row):
                    if query in str(cell).lower():
                        col_name = columns[j] if j < len(columns) else f"col_{j}"
                        row_matches.append({
                            "col": j,
                            "column_name": col_name,
                            "value": cell,
                        })
                if row_matches:
                    matches.append({
                        "row": i,
                        "cells": row_matches,
                    })
            return json.dumps({
                "document_id": doc.id,
                "query": arguments["query"],
                "matches": matches,
                "total_matches": sum(len(m["cells"]) for m in matches),
            })

        elif name == "table_sort":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "table":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'table'"})
            data = doc.data or {}
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            column = arguments["column"]
            descending = arguments.get("descending", False)
            # Resolve column to index
            if isinstance(column, str):
                if column in columns:
                    col_idx = columns.index(column)
                else:
                    return json.dumps({"error": f"Column '{column}' not found"})
            else:
                col_idx = int(column)
            if col_idx < 0 or col_idx >= len(columns):
                return json.dumps({"error": f"Column index {col_idx} out of range (0-{len(columns) - 1})"})

            def sort_key(row):
                val = row[col_idx] if col_idx < len(row) else ""
                # Try numeric sort, fall back to string
                try:
                    return (0, float(val))
                except (ValueError, TypeError):
                    return (1, str(val).lower())

            rows.sort(key=sort_key, reverse=descending)
            data["rows"] = rows
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "sorted",
                "document_id": doc.id,
                "column": columns[col_idx],
                "descending": descending,
                "version": updated.version,
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
