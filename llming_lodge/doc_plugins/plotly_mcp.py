"""PlotlyDocumentMCP -- in-process MCP server for editing Plotly chart documents."""

import json
import logging
from typing import Any, Dict, List

from llming_models.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)


class PlotlyDocumentMCP(InProcessMCPServer):
    """MCP server that lets the LLM inspect and edit Plotly chart documents."""

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "plotly_list_charts",
                "displayName": "List Charts",
                "displayDescription": "List all Plotly chart documents",
                "icon": "bar_chart",
                "description": (
                    "List all Plotly chart documents in the current session. "
                    "Returns id, name, version, and trace count for each chart."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "plotly_get_data",
                "displayName": "Get Chart Data",
                "displayDescription": "Get full data of a Plotly chart",
                "icon": "data_object",
                "description": (
                    "Get the full data and layout of a Plotly chart document. "
                    "Returns the complete {data: [...], layout: {...}} structure."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "plotly_get_trace",
                "displayName": "Get Trace",
                "displayDescription": "Get a specific trace from a chart",
                "icon": "show_chart",
                "description": (
                    "Get a specific trace by its index from a Plotly chart document. "
                    "Returns the full trace object (x, y, type, name, etc.)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                        "trace_index": {
                            "type": "integer",
                            "description": "Zero-based index of the trace to retrieve",
                        },
                    },
                    "required": ["document_id", "trace_index"],
                },
            },
            {
                "name": "plotly_add_trace",
                "displayName": "Add Trace",
                "displayDescription": "Add a new trace to a chart",
                "icon": "add_chart",
                "description": (
                    "Add a new trace to a Plotly chart document. "
                    "The trace object should contain x, y, type, name, and any "
                    "other valid Plotly trace properties."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                        "trace": {
                            "type": "object",
                            "description": (
                                "Plotly trace object with properties like x, y, type, "
                                "name, mode, marker, etc."
                            ),
                        },
                    },
                    "required": ["document_id", "trace"],
                },
            },
            {
                "name": "plotly_remove_trace",
                "displayName": "Remove Trace",
                "displayDescription": "Remove a trace from a chart",
                "icon": "remove_circle",
                "description": (
                    "Remove a trace by its index from a Plotly chart document."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                        "trace_index": {
                            "type": "integer",
                            "description": "Zero-based index of the trace to remove",
                        },
                    },
                    "required": ["document_id", "trace_index"],
                },
            },
            {
                "name": "plotly_update_trace",
                "displayName": "Update Trace",
                "displayDescription": "Update properties of a trace",
                "icon": "edit",
                "description": (
                    "Update specific properties of a trace in a Plotly chart document. "
                    "Only the provided properties are changed; others are preserved."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                        "trace_index": {
                            "type": "integer",
                            "description": "Zero-based index of the trace to update",
                        },
                        "updates": {
                            "type": "object",
                            "description": (
                                "Object with trace properties to update "
                                "(e.g. {name: 'new', marker: {color: 'red'}})"
                            ),
                        },
                    },
                    "required": ["document_id", "trace_index", "updates"],
                },
            },
            {
                "name": "plotly_update_layout",
                "displayName": "Update Layout",
                "displayDescription": "Update the chart layout",
                "icon": "dashboard_customize",
                "description": (
                    "Update the layout of a Plotly chart document. "
                    "The provided layout object is merged with the existing layout, "
                    "so you only need to specify the properties you want to change."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the Plotly chart",
                        },
                        "layout": {
                            "type": "object",
                            "description": (
                                "Layout properties to merge (e.g. {title: 'My Chart', "
                                "xaxis: {title: 'X'}, template: 'plotly_dark'})"
                            ),
                        },
                    },
                    "required": ["document_id", "layout"],
                },
            },
            {
                "name": "plotly_search_data",
                "displayName": "Search Charts",
                "displayDescription": "Search across all Plotly chart data",
                "icon": "search",
                "description": (
                    "Search across all Plotly chart documents for values matching "
                    "a query string. Searches trace names, axis titles, chart titles, "
                    "and data values."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "plotly_list_charts":
            docs = self._store.list_by_type("plotly")
            result = []
            for d in docs:
                data = d.data or {}
                trace_count = len(data.get("data", []))
                result.append({
                    "id": d.id,
                    "name": d.name,
                    "version": d.version,
                    "trace_count": trace_count,
                })
            return json.dumps(result)

        elif name == "plotly_get_data":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            return json.dumps({
                "document_id": doc.id,
                "name": doc.name,
                "version": doc.version,
                "data": doc.data,
            })

        elif name == "plotly_get_trace":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            data = doc.data or {}
            traces = data.get("data", [])
            idx = arguments["trace_index"]
            if idx < 0 or idx >= len(traces):
                return json.dumps({"error": f"Trace index {idx} out of range (0-{len(traces) - 1})"})
            return json.dumps({"trace_index": idx, "trace": traces[idx]})

        elif name == "plotly_add_trace":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            data = doc.data or {}
            if "data" not in data:
                data["data"] = []
            data["data"].append(arguments["trace"])
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "trace_added",
                "document_id": doc.id,
                "trace_index": len(data["data"]) - 1,
                "version": updated.version,
            })

        elif name == "plotly_remove_trace":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            data = doc.data or {}
            traces = data.get("data", [])
            idx = arguments["trace_index"]
            if idx < 0 or idx >= len(traces):
                return json.dumps({"error": f"Trace index {idx} out of range (0-{len(traces) - 1})"})
            removed = traces.pop(idx)
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "trace_removed",
                "document_id": doc.id,
                "removed_trace_name": removed.get("name", ""),
                "version": updated.version,
            })

        elif name == "plotly_update_trace":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            data = doc.data or {}
            traces = data.get("data", [])
            idx = arguments["trace_index"]
            if idx < 0 or idx >= len(traces):
                return json.dumps({"error": f"Trace index {idx} out of range (0-{len(traces) - 1})"})
            traces[idx].update(arguments["updates"])
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "trace_updated",
                "document_id": doc.id,
                "trace_index": idx,
                "version": updated.version,
            })

        elif name == "plotly_update_layout":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "plotly":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'plotly'"})
            data = doc.data or {}
            if "layout" not in data:
                data["layout"] = {}
            data["layout"].update(arguments["layout"])
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "layout_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "plotly_search_data":
            query = arguments["query"].lower()
            docs = self._store.list_by_type("plotly")
            results = []
            for d in docs:
                data = d.data or {}
                matches = []
                # Search layout title
                layout = data.get("layout", {})
                title = layout.get("title")
                if isinstance(title, dict):
                    title = title.get("text", "")
                if title and query in str(title).lower():
                    matches.append({"location": "layout.title", "value": title})
                # Search axis titles
                for axis_key in ("xaxis", "yaxis", "zaxis"):
                    axis = layout.get(axis_key, {})
                    axis_title = axis.get("title")
                    if isinstance(axis_title, dict):
                        axis_title = axis_title.get("text", "")
                    if axis_title and query in str(axis_title).lower():
                        matches.append({"location": f"layout.{axis_key}.title", "value": axis_title})
                # Search traces
                for i, trace in enumerate(data.get("data", [])):
                    trace_name = trace.get("name", "")
                    if trace_name and query in str(trace_name).lower():
                        matches.append({"location": f"data[{i}].name", "value": trace_name})
                    # Search data values
                    for key in ("x", "y", "z", "text", "labels", "values"):
                        vals = trace.get(key)
                        if vals and isinstance(vals, list):
                            for j, v in enumerate(vals):
                                if query in str(v).lower():
                                    matches.append({
                                        "location": f"data[{i}].{key}[{j}]",
                                        "value": v,
                                    })
                if matches:
                    results.append({
                        "document_id": d.id,
                        "name": d.name,
                        "matches": matches,
                    })
            return json.dumps(results)

        return json.dumps({"error": f"Unknown tool: {name}"})
