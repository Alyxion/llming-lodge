"""PresentationMCP -- in-process MCP server for editing presentation slide decks."""

import json
import logging
from typing import Any, Dict, List

from llming_lodge.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)


class PresentationMCP(InProcessMCPServer):
    """MCP server that lets the LLM inspect and edit presentation slide decks."""

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    def _find_slide(self, slides: List[Dict], slide_id: str) -> tuple[int, Dict | None]:
        """Find a slide by ID, returning (index, slide) or (-1, None)."""
        for i, s in enumerate(slides):
            if s.get("id") == slide_id:
                return i, s
        return -1, None

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "pptx_list_slides",
                "displayName": "List Slides",
                "displayDescription": "List all slides in a presentation",
                "icon": "slideshow",
                "description": (
                    "List all slides in a presentation (PowerPoint/PPTX style) document. "
                    "Returns the slide id, title, and element count for each slide."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "pptx_get_slide",
                "displayName": "Get Slide",
                "displayDescription": "Get slide details",
                "icon": "crop_landscape",
                "description": (
                    "Get the full details of a specific slide in a presentation "
                    "(PowerPoint/PPTX style), including all elements. Specify "
                    "either slide_id or a zero-based index."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                        "slide_id": {
                            "type": "string",
                            "description": "ID of the slide to retrieve",
                        },
                        "index": {
                            "type": "integer",
                            "description": "Zero-based index of the slide to retrieve",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "pptx_update_element",
                "displayName": "Update Element",
                "displayDescription": "Update an element on a slide",
                "icon": "edit",
                "description": (
                    "Update a specific element on a slide in a presentation. "
                    "Elements are identified by their zero-based index within "
                    "the slide's elements array."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                        "slide_id": {
                            "type": "string",
                            "description": "ID of the slide containing the element",
                        },
                        "element_index": {
                            "type": "integer",
                            "description": "Zero-based index of the element on the slide",
                        },
                        "updates": {
                            "type": "object",
                            "description": (
                                "Properties to update on the element (e.g. "
                                "{text: 'new text', fontSize: 24, color: '#333'})"
                            ),
                        },
                    },
                    "required": ["document_id", "slide_id", "element_index", "updates"],
                },
            },
            {
                "name": "pptx_add_slide",
                "displayName": "Add Slide",
                "displayDescription": "Add a new slide to the presentation",
                "icon": "add_to_photos",
                "description": (
                    "Add a new slide to a presentation. Provide a "
                    "title and optionally a layout name, elements array, "
                    "placeholders dict (for template-native slides), and "
                    "position to insert at. Use 'placeholders' when the "
                    "presentation uses a template with defined layouts."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the new slide",
                        },
                        "layout": {
                            "type": "string",
                            "description": "Layout name (e.g. 'title', 'text', 'two_columns', 'end')",
                        },
                        "elements": {
                            "type": "array",
                            "description": "Array of element objects (abstract format)",
                        },
                        "placeholders": {
                            "type": "object",
                            "description": (
                                "Dict of placeholder values (template-native format). "
                                "Keys are placeholder names from the template layout. "
                                "Values are strings or objects like "
                                "{type: 'list', items: [...]}, {type: 'table', headers: [], rows: []}."
                            ),
                        },
                        "position": {
                            "type": "integer",
                            "description": "Zero-based position to insert the slide (default: end)",
                        },
                    },
                    "required": ["document_id", "title"],
                },
            },
            {
                "name": "pptx_delete_slide",
                "displayName": "Delete Slide",
                "displayDescription": "Delete a slide from the presentation",
                "icon": "delete",
                "description": "Delete a slide from a presentation by its ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                        "slide_id": {
                            "type": "string",
                            "description": "ID of the slide to delete",
                        },
                    },
                    "required": ["document_id", "slide_id"],
                },
            },
            {
                "name": "pptx_reorder_slides",
                "displayName": "Reorder Slides",
                "displayDescription": "Reorder slides in the presentation",
                "icon": "reorder",
                "description": (
                    "Reorder slides in a presentation. Provide an "
                    "array of slide IDs in the desired new order. All existing "
                    "slide IDs must be included."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
                        },
                        "order": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of slide IDs in the desired order",
                        },
                    },
                    "required": ["document_id", "order"],
                },
            },
            {
                "name": "pptx_search",
                "displayName": "Search Slides",
                "displayDescription": "Search slide content",
                "icon": "search",
                "description": (
                    "Search across all slides in a presentation for "
                    "text matching a query string. Searches slide titles and "
                    "element content."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the presentation",
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
        if name == "pptx_list_slides":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            result = []
            for s in slides:
                entry = {
                    "id": s.get("id", ""),
                    "title": s.get("title", ""),
                    "element_count": len(s.get("elements", [])),
                }
                if s.get("layout"):
                    entry["layout"] = s["layout"]
                if s.get("placeholders"):
                    entry["placeholder_count"] = len(s["placeholders"])
                result.append(entry)
            return json.dumps({
                "document_id": doc.id,
                "slide_count": len(slides),
                "slides": result,
            })

        elif name == "pptx_get_slide":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            slide_id = arguments.get("slide_id")
            index = arguments.get("index")
            slide = None
            if slide_id:
                _, slide = self._find_slide(slides, slide_id)
            elif index is not None:
                if 0 <= index < len(slides):
                    slide = slides[index]
            else:
                return json.dumps({"error": "Must provide either 'slide_id' or 'index'"})
            if not slide:
                return json.dumps({"error": "Slide not found"})
            return json.dumps({
                "document_id": doc.id,
                "slide": slide,
            })

        elif name == "pptx_update_element":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            _, slide = self._find_slide(slides, arguments["slide_id"])
            if not slide:
                return json.dumps({"error": f"Slide '{arguments['slide_id']}' not found"})
            elements = slide.get("elements", [])
            idx = arguments["element_index"]
            if idx < 0 or idx >= len(elements):
                return json.dumps({"error": f"Element index {idx} out of range (0-{len(elements) - 1})"})
            elements[idx].update(arguments["updates"])
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "element_updated",
                "document_id": doc.id,
                "slide_id": arguments["slide_id"],
                "element_index": idx,
                "version": updated.version,
            })

        elif name == "pptx_add_slide":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            if "slides" not in data:
                data["slides"] = []
            slides = data["slides"]
            from uuid import uuid4
            new_slide: Dict[str, Any] = {
                "id": uuid4().hex[:8],
                "title": arguments["title"],
            }
            if arguments.get("placeholders"):
                new_slide["placeholders"] = arguments["placeholders"]
            else:
                new_slide["elements"] = arguments.get("elements", [])
            if arguments.get("layout"):
                new_slide["layout"] = arguments["layout"]
            position = arguments.get("position")
            if position is not None and 0 <= position <= len(slides):
                slides.insert(position, new_slide)
            else:
                slides.append(new_slide)
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "slide_added",
                "document_id": doc.id,
                "slide_id": new_slide["id"],
                "slide_count": len(slides),
                "version": updated.version,
            })

        elif name == "pptx_delete_slide":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            idx, slide = self._find_slide(slides, arguments["slide_id"])
            if slide is None:
                return json.dumps({"error": f"Slide '{arguments['slide_id']}' not found"})
            slides.pop(idx)
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "slide_deleted",
                "document_id": doc.id,
                "deleted_slide_id": arguments["slide_id"],
                "slide_count": len(slides),
                "version": updated.version,
            })

        elif name == "pptx_reorder_slides":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            order = arguments["order"]
            # Build lookup
            slide_map = {s.get("id"): s for s in slides}
            existing_ids = set(slide_map.keys())
            order_ids = set(order)
            if order_ids != existing_ids:
                missing = existing_ids - order_ids
                extra = order_ids - existing_ids
                errors = []
                if missing:
                    errors.append(f"missing: {list(missing)}")
                if extra:
                    errors.append(f"unknown: {list(extra)}")
                return json.dumps({"error": f"Order must contain all slide IDs. {', '.join(errors)}"})
            data["slides"] = [slide_map[sid] for sid in order]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "slides_reordered",
                "document_id": doc.id,
                "slide_count": len(data["slides"]),
                "version": updated.version,
            })

        elif name == "pptx_search":
            doc = self._store.get(arguments["document_id"])
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type not in ("presentation", "powerpoint"):
                return json.dumps({"error": f"Document is type '{doc.type}', not 'presentation'"})
            data = doc.data or {}
            slides = data.get("slides", [])
            query = arguments["query"].lower()
            matches = []
            for s in slides:
                slide_matches = []
                title = s.get("title", "")
                if query in str(title).lower():
                    slide_matches.append({"location": "title", "value": title})
                for i, elem in enumerate(s.get("elements", [])):
                    elem_str = json.dumps(elem) if isinstance(elem, (dict, list)) else str(elem)
                    if query in elem_str.lower():
                        slide_matches.append({
                            "location": f"elements[{i}]",
                            "value": elem.get("text", elem_str[:100]) if isinstance(elem, dict) else elem_str[:100],
                        })
                if slide_matches:
                    matches.append({
                        "slide_id": s.get("id", ""),
                        "title": title,
                        "matches": slide_matches,
                    })
            return json.dumps({
                "document_id": doc.id,
                "query": arguments["query"],
                "slides_matched": len(matches),
                "matches": matches,
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
