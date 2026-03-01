"""ImageRegistry — source-agnostic registry of images available to the LLM.

Tracks images from any source (PDF pages, conversation history, droplets,
projects, etc.) with metadata so the LLM can decide when to use them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageEntry:
    """A single image available in the registry."""
    id: str               # unique id, e.g. "pdf:invoice.pdf:page:3"
    name: str             # display name, e.g. "invoice.pdf — Page 3"
    source: str           # source type: "pdf", "conversation", "droplet", ...
    description: str = "" # optional detailed description for LLM decision-making
    page: int = 0         # optional page number (for paginated sources)
    file_id: str = ""     # optional file identifier this image belongs to
    _data_getter: Optional[Callable[[], bytes]] = field(default=None, repr=False)

    def get_data(self) -> bytes:
        """Get the JPEG image bytes (lazy-loaded)."""
        if self._data_getter is None:
            raise ValueError(f"No data getter for image {self.id}")
        return self._data_getter()


class ImageRegistry:
    """Registry of images available to the LLM session.

    Images are registered from various sources and can be listed/retrieved
    by the LLM through MCP tools. Designed to be extended with new sources
    (droplets, projects, etc.) without changing the tool interface.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ImageEntry] = {}

    def register(self, entry: ImageEntry) -> None:
        """Register an image entry."""
        self._entries[entry.id] = entry

    def register_many(self, entries: list[ImageEntry]) -> None:
        """Register multiple entries at once."""
        for entry in entries:
            self._entries[entry.id] = entry

    def unregister(self, image_id: str) -> None:
        """Remove a single image by ID."""
        self._entries.pop(image_id, None)

    def unregister_source(self, source: str, file_id: str = "") -> None:
        """Remove all images from a given source (and optionally file_id)."""
        to_remove = [
            eid for eid, e in self._entries.items()
            if e.source == source and (not file_id or e.file_id == file_id)
        ]
        for eid in to_remove:
            del self._entries[eid]

    def list_entries(self, source: str = "") -> list[ImageEntry]:
        """List all entries, optionally filtered by source."""
        if source:
            return [e for e in self._entries.values() if e.source == source]
        return list(self._entries.values())

    def get(self, image_id: str) -> Optional[ImageEntry]:
        """Get a single entry by ID."""
        return self._entries.get(image_id)

    def has_images(self) -> bool:
        return bool(self._entries)

    def get_prompt_hints(self) -> list[str]:
        """Generate prompt hint lines describing available images."""
        if not self._entries:
            return []

        by_source: dict[str, list[ImageEntry]] = {}
        for e in self._entries.values():
            by_source.setdefault(e.source, []).append(e)

        hints = []
        for source, entries in by_source.items():
            if source == "pdf":
                # Group by file_id
                by_file: dict[str, list[ImageEntry]] = {}
                for e in entries:
                    by_file.setdefault(e.file_id or "unknown", []).append(e)
                for fid, file_entries in by_file.items():
                    pages = sorted(e.page for e in file_entries)
                    hints.append(
                        f"PDF '{fid}': {len(pages)} page(s) available as images "
                        f"(pages {pages[0]}–{pages[-1]}). "
                        f"Use pdf_get_pages to inspect visually."
                    )
            else:
                hints.append(f"{source}: {len(entries)} image(s) available.")

        return hints

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
