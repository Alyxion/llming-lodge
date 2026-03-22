"""Built-in system nudges — shipped in code, not stored in MongoDB.

Registry pattern: any module can call ``register_system_nudge()`` to add
system nudges.  llming-lodge provides the infrastructure; host apps
register their own nudges at startup.

UIDs use the ``sys:`` prefix so they never collide with MongoDB nudges.
System nudges appear in the nudge explorer and are filterable by name
and category like regular nudges.

Usage from a host app::

    from llming_lodge.system_nudges import register_system_nudge

    register_system_nudge("math", {
        "name": "Math Assistant",
        "category": "tools",
        ...
    })

Control which system nudges are active via
``ChatAppConfig.system_nudges`` (list of keys to enable, or ``None``
for all registered).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

SYSTEM_NUDGE_REGISTRY: dict[str, dict[str, Any]] = {}

UID_PREFIX = "sys:"


def register_system_nudge(key: str, nudge: dict[str, Any]) -> None:
    """Register a system nudge under *key*.

    The uid is forced to ``sys:<key>`` and mode to ``live``.
    Can be called from any module — llming-lodge core, host apps,
    plugins, etc.
    """
    nudge = dict(nudge)  # shallow copy
    nudge["uid"] = f"{UID_PREFIX}{key}"
    nudge["mode"] = "live"
    nudge["is_system"] = True
    nudge.setdefault("visibility", ["*"])
    nudge.setdefault("creator_email", "system")
    nudge.setdefault("creator_name", "System")
    nudge.setdefault("version", "1.0.0")
    nudge.setdefault("updated_at", "")
    nudge.setdefault("created_at", "")
    nudge.setdefault("files", [])
    nudge.setdefault("doc_plugins", [])
    nudge.setdefault("translations", {})
    nudge.setdefault("category", "")
    nudge.setdefault("sub_category", "")
    nudge.setdefault("suggestions", [])
    nudge.setdefault("capabilities", {})
    nudge.setdefault("auto_discover", False)
    nudge.setdefault("bolts", [])
    SYSTEM_NUDGE_REGISTRY[key] = nudge
    logger.info("[SYSTEM_NUDGE] Registered '%s' (uid=%s, category=%s)",
                key, nudge["uid"], nudge.get("category", ""))


def is_system_nudge_uid(uid: str) -> bool:
    """Check if a uid belongs to a system nudge."""
    return uid.startswith(UID_PREFIX)


def get_system_nudge(uid: str) -> dict[str, Any] | None:
    """Look up a system nudge by its full uid (``sys:key``)."""
    if not uid.startswith(UID_PREFIX):
        return None
    key = uid[len(UID_PREFIX):]
    return SYSTEM_NUDGE_REGISTRY.get(key)


def get_enabled_system_nudges(enabled_keys: list[str] | None) -> list[dict[str, Any]]:
    """Return system nudge dicts for the given keys.

    If *enabled_keys* is ``None``, all registered nudges are returned.
    If ``[]``, none are returned.
    """
    if enabled_keys is None:
        return list(SYSTEM_NUDGE_REGISTRY.values())
    return [
        SYSTEM_NUDGE_REGISTRY[k]
        for k in enabled_keys
        if k in SYSTEM_NUDGE_REGISTRY
    ]


def search_system_nudges(
    enabled_keys: list[str] | None,
    *,
    query: str = "",
    category: str = "",
) -> list[dict[str, Any]]:
    """Filter system nudges by text query and/or category.

    Matches against name, description, and translations (case-insensitive).
    """
    nudges = get_enabled_system_nudges(enabled_keys)
    if category and category != "general":
        nudges = [n for n in nudges if n.get("category") == category]
    if query:
        try:
            pat = re.compile(query, re.IGNORECASE)
        except re.error:
            pat = re.compile(re.escape(query), re.IGNORECASE)
        filtered = []
        for n in nudges:
            # Search in name, description, and all translated names/descriptions
            searchable = [n.get("name", ""), n.get("description", "")]
            for tr in (n.get("translations") or {}).values():
                if isinstance(tr, dict):
                    searchable.append(tr.get("name", ""))
                    searchable.append(tr.get("description", ""))
            if any(pat.search(s) for s in searchable if s):
                filtered.append(n)
        nudges = filtered
    return nudges


def _meta(nudge: dict[str, Any]) -> dict[str, Any]:
    """Return metadata-only dict (strip files) for search results."""
    return {k: v for k, v in nudge.items() if k not in ("files", "_id")}
