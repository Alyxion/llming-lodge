"""Dev MCP registry — secret developer-only MCP servers for testing.

Dev MCPs are never loaded by default. They are activated per-session via
secret chat commands (e.g. ``/dev enable mcp:sapperlot``) and cleaned up
on "New Chat".

Usage in the host app::

    from llming_lodge.dev import register_dev_mcp, DevMCPDef

    register_dev_mcp(DevMCPDef(
        secret_command="/dev enable mcp:sapperlot",
        label="SAP Simulation",
        description="Simulated SAP data (DuckDB)",
        factory=lambda: SAPSimulationMCPServer(),
        activation_message="SAP Simulation activated.",
    ))
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class DevMCPDef:
    """Definition of a dev-only MCP server."""
    secret_command: str        # e.g. "/dev enable mcp:sapperlot"
    label: str                 # e.g. "SAP Simulation"
    description: str
    factory: Callable          # () -> InProcessMCPServer
    category: str = "Dev"
    enabled_by_default: bool = True
    activation_message: str = ""


_REGISTRY: list[DevMCPDef] = []


def register_dev_mcp(defn: DevMCPDef) -> None:
    """Register a dev MCP definition (metadata only -- no instance created).

    Idempotent: skips if a definition with the same label already exists.
    """
    if any(d.label == defn.label for d in _REGISTRY):
        return
    _REGISTRY.append(defn)
    logger.info("[DEV_MCP] Registered: %s (command: %s)", defn.label, defn.secret_command)


async def try_activate_dev_mcp(text: str, controller) -> Optional[str]:
    """Match text against registered dev MCP commands.

    Returns activation message string if matched (caller should intercept),
    or None if no match (message passes through to LLM).
    """
    from llming_models.tools.tool_definition import MCPServerConfig

    normalized = text.strip().lower()
    for defn in _REGISTRY:
        if normalized != defn.secret_command.lower():
            continue

        # Already active?
        active: set = getattr(controller, "_active_dev_mcps", set())
        if defn.label in active:
            return f"{defn.label} is already active."

        # Create instance and register
        server = defn.factory()
        mcp_config = MCPServerConfig(
            server_instance=server,
            label=defn.label,
            description=defn.description,
            category=defn.category,
            enabled_by_default=defn.enabled_by_default,
        )

        from llming_lodge.api.chat_session_api import _register_mcp_tools
        await _register_mcp_tools(controller, mcp_config)

        # Mark group as hidden so tools never appear in the UI settings panel
        server_groups = getattr(controller.session, '_mcp_server_groups', {})
        if defn.label in server_groups:
            server_groups[defn.label]["hidden"] = True

        # Append new prompt hints to context_preamble
        new_hints = controller.session.mcp_prompt_hints
        if new_hints:
            hint_block = "\n\n".join(new_hints)
            controller._mcp_prompt_hints_block = hint_block
            base = controller.context_preamble or ""
            if hint_block not in base:
                controller.context_preamble = base + "\n\n" + hint_block
                controller.session._context_preamble = controller.context_preamble

        # Track activation
        if not hasattr(controller, "_active_dev_mcps"):
            controller._active_dev_mcps = set()
        controller._active_dev_mcps.add(defn.label)

        # Send tools_updated so the frontend shows the new tools
        if controller._ws:
            await controller._send({
                "type": "tools_updated",
                "tools": controller.get_all_known_tools(),
            })

        logger.info("[DEV_MCP] Activated: %s", defn.label)
        return defn.activation_message or f"{defn.label} activated."

    return None


async def deactivate_all_dev_mcps(controller) -> None:
    """Unregister all active dev MCPs (called on new chat)."""
    active: set = getattr(controller, "_active_dev_mcps", set())
    if not active:
        return

    from llming_lodge.api.chat_session_api import _unregister_mcp_tools

    for label in list(active):
        await _unregister_mcp_tools(controller, label)
        logger.info("[DEV_MCP] Deactivated: %s", label)

    controller._active_dev_mcps = set()

    # Notify frontend
    if controller._ws:
        await controller._send({
            "type": "tools_updated",
            "tools": controller.get_all_known_tools(),
        })
