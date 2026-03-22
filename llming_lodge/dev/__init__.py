"""llming_lodge.dev — Developer tools: dev commands, remote mode, dev MCPs.

Built-in dev command handling for llming-lodge chat sessions.
These are framework-level features, not app-specific.
"""

from llming_lodge.dev.dev_mcp import DevMCPDef, register_dev_mcp, try_activate_dev_mcp, deactivate_all_dev_mcps
from llming_lodge.dev.dev_commands import handle_dev_command
from llming_lodge.dev.remote_mode import activate_remote_mode, deactivate_remote_mode

__all__ = [
    "DevMCPDef",
    "register_dev_mcp",
    "try_activate_dev_mcp",
    "deactivate_all_dev_mcps",
    "handle_dev_command",
    "activate_remote_mode",
    "deactivate_remote_mode",
]
