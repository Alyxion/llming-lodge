"""Browser-hosted MCP connection via WebSocket + Web Worker.

Proxies MCP tool calls through the WebSocket to a browser-side Web Worker
that executes the MCP server's JavaScript code. The Worker is spawned from
nudge JS files stored in MongoDB.
"""
import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)


class MCPBrowserConnection:
    """Proxies MCP tool calls through WebSocket to a browser Web Worker.

    Each instance manages one activated MCP nudge. Tool calls are forwarded
    as WebSocket messages and resolved via asyncio Futures stored in the
    session's ``pending_requests`` dict.
    """

    def __init__(self, nudge_uid: str, session_ctx: dict):
        """
        Args:
            nudge_uid: UID of the MCP nudge this connection serves.
            session_ctx: The session's ``_browser_mcp_sessions`` entry containing
                ``controller``, ``pending_requests``, ``loop``, etc.
        """
        self.nudge_uid = nudge_uid
        self._ctx = session_ctx

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Forward a tool call to the browser Worker and await the result.

        Sends ``{type: "browser_mcp_call", ...}`` over the WebSocket and
        creates an asyncio Future that the WS response handler resolves.
        """
        controller = self._ctx.get("controller")
        if not controller:
            raise RuntimeError("No controller in browser MCP session context")

        request_id = str(uuid4())
        loop = self._ctx.get("loop") or asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._ctx["pending_requests"][request_id] = future

        # Send tool call request to browser
        await controller._send({
            "type": "browser_mcp_call",
            "request_id": request_id,
            "nudge_uid": self.nudge_uid,
            "tool_name": name,
            "arguments": arguments,
        })

        # Await result with timeout
        try:
            result_msg = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._ctx["pending_requests"].pop(request_id, None)
            raise RuntimeError(f"Browser MCP tool call '{name}' timed out (30s)")

        if "error" in result_msg:
            raise RuntimeError(f"Browser MCP tool error: {result_msg['error']}")
        return result_msg.get("result", "")

    async def close(self):
        """Send stop message to browser to terminate the Worker."""
        controller = self._ctx.get("controller")
        if controller:
            await controller._send({
                "type": "stop_browser_mcp",
                "nudge_uid": self.nudge_uid,
            })
