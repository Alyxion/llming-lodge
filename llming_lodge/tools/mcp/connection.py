"""MCP (Model Context Protocol) connections for stdio and HTTP transports.

This module provides connection classes for communicating with MCP servers
over stdio (local processes) or HTTP (remote servers).
"""
import asyncio
import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llming_lodge.tools.mcp.config import MCPServerConfig
from llming_lodge.tools.tool_definition import ToolDefinition, ToolSource

logger = logging.getLogger(__name__)


class MCPConnection(ABC):
    """Abstract base class for MCP connections."""

    @abstractmethod
    async def start(self) -> None:
        """Start the connection."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the MCP server."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with the given arguments."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass


class MCPStdioConnection(MCPConnection):
    """MCP connection over stdio (JSON-RPC to local process).

    Spawns a child process and communicates via stdin/stdout using
    JSON-RPC 2.0 protocol as defined by MCP.
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize stdio connection.

        Args:
            config: MCP server configuration with command, args, env
        """
        if not config.command:
            raise ValueError("MCPServerConfig must have 'command' for stdio transport")

        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self) -> None:
        """Start the MCP server process."""
        if self._started:
            return

        # Build command
        cmd = [self.config.command] + (self.config.args or [])

        # Build environment
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        # Start process
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self.config.cwd,
            text=True,
            bufsize=1,
        )

        # Start reader task
        self._reader_task = asyncio.create_task(self._read_responses())
        self._started = True

        # Send initialize request
        await self._initialize()

        logger.info(f"MCP stdio connection started: {self.config.command}")

    async def _initialize(self) -> Dict[str, Any]:
        """Send MCP initialize request and initialized notification."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "llming-lodge",
                "version": "1.0.0"
            }
        })
        # Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized", {})
        return result

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or not self.process.stdin:
            raise MCPError("Connection not started")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str)
        self.process.stdin.flush()

    async def _read_responses(self) -> None:
        """Read JSON-RPC responses from stdout."""
        try:
            while self.process and self.process.poll() is None:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                if not line:
                    break

                try:
                    response = json.loads(line)
                    if "id" in response:
                        request_id = response["id"]
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            if "error" in response:
                                future.set_exception(
                                    MCPError(response["error"].get("message", "Unknown error"))
                                )
                            else:
                                future.set_result(response.get("result"))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse MCP response: {e}")
        except Exception as e:
            logger.error(f"MCP reader error: {e}")

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or not self.process.stdin:
            raise MCPError("Connection not started")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()

        # Wait for response (with timeout)
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Request {method} timed out")

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        tools = []

        for tool_data in result.get("tools", []):
            tool = ToolDefinition(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                source=ToolSource.MCP_STDIO,
                mcp_server=self.config,
            )
            tools.append(tool)

        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with the given arguments."""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        # MCP returns content as list of content blocks
        content = result.get("content", [])
        if len(content) == 1:
            return content[0].get("text", content[0])
        return content

    async def close(self) -> None:
        """Close the connection and terminate the process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self._started = False
        logger.info(f"MCP stdio connection closed: {self.config.command}")


class MCPHTTPConnection(MCPConnection):
    """MCP connection over HTTP/SSE (remote server).

    Connects to a remote MCP server using HTTP for requests and
    optionally SSE for streaming responses.
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize HTTP connection.

        Args:
            config: MCP server configuration with url, api_key, headers
        """
        if not config.url:
            raise ValueError("MCPServerConfig must have 'url' for HTTP transport")

        self.config = config
        self._session = None
        self._started = False

    async def start(self) -> None:
        """Start the HTTP connection."""
        if self._started:
            return

        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers=self._build_headers()
            )
            self._started = True
            logger.info(f"MCP HTTP connection started: {self.config.url}")
        except ImportError:
            raise MCPError("aiohttp is required for MCP HTTP connections")

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for requests."""
        headers = {"Content-Type": "application/json"}

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if self.config.headers:
            headers.update(self.config.headers)

        return headers

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the MCP server."""
        if not self._session:
            raise MCPError("Connection not started")

        async with self._session.get(f"{self.config.url}/tools/list") as resp:
            if resp.status != 200:
                raise MCPError(f"Failed to list tools: {resp.status}")

            result = await resp.json()
            tools = []

            for tool_data in result.get("tools", []):
                tool = ToolDefinition(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                    source=ToolSource.MCP_HTTP,
                    mcp_server=self.config,
                )
                tools.append(tool)

            return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with the given arguments."""
        if not self._session:
            raise MCPError("Connection not started")

        payload = {
            "name": name,
            "arguments": arguments
        }

        async with self._session.post(f"{self.config.url}/tools/call", json=payload) as resp:
            if resp.status != 200:
                raise MCPError(f"Tool call failed: {resp.status}")

            result = await resp.json()

            # MCP returns content as list of content blocks
            content = result.get("content", [])
            if len(content) == 1:
                return content[0].get("text", content[0])
            return content

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

        self._started = False
        logger.info(f"MCP HTTP connection closed: {self.config.url}")


class InProcessMCPServer(ABC):
    """Abstract base class for in-process MCP servers.

    Subclass this to create MCP-compatible tool servers that run in the same
    process as the host application, with direct access to runtime state
    (e.g. authenticated user sessions, database connections).
    """

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools.

        Returns:
            List of tool descriptors, each with keys: name, description, inputSchema
        """
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as a string
        """
        pass


class MCPInProcessConnection(MCPConnection):
    """MCP connection adapter wrapping an InProcessMCPServer.

    Provides the MCPConnection interface for in-process servers,
    enabling them to integrate with the existing MCP discovery
    and execution pipeline without subprocess or network overhead.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.server: InProcessMCPServer = config.server_instance

    async def start(self) -> None:
        """No-op — in-process server is already running."""
        logger.info(f"MCP in-process connection ready: {self.config.label or 'unnamed'}")

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the in-process server."""
        from llming_lodge.tools.tool_definition import ToolUIMetadata

        raw_tools = await self.server.list_tools()
        tools = []
        for tool_data in raw_tools:
            # Build ToolUIMetadata from optional displayName / icon / displayDescription
            ui_meta = None
            display_name = tool_data.get("displayName")
            icon = tool_data.get("icon")
            ui_description = tool_data.get("displayDescription")
            if display_name or icon or ui_description:
                ui_meta = ToolUIMetadata(display_name=display_name, icon=icon, description=ui_description)

            tool = ToolDefinition(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                source=ToolSource.MCP_INPROCESS,
                mcp_server=self.config,
                ui=ui_meta,
            )
            tools.append(tool)
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the in-process server."""
        return await self.server.call_tool(name, arguments)

    async def close(self) -> None:
        """No-op — in-process server lifecycle is managed externally."""
        logger.info(f"MCP in-process connection closed: {self.config.label or 'unnamed'}")


class MCPError(Exception):
    """Exception raised for MCP communication errors."""
    pass


def create_connection(config: MCPServerConfig) -> MCPConnection:
    """Create the appropriate MCP connection based on configuration.

    Args:
        config: MCP server configuration (MCPServerConfig or dict)

    Returns:
        MCPConnection instance (stdio, HTTP, or in-process)

    Raises:
        ValueError: If configuration is invalid
    """
    # Handle dict configs (from model_dump serialization)
    if isinstance(config, dict):
        config = MCPServerConfig(**config)

    if config.is_inprocess():
        return MCPInProcessConnection(config)
    elif config.is_stdio():
        return MCPStdioConnection(config)
    elif config.is_http():
        return MCPHTTPConnection(config)
    else:
        raise ValueError("MCPServerConfig must have either 'command' (stdio), 'url' (HTTP), or 'server_instance' (in-process)")
