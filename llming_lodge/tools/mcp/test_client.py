#!/usr/bin/env python3
"""
MCP Test Client for testing MCP servers via subprocess.

Based on the pattern from nice-vibes MCP implementation.
Communicates with MCP servers using JSON-RPC over stdio.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional


class MCPTestClient:
    """Test client for MCP servers via subprocess."""

    def __init__(self, server_module: str = "llming_lodge.tools.mcp.sample_server"):
        """Initialize the test client.

        Args:
            server_module: Python module path to the MCP server
        """
        self.server_module = server_module
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id = 0
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the MCP server subprocess."""
        python = sys.executable

        self.process = await asyncio.create_subprocess_exec(
            python, "-m", self.server_module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=10 * 1024 * 1024,  # 10MB buffer
        )

        # Start reading stderr in background (for server logs)
        self._stderr_task = asyncio.create_task(self._read_stderr())

        # Initialize the connection
        await self._initialize()

    async def _read_stderr(self) -> None:
        """Read and print stderr from server."""
        while self.process and self.process.stderr:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break
                # Uncomment for debugging:
                # print(f"[SERVER] {line.decode().strip()}", file=sys.stderr)
            except Exception:
                break

    async def _send_request(self, method: str, params: Optional[dict] = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Server not started")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Send request
        request_json = json.dumps(request)
        self.process.stdin.write(request_json.encode() + b"\n")
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("Server closed connection")

        response = json.loads(response_line.decode())
        return response

    async def _initialize(self) -> dict:
        """Initialize the MCP connection."""
        response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "llming-lodge-test-client",
                "version": "1.0.0"
            }
        })

        if "error" in response:
            raise RuntimeError(f"Initialize error: {response['error']}")

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        self.process.stdin.write(json.dumps(notification).encode() + b"\n")
        await self.process.stdin.drain()

        return response.get("result", {})

    async def list_tools(self) -> list[dict]:
        """List available tools."""
        response = await self._send_request("tools/list")
        if "error" in response:
            raise RuntimeError(f"Error listing tools: {response['error']}")
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Call a tool and return the result."""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        response = await self._send_request("tools/call", params)
        if "error" in response:
            return {"error": response["error"]}
        return response.get("result", {})

    async def call_tool_text(self, name: str, arguments: Optional[dict] = None) -> str:
        """Call a tool and return text content only."""
        result = await self.call_tool(name, arguments)
        if "error" in result:
            raise RuntimeError(f"Tool error: {result['error']}")

        content = result.get("content", [])
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)

    async def stop(self) -> None:
        """Stop the server."""
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()


async def interactive_session():
    """Run an interactive test session."""
    client = MCPTestClient()

    try:
        await client.start()

        print("\n" + "=" * 60)
        print("MCP Server Test Client")
        print("=" * 60)
        print("\nCommands:")
        print("  tools          - List available tools")
        print("  call <tool>    - Call a tool (will prompt for args)")
        print("  quit           - Exit")
        print()

        while True:
            try:
                cmd = input("\n> ").strip()
            except EOFError:
                break

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if command in ("quit", "exit"):
                break

            elif command == "tools":
                tools = await client.list_tools()
                print(f"\nAvailable tools ({len(tools)}):")
                for tool in tools:
                    name = tool.get("name", "")
                    desc = (tool.get("description", "") or "").split("\n")[0]
                    print(f"  - {name}: {desc}")

            elif command == "call":
                if not arg:
                    print("Usage: call <tool_name>")
                    continue

                tool_name = arg

                # Get tool info
                tools = await client.list_tools()
                tool = next((t for t in tools if t["name"] == tool_name), None)

                if not tool:
                    print(f"Unknown tool: {tool_name}")
                    continue

                # Build arguments
                schema = tool.get("inputSchema", {})
                properties = schema.get("properties", {})
                required = schema.get("required", [])

                arguments = {}
                for prop_name, prop_info in properties.items():
                    is_required = prop_name in required
                    prompt = f"  {prop_name}"
                    if prop_info.get("description"):
                        prompt += f" ({prop_info['description'][:50]})"
                    if not is_required:
                        prompt += " [optional]"
                    prompt += ": "

                    value = input(prompt).strip()
                    if value:
                        # Try to parse as JSON for complex types
                        try:
                            arguments[prop_name] = json.loads(value)
                        except json.JSONDecodeError:
                            arguments[prop_name] = value
                    elif is_required:
                        print(f"  {prop_name} is required!")
                        break
                else:
                    # All args collected, call the tool
                    print(f"\nCalling {tool_name}...")
                    try:
                        text = await client.call_tool_text(tool_name, arguments)
                        print(text)
                    except RuntimeError as e:
                        print(f"Error: {e}")

            else:
                print(f"Unknown command: {command}")

    finally:
        await client.stop()
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(interactive_session())
