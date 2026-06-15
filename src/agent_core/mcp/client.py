from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode

@dataclass
class McpClient:
    server_id: str
    config: dict
    connected: bool = False
    tools: list[dict] = field(default_factory=list)
    _process: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)
    _request_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def connect(self) -> None:
        if self.connected:
            return
        if "tools" in self.config and "command" not in self.config:
            self.tools = list(self.config.get("tools") or [])
            self.connected = True
            return
        command = self.config.get("command")
        if not command:
            raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"MCP server {self.server_id} missing command")
        argv = [str(command), *[str(item) for item in self.config.get("args", [])]]
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (self.config.get("env") or {}).items()})
        try:
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"failed to start MCP server {self.server_id}: {exc}") from exc
        self.connected = True
        await self._request(
            "initialize",
            {
                "protocolVersion": self.config.get("protocolVersion", "2024-11-05"),
                "capabilities": {},
                "clientInfo": {"name": "soong-agent", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})

    async def close(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self.connected = False

    async def list_tools(self) -> list[dict]:
        if not self.connected:
            await self.connect()
        if self.tools:
            return self.tools
        result = await self._request("tools/list", {})
        self.tools = list((result or {}).get("tools") or [])
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.connected:
            await self.connect()
        if "command" not in self.config:
            raise AgentCoreError(ErrorCode.UNSUPPORTED_CAPABILITY, f"MCP server {self.server_id} is discovery-only")
        result = await self._request("tools/call", {"name": tool_name, "arguments": arguments})
        return dict(result or {})

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            while True:
                message = await self._read()
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    raise AgentCoreError(ErrorCode.INTERNAL_ERROR, str(error.get("message") or error))
                return dict(message.get("result") or {})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"MCP server {self.server_id} is not connected")
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        self._process.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload)
        await self._process.stdin.drain()

    async def _read(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"MCP server {self.server_id} is not connected")
        content_length: int | None = None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"MCP server {self.server_id} closed stdout")
            stripped = line.strip()
            if not stripped:
                break
            key, _, value = stripped.decode("ascii", errors="replace").partition(":")
            if key.lower() == "content-length":
                content_length = int(value.strip())
        if content_length is None:
            raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"MCP server {self.server_id} sent message without Content-Length")
        payload = await self._process.stdout.readexactly(content_length)
        return dict(json.loads(payload.decode("utf-8")))
