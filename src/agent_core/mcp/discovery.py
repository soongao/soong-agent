from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent_core.config.models import ToolsConfig
from agent_core.errors import AgentCoreError
from agent_core.mcp.client import McpClient
from agent_core.types.tools import ToolDefinition


class McpDiscovery:
    def __init__(self, config: dict, tools_config: ToolsConfig) -> None:
        self.config = config
        self.tools_config = tools_config

    def available_servers(self) -> list[str]:
        servers = self.config.get("servers") or {}
        disabled = set(self.tools_config.mcp.disabled_servers)
        return [server_id for server_id in sorted(servers) if server_id not in disabled]

    def tool_enabled(self, canonical_name: str) -> bool:
        return canonical_name not in set(self.tools_config.mcp.disabled_tools)


@dataclass
class McpToolBinding:
    definition: ToolDefinition
    client: McpClient
    mcp_tool_name: str


@dataclass
class McpDiscoveryResult:
    tools: dict[str, McpToolBinding] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


class McpToolManager:
    def __init__(self, config: dict[str, Any], tools_config: ToolsConfig) -> None:
        self.config = config
        self.tools_config = tools_config
        self.discovery = McpDiscovery(config, tools_config)
        self._cache_key: str | None = None
        self._cache_expires_at = 0.0
        self._result = McpDiscoveryResult()

    async def discover(self) -> McpDiscoveryResult:
        key = self._config_hash()
        now = time.monotonic()
        if key == self._cache_key and now < self._cache_expires_at:
            return self._result
        result = McpDiscoveryResult()
        servers = self.config.get("servers") or {}
        for server_id in self.discovery.available_servers():
            server_config = servers.get(server_id) or {}
            client = McpClient(server_id=server_id, config=server_config)
            try:
                tools = await client.list_tools()
            except Exception as exc:
                await client.close()
                result.failures.append({"server_id": server_id, "message": str(exc)})
                continue
            for raw_tool in tools:
                try:
                    binding = self._definition_from_raw_tool(server_id, client, raw_tool)
                except AgentCoreError as exc:
                    result.failures.append({"server_id": server_id, "tool_name": raw_tool.get("name"), "message": exc.message})
                    continue
                if self.discovery.tool_enabled(binding.definition.name):
                    result.tools[binding.definition.name] = binding
        self._cache_key = key
        self._cache_expires_at = now + (self.tools_config.mcp.discovery_cache_ttl_ms / 1000)
        self._result = result
        return result

    async def close(self) -> None:
        seen: set[int] = set()
        for binding in self._result.tools.values():
            if id(binding.client) in seen:
                continue
            seen.add(id(binding.client))
            await binding.client.close()

    def _definition_from_raw_tool(self, server_id: str, client: McpClient, raw_tool: dict[str, Any]) -> McpToolBinding:
        tool_name = str(raw_tool.get("name") or "")
        if not tool_name:
            raise AgentCoreError("validation_error", f"MCP server {server_id} returned a tool without name")
        canonical = f"mcp.{server_id}.{tool_name}"
        override = dict(self.tools_config.mcp.tool_overrides.get(canonical) or {})
        permission = override.get("permission") or _infer_permission(raw_tool)
        tags = set(raw_tool.get("tags") or [])
        tags.add("mcp")
        if permission == "readonly":
            tags.add("readonly")
        else:
            tags.add("write")
        tags.update(override.get("tags") or [])
        definition = ToolDefinition(
            name=canonical,
            description=str(raw_tool.get("description") or override.get("description") or f"MCP tool {tool_name}"),
            input_schema=dict(raw_tool.get("inputSchema") or raw_tool.get("input_schema") or {"type": "object", "properties": {}}),
            permission=permission,
            tags=tags,
            metadata={"mcp_server_id": server_id, "mcp_tool_name": tool_name, "raw_tool": raw_tool},
        )
        return McpToolBinding(definition=definition, client=client, mcp_tool_name=tool_name)

    def _config_hash(self) -> str:
        payload = {
            "mcp": self.config,
            "disabled_servers": self.tools_config.mcp.disabled_servers,
            "disabled_tools": self.tools_config.mcp.disabled_tools,
            "tool_overrides": self.tools_config.mcp.tool_overrides,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _infer_permission(raw_tool: dict[str, Any]) -> str:
    metadata = raw_tool.get("annotations") or raw_tool.get("metadata") or {}
    if raw_tool.get("permission") == "readonly" or metadata.get("permission") == "readonly" or metadata.get("readOnlyHint") is True:
        return "readonly"
    return "write"
