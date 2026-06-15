from __future__ import annotations

from typing import Any

from agent_core.mcp.discovery import McpToolBinding
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.content import ArtifactRefBlock, JsonBlock, TextBlock
from agent_core.types.tools import ToolResult
from agent_core.tools.output_limits import truncate_bytes


def register_mcp_tools(registry: ToolRegistry, bindings: dict[str, McpToolBinding]) -> None:
    for canonical_name, binding in bindings.items():
        if registry.get(canonical_name) is not None:
            continue
        registry.register_tool(binding.definition, _make_mcp_handler(binding))


def _make_mcp_handler(binding: McpToolBinding):
    async def handler(context: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
        raw_result = await binding.client.call_tool(binding.mcp_tool_name, args)
        return _normalize_mcp_result(context, binding.definition.name, raw_result)

    return handler


def _normalize_mcp_result(context: ToolExecutionContext, tool_name: str, raw_result: dict[str, Any]) -> ToolResult:
    blocks = []
    metadata: dict[str, Any] = {}
    is_error = bool(raw_result.get("isError") or raw_result.get("is_error"))
    for item in raw_result.get("content") or []:
        item_type = item.get("type")
        if item_type == "text":
            original_text = str(item.get("text") or "")
            text, truncated = truncate_bytes(original_text, context.config.tools.stdout_limit_bytes)
            if truncated:
                artifact = context.artifact_manager.write_text(
                    session_id=context.session_id,
                    text=original_text,
                    filename=f"{tool_name.replace('.', '_')}.txt",
                    summary="truncated MCP text output",
                )
                blocks.append(ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=artifact.summary, mime_type="text/plain"))
                metadata["stdout_artifact_id"] = artifact.artifact_id
            blocks.append(TextBlock(text=text))
        elif item_type == "json":
            blocks.append(JsonBlock(data=item.get("data")))
        else:
            blocks.append(JsonBlock(data=item))
    if not blocks:
        blocks.append(JsonBlock(data=raw_result.get("structuredContent") or raw_result.get("structured_content") or raw_result))
    return ToolResult(tool_call_id="", tool_name=tool_name, content=blocks, is_error=is_error, metadata=metadata)
