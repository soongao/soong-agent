from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agent_core.types.common import ErrorPayload, StrictModel
from agent_core.types.content import ContentBlock, JsonBlock, TextBlock, ToolResultBlock


class ToolDefinition(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    permission: Literal["readonly", "write"]
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    tool_call_id: str
    tool_name: str
    content: list[ContentBlock] = Field(default_factory=list)
    is_error: bool = False
    error: ErrorPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(StrictModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_tool_result(value: Any, *, tool_call_id: str, tool_name: str) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, str):
        content: list[ContentBlock] = [TextBlock(text=value)]
    elif value is None:
        content = [JsonBlock(data=None)]
    elif isinstance(value, (dict, list, int, float, bool)):
        content = [JsonBlock(data=value)]
    else:
        content = [TextBlock(text=str(value))]
    return ToolResult(tool_call_id=tool_call_id, tool_name=tool_name, content=content)


def error_tool_result(
    *,
    tool_call_id: str,
    tool_name: str,
    error: ErrorPayload,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    block = ToolResultBlock(tool_call_id=tool_call_id, is_error=True, content=[], error=error)
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[block],
        is_error=True,
        error=error,
        metadata=metadata or {},
    )

