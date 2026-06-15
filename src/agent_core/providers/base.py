from __future__ import annotations

from enum import StrEnum
from typing import Any, AsyncIterator, Protocol

from pydantic import Field

from agent_core.types.common import ErrorPayload, StrictModel
from agent_core.types.content import ContentBlock
from agent_core.types.tools import ToolCall, ToolDefinition


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class SystemBlock(StrictModel):
    block_id: str
    source: str
    content: str
    priority: int = 100
    dynamic: bool = False
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelMessage(StrictModel):
    role: ModelRole
    content: list[ContentBlock] = Field(default_factory=list)
    name: str | None = None
    node_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw_usage: dict[str, Any] | None = None


class ModelRequest(StrictModel):
    model: str
    system: list[SystemBlock] = Field(default_factory=list)
    messages: list[ModelMessage] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelEvent(StrictModel):
    event_type: str
    text_delta: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    content: list[ContentBlock] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: Usage | None = None
    error: ErrorPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderAdapter(Protocol):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        ...

    async def close(self) -> None:
        ...


ProviderFactory = Any

