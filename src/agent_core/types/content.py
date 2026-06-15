from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field

from agent_core.types.common import ErrorPayload, StrictModel


class TextBlock(StrictModel):
    type: Literal["text"] = "text"
    text: str


class JsonBlock(StrictModel):
    type: Literal["json"] = "json"
    data: Any | None = None
    summary: str | None = None
    artifact_id: str | None = None


class ToolCallBlock(StrictModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRefBlock(StrictModel):
    type: Literal["artifact_ref"] = "artifact_ref"
    artifact_id: str
    summary: str | None = None
    mime_type: str | None = None


class ToolResultBlock(StrictModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    is_error: bool = False
    content: list["ContentBlock"] = Field(default_factory=list)
    error: ErrorPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Annotated[
    Union[TextBlock, JsonBlock, ToolCallBlock, ToolResultBlock, ArtifactRefBlock],
    Field(discriminator="type"),
]


def text_block(text: str) -> TextBlock:
    return TextBlock(text=text)


def json_block(data: Any, *, summary: str | None = None, artifact_id: str | None = None) -> JsonBlock:
    return JsonBlock(data=data, summary=summary, artifact_id=artifact_id)

