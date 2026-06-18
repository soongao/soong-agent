from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agent_core.types.common import StrictModel


class AgentDefinition(StrictModel):
    agent_definition_id: str
    name: str
    description: str
    body: str = ""
    model_profile: str | dict[str, Any] | None = None
    suggested_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    overrides: str | None = None
    source: Literal["builtin", "user", "code", "json", "dynamic", "config"]
    metadata: dict[str, Any] = Field(default_factory=dict)
