from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from agent_core.types.common import StrictModel


class PermissionDecisionKind(StrEnum):
    ALLOW_ONCE = "allow_once"
    ALLOW_FOR_SESSION = "allow_for_session"
    DENY = "deny"


class PermissionRequest(StrictModel):
    request_id: str
    session_id: str
    agent_id: str
    run_id: str
    parent_agent_id: str | None = None
    parent_run_id: str | None = None
    agent_role: str
    tool_name: str
    permission: Literal["readonly", "write"]
    tags: list[str] = Field(default_factory=list)
    args_summary: str
    target_scope: str | None = None
    cwd: str
    env_summary: dict[str, Any] = Field(default_factory=dict)
    network_host: str | None = None
    dangerous: bool = False
    hook_summary: dict[str, Any] | None = None
    suggested_decision: PermissionDecisionKind = PermissionDecisionKind.DENY
    metadata: dict[str, Any] = Field(default_factory=dict)


class PermissionDecision(StrictModel):
    decision: PermissionDecisionKind
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

