from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from agent_core.types.common import ErrorPayload, StrictModel
from agent_core.types.content import ContentBlock, TextBlock


class RunStatus(StrEnum):
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"


class RunMode(StrEnum):
    NORMAL = "normal"
    ORCHESTRATOR = "orchestrator"


class RuntimeEvent(StrictModel):
    event_id: str
    seq: int | None = None
    run_seq: int | None = None
    session_id: str
    agent_id: str | None = None
    run_id: str | None = None
    level: Literal["debug", "info", "warning", "error"] = "info"
    event_type: str
    node_id: str | None = None
    tool_call_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class UserMessage(StrictModel):
    content: str | list[ContentBlock]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str) -> "UserMessage":
        return cls(content=[TextBlock(text=text)])


class Node(StrictModel):
    node_id: str
    parent_id: str | None = None
    agent_id: str
    run_id: str | None = None
    role: str
    node_type: str
    content: list[ContentBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None
    created_at: datetime


class ReplayResult(StrictModel):
    session_id: str
    run_id: str | None = None
    nodes: list[Node] = Field(default_factory=list)
    events: list[RuntimeEvent] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    model_requests: list[dict[str, Any]] = Field(default_factory=list)
    task_wal_errors: list[dict[str, Any]] = Field(default_factory=list)


class SessionInfo(StrictModel):
    session_id: str
    cwd: str
    root_agent_id: str
    active_node_id: str | None = None
    parent_session_id: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionNodeInfo(StrictModel):
    node_id: str
    parent_id: str | None = None
    role: str
    node_type: str
    content_preview: str = ""
    created_at: datetime
    active: bool = False


class ForkSessionResult(StrictModel):
    source_session_id: str
    session_id: str | None = None
    source_node_id: str | None = None
    active_node_id: str | None = None
    forked: bool
    copied_nodes: int = 0
    error: ErrorPayload | None = None


class CancelResult(StrictModel):
    run_id: str
    status: RunStatus
    cancelled: bool
    reason: str | None = None


class InspectResult(StrictModel):
    run_id: str
    nodes: list[Node] = Field(default_factory=list)
    events: list[RuntimeEvent] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    model_requests: list[dict[str, Any]] = Field(default_factory=list)
    task_wal_errors: list[dict[str, Any]] = Field(default_factory=list)


class DeleteSessionResult(StrictModel):
    session_id: str
    deleted: bool
    error: ErrorPayload | None = None


class CleanupResult(StrictModel):
    dry_run: bool
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    errors: list[ErrorPayload] = Field(default_factory=list)


class SwitchNodeResult(StrictModel):
    session_id: str
    node_id: str
    switched: bool
    error: ErrorPayload | None = None


class SkillInfo(StrictModel):
    name: str
    description: str = ""
    path: str


class LoadSkillResult(StrictModel):
    session_id: str
    name: str
    path: str | None = None
    hash: str | None = None
    node_id: str | None = None
    loaded: bool
    already_loaded: bool = False
    error: ErrorPayload | None = None
