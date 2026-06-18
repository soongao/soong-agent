from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from agent_core.types.common import StrictModel


class HubEvent(StrictModel):
    id: str
    type: str
    conversation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConversationView(StrictModel):
    conversation_id: str
    core_session_id: str
    title: str
    status: str
    active_core_node_id: str | None = None
    last_message_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | str
    updated_at: datetime | str


class ConversationListResponse(StrictModel):
    conversations: list[ConversationView]


class ConversationCreateRequest(StrictModel):
    title: str = "New conversation"


class MessageView(StrictModel):
    message_id: str
    conversation_id: str
    parent_message_id: str | None = None
    sender_type: Literal["user", "orchestrator", "worker", "system"] | str
    sender_id: str | None = None
    sender_name: str
    target_type: Literal["orchestrator", "worker", "none"] | str | None = None
    target_id: str | None = None
    original_text: str = ""
    display_text: str = ""
    status: str
    core_session_id: str | None = None
    core_run_id: str | None = None
    core_node_id: str | None = None
    child_run_id: str | None = None
    task_id: str | None = None
    worker_id: str | None = None
    queue_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | str
    updated_at: datetime | str


class MessageListResponse(StrictModel):
    messages: list[MessageView]


class MessageSendRequest(StrictModel):
    text: str


class MessageSendResponse(StrictModel):
    message_id: str
    conversation_id: str
    core_session_id: str
    core_run_id: str
    status: str


class SkillLoadRequest(StrictModel):
    name: str


class ConversationCancelRequest(StrictModel):
    core_run_id: str | None = None
    queue_id: str | None = None


class WorkerListResponse(StrictModel):
    workers: list[dict[str, Any]]


class BranchableNodeResponse(StrictModel):
    nodes: list[dict[str, Any]]


class BranchRequest(StrictModel):
    core_node_id: str


class ForkRequest(StrictModel):
    core_node_id: str
    title: str | None = None


class PermissionDecisionRequest(StrictModel):
    decision: Literal["allow_once", "allow_for_session", "deny"]
