from __future__ import annotations

from typing import Any

from pydantic import Field

from agent_core.types.common import StrictModel


class TaskStep(StrictModel):
    step_id: str
    title: str
    summary: str = ""
    status: str = "pending"
    depends_on_step_ids: list[str] = Field(default_factory=list)
    required: bool = True
    worker_pool_id: str | None = None
    claimed_by_agent_id: str | None = None
    claimed_by_run_id: str | None = None
    lease_expires_at: str | None = None
    result_summary: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(StrictModel):
    task_id: str
    wal_name: str | None = None
    wal_path: str | None = None
    title: str
    summary: str = ""
    status: str = "pending"
    root_step_ids: list[str] = Field(default_factory=list)
    created_by_agent_id: str | None = None
    created_by_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    steps: list[TaskStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
