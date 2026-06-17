from __future__ import annotations

from typing import Any, Callable

from agent_core.tasks.helpers import find_step, replace_task
from agent_core.tasks.models import Task
from agent_core.tasks.state import apply_operation, promote_ready, promote_running


def replay_task_event(
    task: Task,
    event_type: str,
    step_id: str | None,
    payload: dict[str, Any],
    *,
    created_at: str | None = None,
    apply_operation_func: Callable[[Task, dict[str, Any]], None] = apply_operation,
) -> None:
    if event_type == "task_updated":
        if payload.get("task_summary"):
            replace_task(task, Task.model_validate(payload["task_summary"]))
        else:
            for op in payload.get("operations") or []:
                apply_operation_func(task, op)
            promote_ready(task)
            promote_running(task)
    elif event_type == "task_running":
        task.status = "running"
    elif event_type == "task_reopened":
        task.status = "pending"
        promote_ready(task)
        promote_running(task)
    elif event_type == "task_completed":
        task.status = "completed"
        for cancelled_step_id in payload.get("cancelled_optional_step_ids") or []:
            step = find_step(task, str(cancelled_step_id))
            step.status = "cancelled"
            step.reason = "task_completed"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
    elif event_type == "task_failed":
        task.status = "failed"
        for failed_step_id in payload.get("failed_step_ids") or []:
            step = find_step(task, str(failed_step_id))
            step.status = "failed"
            step.reason = "task_failed"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
    elif event_type == "task_cancelled":
        task.status = "cancelled"
        for cancelled_step_id in payload.get("cancelled_step_ids") or []:
            step = find_step(task, str(cancelled_step_id))
            step.status = "cancelled"
            step.reason = "task_cancelled"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
    elif event_type in {
        "task_step_claimed",
        "task_step_started",
        "task_step_updated",
        "task_step_ready",
        "task_step_blocked",
        "task_step_completed",
        "task_step_failed",
        "task_step_cancelled",
        "task_step_reopened",
        "task_step_lease_expired",
    } and step_id:
        step = find_step(task, step_id)
        if event_type == "task_step_claimed":
            step.status = "claimed"
            step.claimed_by_agent_id = payload.get("claimed_by_agent_id")
            step.claimed_by_run_id = payload.get("claimed_by_run_id")
            step.lease_expires_at = payload.get("lease_expires_at")
        elif event_type == "task_step_started":
            step.status = "running"
            if payload.get("lease_expires_at") is not None:
                step.lease_expires_at = payload.get("lease_expires_at")
        elif event_type == "task_step_blocked":
            step.status = "blocked"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
        elif event_type == "task_step_completed":
            step.status = "completed"
            step.lease_expires_at = None
        elif event_type == "task_step_failed":
            step.status = "failed"
            step.lease_expires_at = None
        elif event_type == "task_step_cancelled":
            step.status = "cancelled"
            step.lease_expires_at = None
        elif event_type == "task_step_ready":
            step.status = "ready"
        elif event_type == "task_step_reopened":
            step.status = "pending"
        elif event_type == "task_step_lease_expired":
            step.status = "pending"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
        if payload.get("result_summary") is not None:
            step.result_summary = payload.get("result_summary")
        if payload.get("artifact_ids") is not None:
            step.artifact_ids = list(payload.get("artifact_ids") or [])
        if payload.get("reason") is not None:
            step.reason = payload.get("reason")
        promote_ready(task)
        promote_running(task)
        step.updated_at = created_at or step.updated_at
    if created_at:
        task.updated_at = created_at
