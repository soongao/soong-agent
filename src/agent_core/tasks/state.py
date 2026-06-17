from __future__ import annotations

from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tasks.helpers import find_step, step_from_input
from agent_core.tasks.models import Task, TaskStep
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.common import validate_safe_id


TERMINAL_STEP_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}


def apply_operation(task: Task, op: dict[str, Any]) -> None:
    kind = op.get("op")
    if kind == "update_task":
        if "title" in op:
            task.title = str(op["title"])
        if "summary" in op:
            task.summary = str(op["summary"])
        if "status" in op:
            status = str(op["status"])
            if status == "blocked":
                task.status = "blocked"
            elif status == "pending" and task.status == "blocked":
                task.status = "pending"
            else:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported task status update: {status}")
    elif kind == "add_step":
        step = step_from_input(op["step"])
        if any(existing.step_id == step.step_id for existing in task.steps):
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"duplicate step_id: {step.step_id}")
        task.steps.append(step)
    elif kind == "update_step":
        step = find_step(task, str(op["step_id"]))
        for field in ("title", "summary", "worker_pool_id", "required"):
            if field in op:
                setattr(step, field, op[field])
        if "depends_on_step_ids" in op:
            step.depends_on_step_ids = [validate_safe_id(str(dep), field_name="depends_on_step_id") for dep in op.get("depends_on_step_ids") or []]
    elif kind == "delete_step":
        step_id = str(op["step_id"])
        step = find_step(task, step_id)
        if step.status not in {"pending", "ready", "cancelled"}:
            raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, f"cannot delete {step.status} step")
        if any(step_id in step.depends_on_step_ids for step in task.steps):
            raise AgentCoreError(ErrorCode.STEP_HAS_DEPENDENTS, "step has dependents")
        task.steps = [step for step in task.steps if step.step_id != step_id]
    elif kind == "add_dependency":
        step = find_step(task, str(op["step_id"]))
        dep = str(op["depends_on_step_id"])
        find_step(task, dep)
        if dep not in step.depends_on_step_ids:
            step.depends_on_step_ids.append(dep)
    elif kind == "remove_dependency":
        step = find_step(task, str(op["step_id"]))
        dep = str(op["depends_on_step_id"])
        step.depends_on_step_ids = [item for item in step.depends_on_step_ids if item != dep]
    elif kind == "cancel_step":
        step = find_step(task, str(op["step_id"]))
        if step.status in {"claimed", "running"}:
            raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "cannot directly cancel claimed/running step")
        step.status = "cancelled"
        step.reason = op.get("reason")
    elif kind == "reopen_step":
        step = find_step(task, str(op["step_id"]))
        if step.status == "cancelled":
            raise AgentCoreError(ErrorCode.TASK_TERMINAL, "cancelled step cannot reopen")
        if step.status in {"blocked", "failed"}:
            step.status = "pending"
            step.reason = op.get("reason")
    else:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported task_update op: {kind}")


def promote_ready(task: Task) -> list[tuple[TaskStep, str]]:
    completed = {step.step_id for step in task.steps if step.status == "completed"}
    transitions: list[tuple[TaskStep, str]] = []
    for step in task.steps:
        if step.status in {"pending", "ready"}:
            previous = step.status
            step.status = "ready" if all(dep in completed for dep in step.depends_on_step_ids) else "pending"
            if previous != step.status:
                transitions.append((step, previous))
    return transitions


def promote_running(task: Task) -> str | None:
    if task.status == "pending" and any(step.status in {"ready", "claimed", "running"} for step in task.steps):
        previous = task.status
        task.status = "running"
        return previous
    return None


def ensure_not_terminal(task: Task) -> None:
    if task.status in TERMINAL_TASK_STATUSES:
        raise AgentCoreError(ErrorCode.TASK_TERMINAL, f"task is terminal: {task.status}")


def ensure_dispatchable(task: Task) -> None:
    ensure_not_terminal(task)
    if task.status == "blocked":
        raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "task is blocked")


def ready_event_entries(
    context: ToolExecutionContext,
    transitions: list[tuple[TaskStep, str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for step, previous_status in transitions:
        if step.status != "ready":
            continue
        entries.append(
            {
                "event_type": "task_step_ready",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "step_id": step.step_id,
                "payload": {
                    "previous_status": previous_status,
                    "status": "ready",
                    "depends_on_step_ids": list(step.depends_on_step_ids),
                },
            }
        )
    return entries
