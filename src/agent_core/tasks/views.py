from __future__ import annotations

from typing import Any

from agent_core.tasks.access import step_visible_to_worker
from agent_core.tasks.helpers import task_updated_sort_key, wal_path_sort_key
from agent_core.tasks.models import Task, TaskStep
from agent_core.tasks.records import TaskRecord, UnavailableTaskRecord
from agent_core.tasks.state import TERMINAL_STEP_STATUSES, TERMINAL_TASK_STATUSES
from agent_core.tools.execution import ToolExecutionContext


def task_detail(record: TaskRecord, *, include_terminal_steps: bool = False) -> dict[str, Any]:
    data = record.task.model_dump(mode="json")
    if not include_terminal_steps:
        data["steps"] = [step for step in data["steps"] if step["status"] not in TERMINAL_STEP_STATUSES]
    return {"task": data, "wal_path": str(record.wal_path)}


def list_tasks_view(
    records: dict[tuple[str, str], TaskRecord],
    terminal_records: dict[tuple[str, str], TaskRecord],
    unavailable_records: dict[tuple[str, str], UnavailableTaskRecord],
    context: ToolExecutionContext,
    args: dict[str, Any],
) -> dict[str, Any]:
    status = args.get("status")
    include_terminal = bool(args.get("include_terminal", False))
    limit = int(args.get("limit") or 50)
    offset = int(args.get("offset") or 0)
    selected_records: list[TaskRecord] = []
    record_items = list(records.items())
    if include_terminal:
        record_items.extend(terminal_records.items())
    for (session_id, _task_id), record in record_items:
        if session_id == context.session_id and (not status or record.task.status == status):
            selected_records.append(record)
    selected_records.sort(key=lambda record: task_updated_sort_key(record.task, record.wal_path), reverse=True)
    sliced = selected_records[offset : offset + limit]
    task_summaries: list[dict[str, Any]] = [record.task.model_dump(mode="json") for record in sliced]
    total = len(selected_records)
    if include_terminal and (not status or status == "unavailable"):
        unavailable = [
            record
            for (session_id, _task_id), record in unavailable_records.items()
            if session_id == context.session_id
        ]
        unavailable.sort(key=lambda record: wal_path_sort_key(record.wal_path), reverse=True)
        available_remaining = max(0, limit - len(task_summaries))
        unavailable_offset = max(0, offset - total)
        if available_remaining > 0:
            task_summaries.extend(
                record.summary()
                for record in unavailable[unavailable_offset : unavailable_offset + available_remaining]
            )
        total += len(unavailable)
    return {"tasks": task_summaries, "truncated": offset + limit < total}


def active_task_summaries(records: dict[tuple[str, str], TaskRecord], session_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (record_session_id, _task_id), record in records.items():
        if record_session_id != session_id:
            continue
        task = record.task
        if task.status in TERMINAL_TASK_STATUSES:
            continue
        steps = [
            step
            for step in task.steps
            if step.status in {"ready", "claimed", "running", "blocked", "failed"} or step.required
        ][:20]
        summaries.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "summary": task.summary,
                "status": task.status,
                "wal_path": str(record.wal_path),
                "steps": [_step_summary(step) for step in steps],
            }
        )
    return summaries[:limit]


def unavailable_task_summaries(
    unavailable_records: dict[tuple[str, str], UnavailableTaskRecord],
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    records = [
        record
        for (record_session_id, _task_id), record in unavailable_records.items()
        if session_id is None or record_session_id == session_id
    ]
    records.sort(key=lambda record: wal_path_sort_key(record.wal_path), reverse=True)
    return [record.summary() for record in records]


def query_steps_view(record: TaskRecord, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    statuses = set(args.get("statuses") or [])
    include_terminal = bool(args.get("include_terminal_steps", False))
    worker_pool_id = args.get("worker_pool_id")
    claimed_by_agent_id = args.get("claimed_by_agent_id")
    default_limit = 5 if context.agent_role == "worker" and statuses == {"ready"} else 50
    limit = int(args["limit"]) if args.get("limit") is not None else default_limit
    offset = int(args.get("offset") or 0)
    steps = []
    for step in record.task.steps:
        if not step_visible_to_worker(context, step):
            continue
        if statuses and step.status not in statuses:
            continue
        if not include_terminal and step.status in TERMINAL_STEP_STATUSES:
            continue
        if worker_pool_id and step.worker_pool_id != worker_pool_id:
            continue
        if claimed_by_agent_id and step.claimed_by_agent_id != claimed_by_agent_id:
            continue
        steps.append(step)
    sliced = steps[offset : offset + limit]
    return {"steps": [step.model_dump(mode="json") for step in sliced], "truncated": offset + limit < len(steps)}


def dispatchable_steps_view(task: Task, *, worker_pool_id: str, allowed_step_ids: list[str] | None = None) -> list[TaskStep]:
    allowed = set(allowed_step_ids) if allowed_step_ids is not None else None
    steps: list[TaskStep] = []
    for step in task.steps:
        if step.status != "ready":
            continue
        if allowed is not None and step.step_id not in allowed:
            continue
        if step.worker_pool_id is not None and step.worker_pool_id != worker_pool_id:
            continue
        steps.append(step)
    return steps


def _step_summary(step: TaskStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "summary": step.summary,
        "status": step.status,
        "worker_pool_id": step.worker_pool_id,
        "claimed_by_agent_id": step.claimed_by_agent_id,
        "claimed_by_run_id": step.claimed_by_run_id,
        "lease_expires_at": step.lease_expires_at,
        "result_summary": step.result_summary,
        "artifact_ids": step.artifact_ids,
    }
