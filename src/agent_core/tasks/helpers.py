from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from agent_core.config.paths import expand_config_path
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tasks.models import Task, TaskStep
from agent_core.tasks.operations import validate_no_dependency_cycle
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.common import utc_iso, utc_now, validate_safe_id


def step_from_input(raw: dict[str, Any]) -> TaskStep:
    return TaskStep(
        step_id=validate_safe_id(str(raw["step_id"]), field_name="step_id"),
        title=str(raw["title"]),
        summary=str(raw.get("summary", "")),
        status=str(raw.get("status", "pending")),
        depends_on_step_ids=[validate_safe_id(str(dep), field_name="depends_on_step_id") for dep in raw.get("depends_on_step_ids", [])],
        required=bool(raw.get("required", True)),
        worker_pool_id=raw.get("worker_pool_id"),
    )


def step_wal_payload(event_type: str, step: TaskStep, *, previous_status: str) -> dict[str, Any]:
    if event_type == "task_step_started":
        return {
            "previous_status": previous_status,
            "status": "running",
            "lease_expires_at": step.lease_expires_at,
        }
    if event_type == "task_step_updated":
        return {
            "previous_status": previous_status,
            "status": step.status,
            "result_summary": step.result_summary,
            "artifact_ids": step.artifact_ids,
            "lease_expires_at": step.lease_expires_at,
        }
    if event_type == "task_step_blocked":
        return {
            "previous_status": previous_status,
            "reason": step.reason,
            "result_summary": step.result_summary,
            "artifact_ids": step.artifact_ids,
        }
    if event_type == "task_step_completed":
        return {
            "previous_status": previous_status,
            "result_summary": step.result_summary,
            "artifact_ids": step.artifact_ids,
        }
    if event_type == "task_step_failed":
        return {
            "previous_status": previous_status,
            "reason": step.reason,
            "result_summary": step.result_summary,
            "artifact_ids": step.artifact_ids,
        }
    if event_type == "task_step_cancelled":
        return {
            "previous_status": previous_status,
            "reason": step.reason,
            "result_summary": step.result_summary,
        }
    return {"previous_status": previous_status}


def stamp_entries(entries: list[dict[str, Any]], created_at: str) -> None:
    for entry in entries:
        entry["created_at"] = created_at


def touch_steps(task: Task, step_ids: Iterable[str], updated_at: str) -> None:
    wanted = {str(step_id) for step_id in step_ids}
    if not wanted:
        return
    for step in task.steps:
        if step.step_id in wanted:
            step.updated_at = updated_at


def sync_task_roots(task: Task) -> None:
    task.root_step_ids = [step.step_id for step in task.steps if not step.depends_on_step_ids]


def operation_step_ids(operations: list[dict[str, Any]]) -> list[str]:
    step_ids: list[str] = []
    for operation in operations:
        kind = operation.get("op")
        if kind == "add_step" and isinstance(operation.get("step"), dict):
            raw_step_id = operation["step"].get("step_id")
            if raw_step_id is not None:
                step_ids.append(str(raw_step_id))
        for field in ("step_id", "depends_on_step_id"):
            if operation.get(field) is not None:
                step_ids.append(str(operation[field]))
    return step_ids


def replace_task(target: Task, source: Task) -> None:
    for field_name in source.__class__.model_fields:
        setattr(target, field_name, getattr(source, field_name))


def task_updated_sort_key(task: Task, wal_path: Path) -> float:
    if task.updated_at:
        try:
            return parse_utc(task.updated_at).timestamp()
        except ValueError:
            pass
    return wal_path_sort_key(wal_path)


def wal_path_sort_key(wal_path: Path) -> float:
    try:
        return wal_path.stat().st_mtime
    except OSError:
        return 0.0


def task_id_from_unavailable_wal(wal_path: Path) -> str:
    name = wal_path.name
    suffix = ".wal.jsonl"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    try:
        return validate_safe_id(name, field_name="task_id")
    except ValueError:
        return f"unavailable_{abs(hash(str(wal_path))) & 0xFFFFFFFF:x}"


def validate_steps(steps: list[TaskStep]) -> None:
    ids = [step.step_id for step in steps]
    if len(ids) != len(set(ids)):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "duplicate step_id")
    id_set = set(ids)
    edges: list[tuple[str, str]] = []
    for step in steps:
        for dep in step.depends_on_step_ids:
            if dep not in id_set:
                raise AgentCoreError(ErrorCode.STEP_NOT_FOUND, f"dependency not found: {dep}")
            edges.append((step.step_id, dep))
    if not validate_no_dependency_cycle(edges):
        raise AgentCoreError(ErrorCode.DEPENDENCY_CYCLE, "task dependency cycle")


def find_step(task: Task, step_id: str) -> TaskStep:
    for step in task.steps:
        if step.step_id == step_id:
            return step
    raise AgentCoreError(ErrorCode.STEP_NOT_FOUND, f"step not found: {step_id}")


def task_wal_dir(context: ToolExecutionContext) -> Path:
    return expand_config_path(context.config.task.wal_dir, home_dir=context.home_dir, project_dir=context.project_dir).resolve()


def worker_scope(context: ToolExecutionContext) -> dict[str, Any]:
    if not context.services:
        return {}
    return dict(context.services.get("worker_scope") or {})


def ensure_orchestrator(context: ToolExecutionContext, action: str) -> None:
    ensure_role(context, {"orchestrator"}, action)


def ensure_task_reader(context: ToolExecutionContext, action: str) -> None:
    ensure_role(context, {"orchestrator", "worker"}, action)
    if context.agent_role == "worker" and not worker_scope(context).get("task_id"):
        raise AgentCoreError(ErrorCode.PERMISSION_DENIED, f"worker cannot {action} without dispatch scope")


def ensure_role(context: ToolExecutionContext, allowed_roles: set[str], action: str) -> None:
    if context.agent_role not in allowed_roles:
        roles = ", ".join(sorted(allowed_roles))
        raise AgentCoreError(ErrorCode.PERMISSION_DENIED, f"{context.agent_role} agent cannot {action}; allowed roles: {roles}")


def lease_expires_at(context: ToolExecutionContext) -> str:
    return utc_iso(utc_now() + timedelta(milliseconds=context.config.task.step_lease_timeout_ms))


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def updated_after_dispatch_step_ids(task: Task, operations: list[dict[str, Any]]) -> list[str]:
    dispatched = {step.step_id for step in task.steps if step.status in {"claimed", "running"}}
    if not dispatched:
        return []
    touched: set[str] = set()
    for operation in operations:
        kind = operation.get("op")
        if kind == "update_task":
            touched.update(dispatched)
            continue
        step_id = operation.get("step_id")
        if step_id is not None and str(step_id) in dispatched:
            touched.add(str(step_id))
        depends_on_step_id = operation.get("depends_on_step_id")
        if depends_on_step_id is not None and str(depends_on_step_id) in dispatched:
            touched.add(str(depends_on_step_id))
    return sorted(touched)


def mark_updated_after_dispatch(task: Task, step_ids: list[str]) -> None:
    if not step_ids:
        return
    targets = set(step_ids)
    for step in task.steps:
        if step.step_id in targets:
            step.metadata = {**step.metadata, "updated_after_dispatch": True}


def with_wal_event_ids(data: dict[str, Any], event_ids: list[str]) -> dict[str, Any]:
    if not event_ids:
        return data
    return {
        **data,
        "wal_event_id": event_ids[-1],
        "wal_event_ids": event_ids,
    }


def safe_wal_name(value: str) -> str:
    name = Path(value).name
    if name != value or not name.endswith(".wal.jsonl"):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "wal_name must be a local *.wal.jsonl filename")
    return name
