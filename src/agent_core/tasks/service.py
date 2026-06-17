from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tasks.access import ensure_worker_task_scope, step_visible_to_worker
from agent_core.tasks.helpers import (
    ensure_orchestrator,
    ensure_role,
    ensure_task_reader,
    find_step,
    lease_expires_at,
    mark_updated_after_dispatch,
    operation_step_ids,
    parse_utc,
    safe_wal_name,
    stamp_entries,
    step_from_input,
    step_wal_payload,
    sync_task_roots,
    task_id_from_unavailable_wal,
    task_wal_dir,
    touch_steps,
    updated_after_dispatch_step_ids as collect_updated_after_dispatch_step_ids,
    validate_steps,
    with_wal_event_ids,
)
from agent_core.tasks.models import Task, TaskStep
from agent_core.tasks.replay import replay_task_event
from agent_core.tasks.records import TaskRecord, UnavailableTaskRecord
from agent_core.tasks.state import (
    TERMINAL_STEP_STATUSES,
    TERMINAL_TASK_STATUSES,
    apply_operation,
    ensure_dispatchable,
    ensure_not_terminal,
    promote_ready,
    promote_running,
    ready_event_entries,
)
from agent_core.tasks.views import (
    active_task_summaries,
    dispatchable_steps_view,
    list_tasks_view,
    query_steps_view,
    task_detail,
    unavailable_task_summaries,
)
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.common import utc_iso, utc_now, validate_safe_id


class TaskService:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], TaskRecord] = {}
        self._terminal_records: dict[tuple[str, str], TaskRecord] = {}
        self._unavailable_records: dict[tuple[str, str], UnavailableTaskRecord] = {}

    def replay_project(self, project_dir: Path) -> None:
        task_root = project_dir / ".soong-agent" / "tasks"
        if not task_root.exists():
            return
        for wal in sorted(task_root.glob("*/*.wal.jsonl")):
            try:
                self.replay_wal(wal)
            except Exception as exc:
                self._mark_wal_unavailable(wal, exc)

    def replay_wal(self, wal_path: Path) -> Task | None:
        task: Task | None = None
        session_id: str | None = None
        wal_seq = 0
        first_created_at: str | None = None
        last_created_at: str | None = None
        for line in wal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            wal_seq = max(wal_seq, int(event.get("wal_seq") or 0))
            session_id = event.get("session_id") or session_id
            created_at = event.get("created_at")
            if created_at:
                first_created_at = first_created_at or str(created_at)
                last_created_at = str(created_at)
            payload = event.get("payload") or {}
            event_type = event.get("event_type")
            if event_type == "task_created":
                task = Task.model_validate(payload["task"])
                task.wal_path = str(wal_path)
                task.created_at = task.created_at or first_created_at
                task.updated_at = task.updated_at or last_created_at or task.created_at
                sync_task_roots(task)
                for step in task.steps:
                    step.updated_at = step.updated_at or task.created_at
            elif task is not None:
                replay_task_event(task, event_type, event.get("step_id"), payload, created_at=created_at)
        if task is None or session_id is None:
            return None
        task.wal_path = str(wal_path)
        sync_task_roots(task)
        task.created_at = task.created_at or first_created_at
        task.updated_at = task.updated_at or last_created_at or task.created_at
        record = TaskRecord(task=task, wal_path=wal_path)
        record.wal_seq = wal_seq
        if task.status in TERMINAL_TASK_STATUSES:
            self._terminal_records[(session_id, task.task_id)] = record
        else:
            self._records[(session_id, task.task_id)] = record
        return task

    def create_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "create task DAG")
        task_id = validate_safe_id(str(args["task_id"]), field_name="task_id")
        key = (context.session_id, task_id)
        if key in self._records and self._records[key].task.status not in TERMINAL_TASK_STATUSES:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"active task_id already exists: {task_id}")
        wal_name = safe_wal_name(str(args["wal_name"]))
        wal_dir = task_wal_dir(context) / context.session_id
        wal_path = wal_dir / wal_name
        if wal_path.exists():
            raise AgentCoreError(ErrorCode.PATH_CONFLICT, f"task WAL already exists: {wal_name}")
        steps = [step_from_input(raw) for raw in args.get("steps", [])]
        validate_steps(steps)
        created_at = utc_iso()
        task = Task(
            task_id=task_id,
            wal_name=wal_name,
            wal_path=str(wal_path),
            title=str(args["title"]),
            summary=str(args.get("summary", "")),
            created_by_agent_id=context.agent_id,
            created_by_run_id=context.run_id,
            created_at=created_at,
            updated_at=created_at,
            steps=steps,
        )
        sync_task_roots(task)
        touch_steps(task, [step.step_id for step in steps], created_at)
        ready_transitions = promote_ready(task)
        record = TaskRecord(task=task, wal_path=wal_path)
        previous_task_status = promote_running(task)
        entries = [
            {
                "event_type": "task_created",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {"task": task.model_dump(mode="json")},
            }
        ]
        entries.extend(ready_event_entries(context, ready_transitions))
        if previous_task_status is not None and task.status == "running":
            entries.append(
                {
                    "event_type": "task_running",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {
                        "previous_status": previous_task_status,
                        "status": "running",
                        "reason": "ready_step_available",
                    },
                }
            )
        stamp_entries(entries, created_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        self._records[key] = record
        return with_wal_event_ids({"task": task.model_dump(mode="json"), "wal_path": str(record.wal_path)}, event_ids)

    def get_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_task_reader(context, "read task DAG")
        task_id = str(args["task_id"])
        ensure_worker_task_scope(context, task_id)
        self._raise_if_unavailable(context.session_id, task_id)
        record = self._record(context.session_id, task_id, include_terminal=True)
        task = record.task
        if task.status not in TERMINAL_TASK_STATUSES:
            self._reconcile_leases(context, record)
        return task_detail(record, include_terminal_steps=bool(args.get("include_terminal_steps", False)))

    def list_tasks(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "list task DAGs")
        return list_tasks_view(self._records, self._terminal_records, self._unavailable_records, context, args)

    def active_task_summaries(self, session_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return active_task_summaries(self._records, session_id, limit=limit)

    def unavailable_task_summaries(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return unavailable_task_summaries(self._unavailable_records, session_id)

    def update_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "modify task DAG")
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        ensure_not_terminal(task)
        operations = args.get("operations") or []
        next_task = task.model_copy(deep=True)
        previous_task_status = task.status
        reopen_events: list[dict[str, Any]] = []
        task_reopened: dict[str, Any] | None = None
        for op in operations:
            if op.get("op") == "reopen_step":
                previous_step = find_step(task, str(op["step_id"]))
                if previous_step.status in {"blocked", "failed"}:
                    reopen_events.append(
                        {
                            "step_id": previous_step.step_id,
                            "previous_status": previous_step.status,
                            "reason": op.get("reason"),
                        }
                    )
            elif op.get("op") == "update_task" and op.get("status") == "pending" and next_task.status == "blocked":
                task_reopened = {"previous_status": next_task.status, "reason": op.get("reason")}
            apply_operation(next_task, op)
        validate_steps(next_task.steps)
        ready_transitions = promote_ready(next_task)
        task_running_previous = promote_running(next_task)
        updated_at = utc_iso()
        sync_task_roots(next_task)
        next_task.updated_at = updated_at
        updated_after_dispatch_step_ids = collect_updated_after_dispatch_step_ids(task, operations)
        mark_updated_after_dispatch(next_task, updated_after_dispatch_step_ids)
        touched_step_ids = operation_step_ids(operations)
        touched_step_ids.extend(step.step_id for step, _previous in ready_transitions)
        touch_steps(next_task, touched_step_ids, updated_at)
        entries = [
            {
                "event_type": "task_updated",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {
                    "operations": operations,
                    "updated_after_dispatch_step_ids": updated_after_dispatch_step_ids,
                    "task_summary": next_task.model_dump(mode="json"),
                },
            }
        ]
        for reopened in reopen_events:
            entries.append(
                {
                    "event_type": "task_step_reopened",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "step_id": reopened["step_id"],
                    "payload": {
                        "previous_status": reopened["previous_status"],
                        "status": "pending",
                        "reason": reopened["reason"],
                    },
                }
            )
        if task_reopened is not None:
            entries.append(
                {
                    "event_type": "task_reopened",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {
                        "previous_status": task_reopened["previous_status"],
                        "status": "pending",
                        "reason": task_reopened["reason"],
                    },
                }
            )
        entries.extend(ready_event_entries(context, ready_transitions))
        if task_running_previous is not None and next_task.status == "running":
            entries.append(
                {
                    "event_type": "task_running",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {
                        "previous_status": previous_task_status,
                        "status": "running",
                        "reason": "ready_step_available",
                    },
                }
            )
        stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return with_wal_event_ids({"task": next_task.model_dump(mode="json")}, event_ids)

    def query_steps(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_task_reader(context, "query task steps")
        task_id = str(args["task_id"])
        ensure_worker_task_scope(context, task_id)
        self._raise_if_unavailable(context.session_id, task_id)
        record = self._record(context.session_id, task_id, include_terminal=bool(args.get("include_terminal_steps", False)))
        if record.task.status not in TERMINAL_TASK_STATUSES:
            self._reconcile_leases(context, record)
        return query_steps_view(record, context, args)

    def dispatchable_steps(
        self,
        context: ToolExecutionContext,
        *,
        task_id: str,
        worker_pool_id: str,
        allowed_step_ids: list[str] | None = None,
    ) -> list[TaskStep]:
        record = self._record(context.session_id, task_id)
        self._reconcile_leases(context, record)
        ensure_dispatchable(record.task)
        return dispatchable_steps_view(record.task, worker_pool_id=worker_pool_id, allowed_step_ids=allowed_step_ids)

    def claim_step(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_role(context, {"orchestrator", "worker"}, "claim task step")
        ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        ensure_dispatchable(task)
        next_task = task.model_copy(deep=True)
        step = find_step(next_task, str(args["step_id"]))
        if not step_visible_to_worker(context, step):
            raise AgentCoreError(ErrorCode.PERMISSION_DENIED, "worker cannot claim step outside dispatch scope")
        existing = [s for s in next_task.steps if s.claimed_by_run_id == context.run_id and s.status in {"claimed", "running"}]
        if existing:
            raise AgentCoreError(ErrorCode.STEP_ALREADY_CLAIMED_BY_RUN, "worker run already claimed a step")
        if step.status != "ready":
            if step.status in {"claimed", "running"}:
                raise AgentCoreError(ErrorCode.STEP_ALREADY_CLAIMED, "step already claimed")
            raise AgentCoreError(ErrorCode.STEP_NOT_READY, f"step is not ready: {step.step_id}")
        previous = step.status
        step.status = "claimed"
        step.claimed_by_agent_id = context.agent_id
        step.claimed_by_run_id = context.run_id
        step.lease_expires_at = lease_expires_at(context)
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        step.updated_at = updated_at
        stamp_entries(
            entries := [
                {
                    "event_type": "task_step_claimed",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "step_id": step.step_id,
                    "payload": {
                        "previous_status": previous,
                        "claimed_by_agent_id": context.agent_id,
                        "claimed_by_run_id": context.run_id,
                        "lease_expires_at": step.lease_expires_at,
                    },
                }
            ],
            updated_at,
        )
        event_ids = record.append_many(
            session_id=context.session_id,
            entries=entries,
        )
        record.task = next_task
        return with_wal_event_ids({"step": step.model_dump(mode="json")}, event_ids)

    def update_step(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_role(context, {"orchestrator", "worker"}, "update task step")
        ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        step = find_step(next_task, str(args["step_id"]))
        if context.agent_role == "worker" and step.claimed_by_run_id != context.run_id:
            raise AgentCoreError(ErrorCode.PERMISSION_DENIED, "worker can only update its claimed step")
        status = args.get("status")
        previous = step.status
        previous_task_status = next_task.status
        if args.get("result_summary") is not None:
            step.result_summary = str(args["result_summary"])
        if args.get("artifact_ids") is not None:
            step.artifact_ids = [str(item) for item in args["artifact_ids"]]
        if args.get("reason") is not None:
            step.reason = str(args["reason"])
        event_type = "task_step_updated"
        if status:
            status = str(status)
            if status == "running":
                step.status = "running"
                step.lease_expires_at = lease_expires_at(context)
                event_type = "task_step_started"
            elif status == "blocked":
                step.status = "blocked"
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
                event_type = "task_step_blocked"
            elif status == "completed":
                step.status = "completed"
                step.lease_expires_at = None
                event_type = "task_step_completed"
            elif status == "failed":
                step.status = "failed"
                step.lease_expires_at = None
                event_type = "task_step_failed"
            elif status == "cancelled":
                if previous in {"claimed", "running"}:
                    raise AgentCoreError(
                        ErrorCode.TASK_NOT_DISPATCHABLE,
                        "cannot directly cancel claimed/running step",
                    )
                step.status = "cancelled"
                step.lease_expires_at = None
                event_type = "task_step_cancelled"
            else:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported step status: {status}")
        ready_transitions = promote_ready(next_task)
        task_running_previous = promote_running(next_task)
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        touch_steps(next_task, [step.step_id, *(transition.step_id for transition, _previous in ready_transitions)], updated_at)
        entries = [
            {
                "event_type": event_type,
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "step_id": step.step_id,
                "payload": step_wal_payload(event_type, step, previous_status=previous),
            }
        ]
        entries.extend(ready_event_entries(context, ready_transitions))
        if task_running_previous is not None and next_task.status == "running":
            entries.append(
                {
                    "event_type": "task_running",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {
                        "previous_status": previous_task_status,
                        "status": "running",
                        "reason": "ready_step_available",
                    },
                }
            )
        stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return with_wal_event_ids({"step": step.model_dump(mode="json")}, event_ids)

    def complete_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "complete task DAG")
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        for step in next_task.steps:
            if step.status in {"claimed", "running"}:
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "task has claimed/running steps")
            if step.required and step.status != "completed":
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "required steps are not completed")
        cancelled_optional: list[str] = []
        entries: list[dict[str, Any]] = []
        for step in next_task.steps:
            if not step.required and step.status not in TERMINAL_STEP_STATUSES:
                previous = step.status
                step.status = "cancelled"
                step.reason = "task_completed"
                cancelled_optional.append(step.step_id)
                entries.append(
                    {
                        "event_type": "task_step_cancelled",
                        "actor_agent_id": context.agent_id,
                        "actor_run_id": context.run_id,
                        "step_id": step.step_id,
                        "payload": {
                            "previous_status": previous,
                            "reason": "task_completed",
                            "result_summary": step.result_summary,
                        },
                    }
                )
        next_task.status = "completed"
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        touch_steps(next_task, cancelled_optional, updated_at)
        entries.append(
            {
                "event_type": "task_completed",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {"result_summary": args.get("result_summary"), "cancelled_optional_step_ids": cancelled_optional},
            }
        )
        stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return with_wal_event_ids({"task": next_task.model_dump(mode="json")}, event_ids)

    def fail_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "fail task DAG")
        return self._terminate_task(context, args, status="failed", event_type="task_failed", step_status="failed", reason="task_failed")

    def cancel_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        ensure_orchestrator(context, "cancel task DAG")
        return self._terminate_task(context, args, status="cancelled", event_type="task_cancelled", step_status="cancelled", reason="task_cancelled")

    def _terminate_task(self, context: ToolExecutionContext, args: dict[str, Any], *, status: str, event_type: str, step_status: str, reason: str) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        changed: list[str] = []
        worker_run_ids: list[str] = []
        entries: list[dict[str, Any]] = []
        for step in next_task.steps:
            if step.status not in TERMINAL_STEP_STATUSES:
                if step.claimed_by_run_id:
                    worker_run_ids.append(step.claimed_by_run_id)
                previous = step.status
                step.status = step_status
                step.reason = reason
                step.result_summary = reason
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
                changed.append(step.step_id)
                entries.append(
                    {
                        "event_type": f"task_step_{step_status}",
                        "actor_agent_id": context.agent_id,
                        "actor_run_id": context.run_id,
                        "step_id": step.step_id,
                        "payload": step_wal_payload(f"task_step_{step_status}", step, previous_status=previous),
                    }
                )
        next_task.status = status
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        touch_steps(next_task, changed, updated_at)
        terminated = sorted(set(worker_run_ids))
        entries.append(
            {
                "event_type": event_type,
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {
                    "reason": args.get("reason"),
                    f"{step_status}_step_ids": changed,
                },
            }
        )
        stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return with_wal_event_ids({"task": next_task.model_dump(mode="json"), "terminated_worker_run_ids": terminated}, event_ids)

    def _reconcile_leases(self, context: ToolExecutionContext, record: TaskRecord) -> None:
        if record.task.status in TERMINAL_TASK_STATUSES:
            return
        next_task = record.task.model_copy(deep=True)
        expired: list[TaskStep] = []
        now = utc_now()
        for step in next_task.steps:
            if step.status not in {"claimed", "running"} or not step.lease_expires_at:
                continue
            if parse_utc(step.lease_expires_at) <= now:
                expired.append(step)
        entries: list[dict[str, Any]] = []
        for step in expired:
            previous = step.status
            claimed_by_agent_id = step.claimed_by_agent_id
            claimed_by_run_id = step.claimed_by_run_id
            expired_at = utc_iso(now)
            step.status = "pending"
            step.claimed_by_agent_id = None
            step.claimed_by_run_id = None
            step.lease_expires_at = None
            entries.append(
                {
                    "event_type": "task_step_lease_expired",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "step_id": step.step_id,
                    "payload": {
                        "previous_status": previous,
                        "claimed_by_agent_id": claimed_by_agent_id,
                        "claimed_by_run_id": claimed_by_run_id,
                        "expired_at": expired_at,
                    },
                }
            )
        if expired:
            previous_task_status = next_task.status
            ready_transitions = promote_ready(next_task)
            task_running_previous = promote_running(next_task)
            entries.extend(ready_event_entries(context, ready_transitions))
            if task_running_previous is not None and next_task.status == "running":
                entries.append(
                    {
                        "event_type": "task_running",
                        "actor_agent_id": context.agent_id,
                        "actor_run_id": context.run_id,
                        "payload": {
                            "previous_status": previous_task_status,
                            "status": "running",
                            "reason": "ready_step_available",
                        },
                    }
                )
            record.append_many(session_id=context.session_id, entries=entries)
            record.task = next_task

    def fail_unclosed_worker_step(
        self,
        context: ToolExecutionContext,
        *,
        task_id: str,
        worker_run_id: str,
        reason: str,
    ) -> dict[str, Any] | None:
        record = self._record(context.session_id, task_id, include_terminal=True)
        task = record.task
        if task.status in TERMINAL_TASK_STATUSES:
            return None
        next_task = task.model_copy(deep=True)
        for step in next_task.steps:
            if step.claimed_by_run_id == worker_run_id and step.status in {"claimed", "running"}:
                previous = step.status
                step.status = "failed"
                step.reason = reason
                step.result_summary = step.result_summary or reason
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
                record.append_many(
                    session_id=context.session_id,
                    entries=[
                        {
                            "event_type": "task_step_failed",
                            "actor_agent_id": context.agent_id,
                            "actor_run_id": context.run_id,
                            "step_id": step.step_id,
                            "payload": step_wal_payload("task_step_failed", step, previous_status=previous),
                        }
                    ],
                )
                record.task = next_task
                return {"step": step.model_dump(mode="json")}
        return None

    def claimed_step_for_run(self, session_id: str, task_id: str, run_id: str) -> TaskStep | None:
        record = self._record(session_id, task_id)
        for step in record.task.steps:
            if step.claimed_by_run_id == run_id:
                return step
        return None

    def _record(self, session_id: str, task_id: str, *, include_terminal: bool = False) -> TaskRecord:
        key = (session_id, task_id)
        record = self._records.get(key)
        if record is None and include_terminal:
            record = self._terminal_records.get(key)
        if record is None:
            raise AgentCoreError(ErrorCode.TASK_NOT_FOUND, f"task not found: {task_id}")
        return record

    def _raise_if_unavailable(self, session_id: str, task_id: str) -> None:
        record = self._unavailable_records.get((session_id, task_id))
        if record is None:
            return
        raise AgentCoreError(
            ErrorCode.TASK_WAL_UNAVAILABLE,
            f"failed to replay Task WAL: {record.wal_path}",
            details={"wal_path": str(record.wal_path), "error": record.error},
        )

    def _mark_wal_unavailable(self, wal_path: Path, exc: Exception) -> None:
        session_id = wal_path.parent.name
        task_id = task_id_from_unavailable_wal(wal_path)
        self._unavailable_records[(session_id, task_id)] = UnavailableTaskRecord(
            session_id=session_id,
            wal_path=wal_path,
            task_id=task_id,
            error=str(exc),
        )
