from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from agent_core.config.paths import expand_config_path
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.storage.task_wal import TaskWalWriter
from agent_core.tasks.models import Task, TaskStep
from agent_core.tasks.operations import validate_no_dependency_cycle
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.common import utc_iso, utc_now, validate_safe_id


TERMINAL_STEP_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class TaskRecord:
    task: Task
    wal_path: Path
    wal_seq: int = 0
    writer: TaskWalWriter = field(init=False)

    def __post_init__(self) -> None:
        self.writer = TaskWalWriter(self.wal_path)

    def append(self, *, session_id: str, event_type: str, actor_agent_id: str, actor_run_id: str, payload: dict[str, Any], step_id: str | None = None) -> str | None:
        event_ids = self.append_many(
            session_id=session_id,
            entries=[
                {
                    "event_type": event_type,
                    "actor_agent_id": actor_agent_id,
                    "actor_run_id": actor_run_id,
                    "step_id": step_id,
                    "payload": payload,
                }
            ],
        )
        return event_ids[-1] if event_ids else None

    def append_many(self, *, session_id: str, entries: list[dict[str, Any]]) -> list[str]:
        if not entries:
            return []
        next_seq = self.wal_seq
        batch_created_at = utc_iso()
        payloads: list[dict[str, Any]] = []
        event_ids: list[str] = []
        for entry in entries:
            next_seq += 1
            event_id = f"task_evt_{next_seq}"
            event_ids.append(event_id)
            payloads.append(
                {
                    "wal_seq": next_seq,
                    "session_id": session_id,
                    "event_id": event_id,
                    "event_type": entry["event_type"],
                    "actor_agent_id": entry["actor_agent_id"],
                    "actor_run_id": entry["actor_run_id"],
                    "task_id": self.task.task_id,
                    "step_id": entry.get("step_id"),
                    "payload": entry["payload"],
                    "created_at": entry.get("created_at") or batch_created_at,
                }
            )
        try:
            self.writer.append_many(payloads)
        except OSError as exc:
            raise AgentCoreError(
                ErrorCode.TASK_WAL_UNAVAILABLE,
                f"failed to append Task WAL: {self.wal_path}",
                details={"wal_path": str(self.wal_path)},
                cause=exc,
            ) from exc
        self.wal_seq = next_seq
        return event_ids


@dataclass(frozen=True)
class UnavailableTaskRecord:
    session_id: str
    wal_path: Path
    task_id: str
    error: str

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "wal_path": str(self.wal_path),
            "status": "unavailable",
            "error": {
                "code": ErrorCode.TASK_WAL_UNAVAILABLE.value,
                "message": f"failed to replay Task WAL: {self.wal_path}",
                "details": {"wal_path": str(self.wal_path), "error": self.error},
            },
        }


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
                _sync_task_roots(task)
                for step in task.steps:
                    step.updated_at = step.updated_at or task.created_at
            elif task is not None:
                self._replay_event(task, event_type, event.get("step_id"), payload, created_at=created_at)
        if task is None or session_id is None:
            return None
        task.wal_path = str(wal_path)
        _sync_task_roots(task)
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
        _ensure_orchestrator(context, "create task DAG")
        task_id = validate_safe_id(str(args["task_id"]), field_name="task_id")
        key = (context.session_id, task_id)
        if key in self._records and self._records[key].task.status not in TERMINAL_TASK_STATUSES:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"active task_id already exists: {task_id}")
        wal_name = _safe_wal_name(str(args["wal_name"]))
        wal_dir = _task_wal_dir(context) / context.session_id
        wal_path = wal_dir / wal_name
        if wal_path.exists():
            raise AgentCoreError(ErrorCode.PATH_CONFLICT, f"task WAL already exists: {wal_name}")
        steps = [_step_from_input(raw) for raw in args.get("steps", [])]
        _validate_steps(steps)
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
        _sync_task_roots(task)
        _touch_steps(task, [step.step_id for step in steps], created_at)
        ready_transitions = self._promote_ready(task)
        record = TaskRecord(task=task, wal_path=wal_path)
        previous_task_status = self._promote_running(task)
        entries = [
            {
                "event_type": "task_created",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {"task": task.model_dump(mode="json")},
            }
        ]
        entries.extend(self._ready_event_entries(context, ready_transitions))
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
        _stamp_entries(entries, created_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        self._records[key] = record
        return _with_wal_event_ids({"task": task.model_dump(mode="json"), "wal_path": str(record.wal_path)}, event_ids)

    def get_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_task_reader(context, "read task DAG")
        task_id = str(args["task_id"])
        self._ensure_worker_task_scope(context, task_id)
        self._raise_if_unavailable(context.session_id, task_id)
        record = self._record(context.session_id, task_id, include_terminal=True)
        task = record.task
        if task.status not in TERMINAL_TASK_STATUSES:
            self._reconcile_leases(context, record)
        include_terminal = bool(args.get("include_terminal_steps", False))
        data = task.model_dump(mode="json")
        if not include_terminal:
            data["steps"] = [step for step in data["steps"] if step["status"] not in TERMINAL_STEP_STATUSES]
        return {"task": data, "wal_path": str(record.wal_path)}

    def list_tasks(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_orchestrator(context, "list task DAGs")
        status = args.get("status")
        include_terminal = bool(args.get("include_terminal", False))
        limit = int(args.get("limit") or 50)
        offset = int(args.get("offset") or 0)
        selected_records: list[TaskRecord] = []
        records = list(self._records.items())
        if include_terminal:
            records.extend(self._terminal_records.items())
        for (session_id, _task_id), record in records:
            if session_id == context.session_id and (not status or record.task.status == status):
                selected_records.append(record)
        selected_records.sort(key=lambda record: _task_updated_sort_key(record.task, record.wal_path), reverse=True)
        sliced = selected_records[offset : offset + limit]
        task_summaries: list[dict[str, Any]] = [record.task.model_dump(mode="json") for record in sliced]
        total = len(selected_records)
        if include_terminal and (not status or status == "unavailable"):
            unavailable = [
                record
                for (session_id, _task_id), record in self._unavailable_records.items()
                if session_id == context.session_id
            ]
            unavailable.sort(key=lambda record: _wal_path_sort_key(record.wal_path), reverse=True)
            available_remaining = max(0, limit - len(task_summaries))
            unavailable_offset = max(0, offset - total)
            if available_remaining > 0:
                task_summaries.extend(
                    record.summary()
                    for record in unavailable[unavailable_offset : unavailable_offset + available_remaining]
                )
            total += len(unavailable)
        return {"tasks": task_summaries, "truncated": offset + limit < total}

    def active_task_summaries(self, session_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for (record_session_id, _task_id), record in self._records.items():
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
                    "steps": [
                        {
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
                        for step in steps
                    ],
                }
            )
        return summaries[:limit]

    def unavailable_task_summaries(self, session_id: str | None = None) -> list[dict[str, Any]]:
        records = [
            record
            for (record_session_id, _task_id), record in self._unavailable_records.items()
            if session_id is None or record_session_id == session_id
        ]
        records.sort(key=lambda record: _wal_path_sort_key(record.wal_path), reverse=True)
        return [record.summary() for record in records]

    def update_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_orchestrator(context, "modify task DAG")
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        self._ensure_not_terminal(task)
        operations = args.get("operations") or []
        next_task = task.model_copy(deep=True)
        previous_task_status = task.status
        reopen_events: list[dict[str, Any]] = []
        task_reopened: dict[str, Any] | None = None
        for op in operations:
            if op.get("op") == "reopen_step":
                previous_step = _find_step(task, str(op["step_id"]))
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
            self._apply_operation(next_task, op)
        _validate_steps(next_task.steps)
        ready_transitions = self._promote_ready(next_task)
        task_running_previous = self._promote_running(next_task)
        updated_at = utc_iso()
        _sync_task_roots(next_task)
        next_task.updated_at = updated_at
        updated_after_dispatch_step_ids = _updated_after_dispatch_step_ids(task, operations)
        _mark_updated_after_dispatch(next_task, updated_after_dispatch_step_ids)
        touched_step_ids = _operation_step_ids(operations)
        touched_step_ids.extend(step.step_id for step, _previous in ready_transitions)
        _touch_steps(next_task, touched_step_ids, updated_at)
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
        entries.extend(self._ready_event_entries(context, ready_transitions))
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
        _stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return _with_wal_event_ids({"task": next_task.model_dump(mode="json")}, event_ids)

    def query_steps(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_task_reader(context, "query task steps")
        statuses = set(args.get("statuses") or [])
        include_terminal = bool(args.get("include_terminal_steps", False))
        task_id = str(args["task_id"])
        self._ensure_worker_task_scope(context, task_id)
        self._raise_if_unavailable(context.session_id, task_id)
        record = self._record(context.session_id, task_id, include_terminal=include_terminal)
        if record.task.status not in TERMINAL_TASK_STATUSES:
            self._reconcile_leases(context, record)
        worker_pool_id = args.get("worker_pool_id")
        claimed_by_agent_id = args.get("claimed_by_agent_id")
        default_limit = 5 if context.agent_role == "worker" and statuses == {"ready"} else 50
        limit = int(args["limit"]) if args.get("limit") is not None else default_limit
        offset = int(args.get("offset") or 0)
        steps = []
        for step in record.task.steps:
            if not self._step_visible_to_worker(context, step):
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
        self._ensure_dispatchable(record.task)
        allowed = set(allowed_step_ids) if allowed_step_ids is not None else None
        steps: list[TaskStep] = []
        for step in record.task.steps:
            if step.status != "ready":
                continue
            if allowed is not None and step.step_id not in allowed:
                continue
            if step.worker_pool_id is not None and step.worker_pool_id != worker_pool_id:
                continue
            steps.append(step)
        return steps

    def claim_step(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_role(context, {"orchestrator", "worker"}, "claim task step")
        self._ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        self._ensure_dispatchable(task)
        next_task = task.model_copy(deep=True)
        step = _find_step(next_task, str(args["step_id"]))
        if not self._step_visible_to_worker(context, step):
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
        step.lease_expires_at = _lease_expires_at(context)
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        step.updated_at = updated_at
        _stamp_entries(
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
        return _with_wal_event_ids({"step": step.model_dump(mode="json")}, event_ids)

    def update_step(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_role(context, {"orchestrator", "worker"}, "update task step")
        self._ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        self._ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        step = _find_step(next_task, str(args["step_id"]))
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
                step.lease_expires_at = _lease_expires_at(context)
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
        ready_transitions = self._promote_ready(next_task)
        task_running_previous = self._promote_running(next_task)
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        _touch_steps(next_task, [step.step_id, *(transition.step_id for transition, _previous in ready_transitions)], updated_at)
        entries = [
            {
                "event_type": event_type,
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "step_id": step.step_id,
                "payload": _step_wal_payload(event_type, step, previous_status=previous),
            }
        ]
        entries.extend(self._ready_event_entries(context, ready_transitions))
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
        _stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return _with_wal_event_ids({"step": step.model_dump(mode="json")}, event_ids)

    def complete_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_orchestrator(context, "complete task DAG")
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        self._ensure_not_terminal(task)
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
        _touch_steps(next_task, cancelled_optional, updated_at)
        entries.append(
            {
                "event_type": "task_completed",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {"result_summary": args.get("result_summary"), "cancelled_optional_step_ids": cancelled_optional},
            }
        )
        _stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return _with_wal_event_ids({"task": next_task.model_dump(mode="json")}, event_ids)

    def fail_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_orchestrator(context, "fail task DAG")
        return self._terminate_task(context, args, status="failed", event_type="task_failed", step_status="failed", reason="task_failed")

    def cancel_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        _ensure_orchestrator(context, "cancel task DAG")
        return self._terminate_task(context, args, status="cancelled", event_type="task_cancelled", step_status="cancelled", reason="task_cancelled")

    def _terminate_task(self, context: ToolExecutionContext, args: dict[str, Any], *, status: str, event_type: str, step_status: str, reason: str) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]), include_terminal=True)
        task = record.task
        self._ensure_not_terminal(task)
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
                        "payload": _step_wal_payload(f"task_step_{step_status}", step, previous_status=previous),
                    }
                )
        next_task.status = status
        updated_at = utc_iso()
        next_task.updated_at = updated_at
        _touch_steps(next_task, changed, updated_at)
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
        _stamp_entries(entries, updated_at)
        event_ids = record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return _with_wal_event_ids({"task": next_task.model_dump(mode="json"), "terminated_worker_run_ids": terminated}, event_ids)

    def _apply_operation(self, task: Task, op: dict[str, Any]) -> None:
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
            step = _step_from_input(op["step"])
            if any(existing.step_id == step.step_id for existing in task.steps):
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"duplicate step_id: {step.step_id}")
            task.steps.append(step)
        elif kind == "update_step":
            step = _find_step(task, str(op["step_id"]))
            for field in ("title", "summary", "worker_pool_id", "required"):
                if field in op:
                    setattr(step, field, op[field])
            if "depends_on_step_ids" in op:
                step.depends_on_step_ids = [validate_safe_id(str(dep), field_name="depends_on_step_id") for dep in op.get("depends_on_step_ids") or []]
        elif kind == "delete_step":
            step_id = str(op["step_id"])
            step = _find_step(task, step_id)
            if step.status not in {"pending", "ready", "cancelled"}:
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, f"cannot delete {step.status} step")
            if any(step_id in step.depends_on_step_ids for step in task.steps):
                raise AgentCoreError(ErrorCode.STEP_HAS_DEPENDENTS, "step has dependents")
            task.steps = [step for step in task.steps if step.step_id != step_id]
        elif kind == "add_dependency":
            step = _find_step(task, str(op["step_id"]))
            dep = str(op["depends_on_step_id"])
            _find_step(task, dep)
            if dep not in step.depends_on_step_ids:
                step.depends_on_step_ids.append(dep)
        elif kind == "remove_dependency":
            step = _find_step(task, str(op["step_id"]))
            dep = str(op["depends_on_step_id"])
            step.depends_on_step_ids = [item for item in step.depends_on_step_ids if item != dep]
        elif kind == "cancel_step":
            step = _find_step(task, str(op["step_id"]))
            if step.status in {"claimed", "running"}:
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "cannot directly cancel claimed/running step")
            step.status = "cancelled"
            step.reason = op.get("reason")
        elif kind == "reopen_step":
            step = _find_step(task, str(op["step_id"]))
            if step.status == "cancelled":
                raise AgentCoreError(ErrorCode.TASK_TERMINAL, "cancelled step cannot reopen")
            if step.status in {"blocked", "failed"}:
                step.status = "pending"
                step.reason = op.get("reason")
        else:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported task_update op: {kind}")

    def _promote_ready(self, task: Task) -> list[tuple[TaskStep, str]]:
        completed = {step.step_id for step in task.steps if step.status == "completed"}
        transitions: list[tuple[TaskStep, str]] = []
        for step in task.steps:
            if step.status in {"pending", "ready"}:
                previous = step.status
                step.status = "ready" if all(dep in completed for dep in step.depends_on_step_ids) else "pending"
                if previous != step.status:
                    transitions.append((step, previous))
        return transitions

    def _promote_running(self, task: Task) -> str | None:
        if task.status == "pending" and any(step.status in {"ready", "claimed", "running"} for step in task.steps):
            previous = task.status
            task.status = "running"
            return previous
        return None

    def _reconcile_leases(self, context: ToolExecutionContext, record: TaskRecord) -> None:
        if record.task.status in TERMINAL_TASK_STATUSES:
            return
        next_task = record.task.model_copy(deep=True)
        expired: list[TaskStep] = []
        now = utc_now()
        for step in next_task.steps:
            if step.status not in {"claimed", "running"} or not step.lease_expires_at:
                continue
            if _parse_utc(step.lease_expires_at) <= now:
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
            ready_transitions = self._promote_ready(next_task)
            task_running_previous = self._promote_running(next_task)
            entries.extend(self._ready_event_entries(context, ready_transitions))
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
                            "payload": _step_wal_payload("task_step_failed", step, previous_status=previous),
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

    def _ensure_worker_task_scope(self, context: ToolExecutionContext, task_id: str) -> None:
        if context.agent_role != "worker":
            return
        scope = _worker_scope(context)
        if scope.get("task_id") != task_id:
            raise AgentCoreError(ErrorCode.PERMISSION_DENIED, "worker cannot access task outside dispatch scope")

    def _step_visible_to_worker(self, context: ToolExecutionContext, step: TaskStep) -> bool:
        if context.agent_role != "worker":
            return True
        scope = _worker_scope(context)
        allowed_step_ids = scope.get("allowed_step_ids")
        if allowed_step_ids is not None and step.step_id not in allowed_step_ids:
            return False
        worker_pool_id = scope.get("worker_pool_id")
        if worker_pool_id and step.worker_pool_id and step.worker_pool_id != worker_pool_id:
            return False
        return True

    def _ensure_not_terminal(self, task: Task) -> None:
        if task.status in TERMINAL_TASK_STATUSES:
            raise AgentCoreError(ErrorCode.TASK_TERMINAL, f"task is terminal: {task.status}")

    def _ensure_dispatchable(self, task: Task) -> None:
        self._ensure_not_terminal(task)
        if task.status == "blocked":
            raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "task is blocked")

    def _append_ready_events(
        self,
        record: TaskRecord,
        context: ToolExecutionContext,
        transitions: list[tuple[TaskStep, str]],
    ) -> None:
        record.append_many(session_id=context.session_id, entries=self._ready_event_entries(context, transitions))

    def _ready_event_entries(
        self,
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
        task_id = _task_id_from_unavailable_wal(wal_path)
        self._unavailable_records[(session_id, task_id)] = UnavailableTaskRecord(
            session_id=session_id,
            wal_path=wal_path,
            task_id=task_id,
            error=str(exc),
        )

    def _replay_event(self, task: Task, event_type: str, step_id: str | None, payload: dict[str, Any], *, created_at: str | None = None) -> None:
        if event_type == "task_updated":
            if payload.get("task_summary"):
                _replace_task(task, Task.model_validate(payload["task_summary"]))
            else:
                for op in payload.get("operations") or []:
                    self._apply_operation(task, op)
                self._promote_ready(task)
                self._promote_running(task)
        elif event_type == "task_running":
            task.status = "running"
        elif event_type == "task_reopened":
            task.status = "pending"
            self._promote_ready(task)
            self._promote_running(task)
        elif event_type == "task_completed":
            task.status = "completed"
            for cancelled_step_id in payload.get("cancelled_optional_step_ids") or []:
                step = _find_step(task, str(cancelled_step_id))
                step.status = "cancelled"
                step.reason = "task_completed"
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
        elif event_type == "task_failed":
            task.status = "failed"
            for failed_step_id in payload.get("failed_step_ids") or []:
                step = _find_step(task, str(failed_step_id))
                step.status = "failed"
                step.reason = "task_failed"
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
        elif event_type == "task_cancelled":
            task.status = "cancelled"
            for cancelled_step_id in payload.get("cancelled_step_ids") or []:
                step = _find_step(task, str(cancelled_step_id))
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
            step = _find_step(task, step_id)
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
            self._promote_ready(task)
            self._promote_running(task)
            step.updated_at = created_at or step.updated_at
        if created_at:
            task.updated_at = created_at


def _step_from_input(raw: dict[str, Any]) -> TaskStep:
    return TaskStep(
        step_id=validate_safe_id(str(raw["step_id"]), field_name="step_id"),
        title=str(raw["title"]),
        summary=str(raw.get("summary", "")),
        status=str(raw.get("status", "pending")),
        depends_on_step_ids=[validate_safe_id(str(dep), field_name="depends_on_step_id") for dep in raw.get("depends_on_step_ids", [])],
        required=bool(raw.get("required", True)),
        worker_pool_id=raw.get("worker_pool_id"),
    )


def _step_wal_payload(event_type: str, step: TaskStep, *, previous_status: str) -> dict[str, Any]:
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


def _stamp_entries(entries: list[dict[str, Any]], created_at: str) -> None:
    for entry in entries:
        entry["created_at"] = created_at


def _touch_steps(task: Task, step_ids: Iterable[str], updated_at: str) -> None:
    wanted = {str(step_id) for step_id in step_ids}
    if not wanted:
        return
    for step in task.steps:
        if step.step_id in wanted:
            step.updated_at = updated_at


def _sync_task_roots(task: Task) -> None:
    task.root_step_ids = [step.step_id for step in task.steps if not step.depends_on_step_ids]


def _operation_step_ids(operations: list[dict[str, Any]]) -> list[str]:
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


def _replace_task(target: Task, source: Task) -> None:
    for field_name in source.__class__.model_fields:
        setattr(target, field_name, getattr(source, field_name))


def _task_updated_sort_key(task: Task, wal_path: Path) -> float:
    if task.updated_at:
        try:
            return _parse_utc(task.updated_at).timestamp()
        except ValueError:
            pass
    return _wal_path_sort_key(wal_path)


def _wal_path_sort_key(wal_path: Path) -> float:
    try:
        return wal_path.stat().st_mtime
    except OSError:
        return 0.0


def _task_id_from_unavailable_wal(wal_path: Path) -> str:
    name = wal_path.name
    suffix = ".wal.jsonl"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    try:
        return validate_safe_id(name, field_name="task_id")
    except ValueError:
        return f"unavailable_{abs(hash(str(wal_path))) & 0xFFFFFFFF:x}"


def _validate_steps(steps: list[TaskStep]) -> None:
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


def _find_step(task: Task, step_id: str) -> TaskStep:
    for step in task.steps:
        if step.step_id == step_id:
            return step
    raise AgentCoreError(ErrorCode.STEP_NOT_FOUND, f"step not found: {step_id}")


def _task_wal_dir(context: ToolExecutionContext) -> Path:
    return expand_config_path(context.config.task.wal_dir, home_dir=context.home_dir, project_dir=context.project_dir).resolve()


def _worker_scope(context: ToolExecutionContext) -> dict[str, Any]:
    if not context.services:
        return {}
    return dict(context.services.get("worker_scope") or {})


def _ensure_orchestrator(context: ToolExecutionContext, action: str) -> None:
    _ensure_role(context, {"orchestrator"}, action)


def _ensure_task_reader(context: ToolExecutionContext, action: str) -> None:
    _ensure_role(context, {"orchestrator", "worker"}, action)
    if context.agent_role == "worker" and not _worker_scope(context).get("task_id"):
        raise AgentCoreError(ErrorCode.PERMISSION_DENIED, f"worker cannot {action} without dispatch scope")


def _ensure_role(context: ToolExecutionContext, allowed_roles: set[str], action: str) -> None:
    if context.agent_role not in allowed_roles:
        roles = ", ".join(sorted(allowed_roles))
        raise AgentCoreError(ErrorCode.PERMISSION_DENIED, f"{context.agent_role} agent cannot {action}; allowed roles: {roles}")


def _lease_expires_at(context: ToolExecutionContext) -> str:
    return utc_iso(utc_now() + timedelta(milliseconds=context.config.task.step_lease_timeout_ms))


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _updated_after_dispatch_step_ids(task: Task, operations: list[dict[str, Any]]) -> list[str]:
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


def _mark_updated_after_dispatch(task: Task, step_ids: list[str]) -> None:
    if not step_ids:
        return
    targets = set(step_ids)
    for step in task.steps:
        if step.step_id in targets:
            step.metadata = {**step.metadata, "updated_after_dispatch": True}


def _with_wal_event_ids(data: dict[str, Any], event_ids: list[str]) -> dict[str, Any]:
    if not event_ids:
        return data
    return {
        **data,
        "wal_event_id": event_ids[-1],
        "wal_event_ids": event_ids,
    }


def _safe_wal_name(value: str) -> str:
    name = Path(value).name
    if name != value or not name.endswith(".wal.jsonl"):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "wal_name must be a local *.wal.jsonl filename")
    return name
