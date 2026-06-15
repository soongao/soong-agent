from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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

    def append(self, *, session_id: str, event_type: str, actor_agent_id: str, actor_run_id: str, payload: dict[str, Any], step_id: str | None = None) -> None:
        self.append_many(
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

    def append_many(self, *, session_id: str, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        next_seq = self.wal_seq
        payloads: list[dict[str, Any]] = []
        for entry in entries:
            next_seq += 1
            payloads.append(
                {
                    "wal_seq": next_seq,
                    "session_id": session_id,
                    "event_id": f"task_evt_{next_seq}",
                    "event_type": entry["event_type"],
                    "actor_agent_id": entry["actor_agent_id"],
                    "actor_run_id": entry["actor_run_id"],
                    "task_id": self.task.task_id,
                    "step_id": entry.get("step_id"),
                    "payload": entry["payload"],
                    "created_at": utc_iso(),
                }
            )
        self.writer.append_many(payloads)
        self.wal_seq = next_seq


class TaskService:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], TaskRecord] = {}

    def replay_project(self, project_dir: Path) -> None:
        task_root = project_dir / ".soong-agent" / "tasks"
        if not task_root.exists():
            return
        for wal in sorted(task_root.glob("*/*.wal.jsonl")):
            self.replay_wal(wal)

    def replay_wal(self, wal_path: Path) -> Task | None:
        task: Task | None = None
        session_id: str | None = None
        wal_seq = 0
        for line in wal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            wal_seq = max(wal_seq, int(event.get("wal_seq") or 0))
            session_id = event.get("session_id") or session_id
            payload = event.get("payload") or {}
            event_type = event.get("event_type")
            if event_type == "task_created":
                task = Task.model_validate(payload["task"])
            elif task is not None:
                self._replay_event(task, event_type, event.get("step_id"), payload)
        if task is None or session_id is None:
            return None
        record = TaskRecord(task=task, wal_path=wal_path)
        record.wal_seq = wal_seq
        self._records[(session_id, task.task_id)] = record
        return task

    def create_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        task_id = validate_safe_id(str(args["task_id"]), field_name="task_id")
        key = (context.session_id, task_id)
        if key in self._records and self._records[key].task.status not in TERMINAL_TASK_STATUSES:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"active task_id already exists: {task_id}")
        wal_name = _safe_wal_name(str(args["wal_name"]))
        steps = [_step_from_input(raw) for raw in args.get("steps", [])]
        _validate_steps(steps)
        task = Task(
            task_id=task_id,
            wal_name=wal_name,
            title=str(args["title"]),
            summary=str(args.get("summary", "")),
            steps=steps,
        )
        ready_transitions = self._promote_ready(task)
        wal_dir = _task_wal_dir(context) / context.session_id
        wal_path = wal_dir / wal_name
        if wal_path.exists():
            raise AgentCoreError(ErrorCode.PATH_CONFLICT, f"task WAL already exists: {wal_name}")
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
        record.append_many(session_id=context.session_id, entries=entries)
        self._records[key] = record
        return {"task": task.model_dump(mode="json"), "wal_path": str(record.wal_path)}

    def get_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]))
        task = record.task
        self._reconcile_leases(context, record)
        include_terminal = bool(args.get("include_terminal_steps", False))
        data = task.model_dump(mode="json")
        if not include_terminal:
            data["steps"] = [step for step in data["steps"] if step["status"] not in TERMINAL_STEP_STATUSES]
        return {"task": data, "wal_path": str(record.wal_path)}

    def list_tasks(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        status = args.get("status")
        include_terminal = bool(args.get("include_terminal", False))
        limit = int(args.get("limit") or 50)
        offset = int(args.get("offset") or 0)
        tasks: list[Task] = []
        for (session_id, _task_id), record in self._records.items():
            if session_id != context.session_id:
                continue
            if status and record.task.status != status:
                continue
            if not include_terminal and record.task.status in TERMINAL_TASK_STATUSES:
                continue
            tasks.append(record.task)
        sliced = tasks[offset : offset + limit]
        return {"tasks": [task.model_dump(mode="json") for task in sliced], "truncated": offset + limit < len(tasks)}

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

    def update_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]))
        task = record.task
        self._ensure_not_terminal(task)
        operations = args.get("operations") or []
        next_task = task.model_copy(deep=True)
        previous_task_status = task.status
        reopen_events: list[dict[str, Any]] = []
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
            self._apply_operation(next_task, op)
        _validate_steps(next_task.steps)
        ready_transitions = self._promote_ready(next_task)
        task_running_previous = self._promote_running(next_task)
        entries = [
            {
                "event_type": "task_updated",
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "payload": {
                    "operations": operations,
                    "updated_after_dispatch_step_ids": _updated_after_dispatch_step_ids(task, operations),
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
        return {"task": next_task.model_dump(mode="json")}

    def query_steps(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]))
        self._reconcile_leases(context, record)
        statuses = set(args.get("statuses") or [])
        include_terminal = bool(args.get("include_terminal_steps", False))
        worker_pool_id = args.get("worker_pool_id")
        claimed_by_agent_id = args.get("claimed_by_agent_id")
        limit = int(args.get("limit") or 50)
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
        self._ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]))
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
        record.append_many(
            session_id=context.session_id,
            entries=[
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
        )
        record.task = next_task
        return {"step": step.model_dump(mode="json")}

    def update_step(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_worker_task_scope(context, str(args["task_id"]))
        record = self._record(context.session_id, str(args["task_id"]))
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
                step.status = "cancelled"
                step.lease_expires_at = None
                event_type = "task_step_cancelled"
            else:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported step status: {status}")
        ready_transitions = self._promote_ready(next_task)
        task_running_previous = self._promote_running(next_task)
        entries = [
            {
                "event_type": event_type,
                "actor_agent_id": context.agent_id,
                "actor_run_id": context.run_id,
                "step_id": step.step_id,
                "payload": {
                    "previous_status": previous,
                    "status": step.status,
                    "result_summary": step.result_summary,
                    "artifact_ids": step.artifact_ids,
                    "reason": step.reason,
                    "lease_expires_at": step.lease_expires_at,
                },
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
        record.append_many(session_id=context.session_id, entries=entries)
        record.task = next_task
        return {"step": step.model_dump(mode="json")}

    def complete_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]))
        task = record.task
        self._ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        for step in next_task.steps:
            if step.status in {"claimed", "running"}:
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "task has claimed/running steps")
            if step.required and step.status != "completed":
                raise AgentCoreError(ErrorCode.TASK_NOT_DISPATCHABLE, "required steps are not completed")
        cancelled_optional: list[str] = []
        for step in next_task.steps:
            if not step.required and step.status not in TERMINAL_STEP_STATUSES:
                step.status = "cancelled"
                step.reason = "task_completed"
                cancelled_optional.append(step.step_id)
        next_task.status = "completed"
        record.append_many(
            session_id=context.session_id,
            entries=[
                {
                    "event_type": "task_completed",
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {"result_summary": args.get("result_summary"), "cancelled_optional_step_ids": cancelled_optional},
                }
            ],
        )
        record.task = next_task
        return {"task": next_task.model_dump(mode="json")}

    def fail_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        return self._terminate_task(context, args, status="failed", event_type="task_failed", step_status="failed", reason="task_failed")

    def cancel_task(self, context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        return self._terminate_task(context, args, status="cancelled", event_type="task_cancelled", step_status="cancelled", reason="task_cancelled")

    def _terminate_task(self, context: ToolExecutionContext, args: dict[str, Any], *, status: str, event_type: str, step_status: str, reason: str) -> dict[str, Any]:
        record = self._record(context.session_id, str(args["task_id"]))
        task = record.task
        self._ensure_not_terminal(task)
        next_task = task.model_copy(deep=True)
        changed: list[str] = []
        worker_run_ids: list[str] = []
        for step in next_task.steps:
            if step.status not in TERMINAL_STEP_STATUSES:
                if step.claimed_by_run_id:
                    worker_run_ids.append(step.claimed_by_run_id)
                step.status = step_status
                step.reason = reason
                step.claimed_by_agent_id = None
                step.claimed_by_run_id = None
                step.lease_expires_at = None
                changed.append(step.step_id)
        next_task.status = status
        terminated = sorted(set(worker_run_ids))
        record.append_many(
            session_id=context.session_id,
            entries=[
                {
                    "event_type": event_type,
                    "actor_agent_id": context.agent_id,
                    "actor_run_id": context.run_id,
                    "payload": {
                        "reason": args.get("reason"),
                        f"{step_status}_step_ids": changed,
                        "terminated_worker_run_ids": terminated,
                    },
                }
            ],
        )
        record.task = next_task
        return {"task": next_task.model_dump(mode="json"), "terminated_worker_run_ids": terminated}

    def _apply_operation(self, task: Task, op: dict[str, Any]) -> None:
        kind = op.get("op")
        if kind == "update_task":
            if "title" in op:
                task.title = str(op["title"])
            if "summary" in op:
                task.summary = str(op["summary"])
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
        elif kind == "delete_step":
            step_id = str(op["step_id"])
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
        record = self._record(context.session_id, task_id)
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
                            "payload": {
                                "previous_status": previous,
                                "status": step.status,
                                "result_summary": step.result_summary,
                                "reason": step.reason,
                            },
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

    def _record(self, session_id: str, task_id: str) -> TaskRecord:
        key = (session_id, task_id)
        if key not in self._records:
            raise AgentCoreError(ErrorCode.TASK_NOT_FOUND, f"task not found: {task_id}")
        return self._records[key]

    def _replay_event(self, task: Task, event_type: str, step_id: str | None, payload: dict[str, Any]) -> None:
        if event_type == "task_updated":
            for op in payload.get("operations") or []:
                self._apply_operation(task, op)
            self._promote_ready(task)
            self._promote_running(task)
        elif event_type == "task_running":
            task.status = "running"
        elif event_type == "task_completed":
            task.status = "completed"
        elif event_type == "task_failed":
            task.status = "failed"
        elif event_type == "task_cancelled":
            task.status = "cancelled"
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


def _safe_wal_name(value: str) -> str:
    name = Path(value).name
    if name != value or not name.endswith(".wal.jsonl"):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "wal_name must be a local *.wal.jsonl filename")
    return name
