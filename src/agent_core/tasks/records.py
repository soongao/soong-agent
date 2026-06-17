from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.storage.task_wal import TaskWalWriter
from agent_core.tasks.models import Task
from agent_core.types.common import utc_iso


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
