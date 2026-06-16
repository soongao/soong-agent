from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

from agent_core.types.common import StrictModel

_writer_locks_guard = threading.Lock()
_writer_locks: dict[Path, threading.Lock] = {}


class TaskWalEvent(StrictModel):
    wal_seq: int
    session_id: str
    event_id: str
    event_type: str
    actor_agent_id: str
    actor_run_id: str
    task_id: str
    step_id: str | None = None
    payload: dict[str, Any]
    created_at: str


class TaskWalWriter:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock_for_path(self.path)

    def append(self, payload: dict[str, Any]) -> None:
        self.append_many([payload])

    def append_many(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        events = [TaskWalEvent.model_validate(payload) for payload in payloads]
        serialized = "".join(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n" for event in events)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(serialized)


def _lock_for_path(path: Path) -> threading.Lock:
    with _writer_locks_guard:
        lock = _writer_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _writer_locks[path] = lock
        return lock
