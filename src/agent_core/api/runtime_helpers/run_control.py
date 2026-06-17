from __future__ import annotations

import asyncio
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.errors.codes import ErrorCode
from agent_core.events import make_event
from agent_core.types import CancelResult, RunStatus, UserMessage


async def cancel_run(runtime: Any, handle: RunHandle) -> CancelResult:
    if handle._queued:
        try:
            runtime._session_queues[handle.session_id].remove(handle)
        except ValueError:
            pass
        handle.status = RunStatus.CANCELLED
        handle._queued = False
        if runtime.store is not None:
            await runtime.store.update_run(
                run_id=handle.run_id,
                status=RunStatus.CANCELLED.value,
                end_reason="aborted_tools",
                error={"code": ErrorCode.CANCELLED.value, "message": "queued run cancelled", "reason": "cancelled"},
            )
        await runtime._emit(handle, "run_dequeued", payload={"cancelled": True})
        await runtime._emit(handle, "run_cancelled", payload={"queued": True})
        await handle._stream.close()
        return CancelResult(run_id=handle.run_id, status=handle.status, cancelled=True)
    if handle._task is not None and not handle._task.done():
        handle._task.cancel()
        try:
            await asyncio.wait_for(handle._task, timeout=(runtime.config.runtime.cancel_timeout_ms if runtime.config else 10000) / 1000)
        except asyncio.TimeoutError:
            await runtime._emit(handle, "cancel_timeout", level="warning")
    return CancelResult(run_id=handle.run_id, status=handle.status, cancelled=handle.status == RunStatus.CANCELLED)


async def cancel_worker_runs(
    runtime: Any,
    *,
    session_id: str,
    task_id: str,
    worker_run_ids: list[str] | None = None,
    reason: str = "task_terminated",
) -> dict[str, Any]:
    await runtime._ensure_started()
    assert runtime.store
    requested = set(worker_run_ids or [])
    for run_id, meta in list(runtime._worker_run_meta.items()):
        if meta.get("session_id") == session_id and meta.get("task_id") == task_id:
            requested.add(run_id)
    current_task = asyncio.current_task()
    cancelled: list[str] = []
    missing: list[str] = []
    for run_id in sorted(requested):
        task = runtime._worker_run_tasks.get(run_id)
        meta = runtime._worker_run_meta.get(run_id)
        if task is None or task.done():
            missing.append(run_id)
            continue
        await runtime.store.add_event(
            make_event(
                session_id=session_id,
                agent_id=meta.get("worker_agent_id") if meta else None,
                run_id=run_id,
                event_type="worker_run_cancel_requested",
                payload={
                    "task_id": task_id,
                    "reason": reason,
                    "worker_id": meta.get("worker_id") if meta else None,
                    "worker_agent_id": meta.get("worker_agent_id") if meta else None,
                },
            )
        )
        if task is current_task:
            continue
        task.cancel()
        cancelled.append(run_id)
    timeout = (runtime.config.runtime.cancel_timeout_ms if runtime.config else 10000) / 1000
    for run_id in cancelled:
        task = runtime._worker_run_tasks.get(run_id)
        if task is None:
            continue
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            await runtime.store.add_event(
                make_event(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="child_agent_cancel_timeout",
                    level="warning",
                    payload={"task_id": task_id, "reason": reason},
                )
            )
        except Exception:
            pass
    return {"cancelled_worker_run_ids": cancelled, "missing_worker_run_ids": missing}


async def start_next_queued(runtime: Any, session_id: str) -> None:
    queue = runtime._session_queues.get(session_id)
    if not queue:
        return
    next_handle = queue.popleft()
    next_handle._queued = False
    runtime._session_active[session_id] = next_handle
    await runtime._emit(next_handle, "run_dequeued")
    next_handle._task = asyncio.create_task(runtime._run(next_handle, next_handle._message or UserMessage.from_text("")))
