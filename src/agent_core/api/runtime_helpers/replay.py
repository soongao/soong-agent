from __future__ import annotations

from typing import Any

from agent_core.api.runtime_helpers.views import _model_request_views_from_events, _redact_replay_payload
from agent_core.types import Node, ReplayResult


async def replay_session(
    runtime: Any,
    session_id: str,
    *,
    from_seq: int | None = None,
    to_seq: int | None = None,
    include_sensitive: bool = False,
) -> ReplayResult:
    await runtime._ensure_started()
    assert runtime.store
    nodes, events = await runtime.store.replay_session(session_id, from_seq=from_seq, to_seq=to_seq)
    artifacts = await runtime.store.list_artifacts(session_id=session_id)
    if not include_sensitive:
        nodes, events, artifacts = _redact_replay_payload(nodes, events, artifacts)
    model_requests = _model_request_views_from_events(events, run_id=None)
    return ReplayResult(
        session_id=session_id,
        nodes=nodes,
        events=events,
        artifacts=artifacts,
        model_requests=model_requests,
        task_wal_errors=runtime.task_service.unavailable_task_summaries(session_id),
    )


async def replay_run(runtime: Any, run_id: str, *, include_sensitive: bool = False) -> ReplayResult:
    await runtime._ensure_started()
    assert runtime.store
    session_id = await runtime.store.find_run_session(run_id)
    if session_id is None:
        return ReplayResult(session_id="", run_id=run_id)
    nodes, events = await runtime.store.replay_run(session_id, run_id)
    artifacts = [
        artifact
        for artifact in await runtime.store.list_artifacts(session_id=session_id)
        if artifact.get("run_id") == run_id
    ]
    if not include_sensitive:
        nodes, events, artifacts = _redact_replay_payload(nodes, events, artifacts)
    model_requests = _model_request_views_from_events(events, run_id=run_id)
    return ReplayResult(
        session_id=session_id,
        run_id=run_id,
        nodes=nodes,
        events=events,
        artifacts=artifacts,
        model_requests=model_requests,
        task_wal_errors=runtime.task_service.unavailable_task_summaries(session_id),
    )


async def get_node_path(runtime: Any, node_id: str) -> list[Node]:
    await runtime._ensure_started()
    assert runtime.store
    return await runtime.store.get_node_path(node_id)
