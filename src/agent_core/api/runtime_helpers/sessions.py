from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent_core.api.runtime_helpers.artifacts import _delete_artifact_path
from agent_core.api.runtime_helpers.views import _node_content_preview, _session_info_from_row
from agent_core.errors.codes import ErrorCode
from agent_core.events import make_event
from agent_core.storage import new_id
from agent_core.types import (
    DeleteSessionResult,
    ErrorPayload,
    ForkSessionResult,
    RunMode,
    SessionInfo,
    SessionNodeInfo,
    SwitchNodeResult,
)


async def list_sessions(runtime: Any, *, limit: int = 20, offset: int = 0) -> list[SessionInfo]:
    await runtime._ensure_started()
    assert runtime.store
    rows = await runtime.store.list_sessions(limit=max(limit, 1), offset=max(offset, 0))
    return [_session_info_from_row(row) for row in rows]


async def list_session_nodes(runtime: Any, session_id: str, *, limit: int = 20, offset: int = 0) -> list[SessionNodeInfo]:
    await runtime._ensure_started()
    assert runtime.store
    session = await runtime.store.get_session(session_id)
    if session is None:
        return []
    active_node_id = session.get("active_node_id")
    nodes = await runtime.store.list_session_nodes(session_id, limit=max(limit, 1), offset=max(offset, 0))
    return [
        SessionNodeInfo(
            node_id=node.node_id,
            parent_id=node.parent_id,
            role=node.role,
            node_type=node.node_type,
            content_preview=_node_content_preview(node),
            created_at=node.created_at,
            active=node.node_id == active_node_id,
        )
        for node in nodes
    ]


async def fork_session(
    runtime: Any,
    source_session_id: str,
    *,
    node_id: str | None = None,
    new_session_id: str | None = None,
    mode: Literal["normal", "orchestrator"] = "normal",
) -> ForkSessionResult:
    await runtime._ensure_started()
    assert runtime.store and runtime.paths
    if source_session_id in runtime._session_active or runtime._session_queues.get(source_session_id):
        return ForkSessionResult(
            source_session_id=source_session_id,
            forked=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="cannot fork a session with active or queued runs"),
        )
    if await runtime.store.has_active_runs(source_session_id):
        return ForkSessionResult(
            source_session_id=source_session_id,
            forked=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="cannot fork a session with active or queued runs"),
        )
    source_session = await runtime.store.get_session(source_session_id)
    if source_session is None:
        return ForkSessionResult(
            source_session_id=source_session_id,
            forked=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="source session not found"),
        )
    source_node_id = node_id or source_session.get("active_node_id")
    if not source_node_id:
        return ForkSessionResult(
            source_session_id=source_session_id,
            forked=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="source session has no active node"),
        )
    path = await runtime.store.get_session_node_path(source_session_id, str(source_node_id))
    if not path:
        return ForkSessionResult(
            source_session_id=source_session_id,
            source_node_id=str(source_node_id),
            forked=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="node not found in source session"),
        )
    run_mode = RunMode(mode)
    target_session_id = new_session_id or new_id("sess")
    root_agent_id = runtime._root_agent_id(session_id=target_session_id, mode=run_mode)
    try:
        result = await runtime.store.fork_session_from_path(
            source_session_id=source_session_id,
            new_session_id=target_session_id,
            cwd=str(source_session.get("cwd") or runtime.paths.project_dir),
            root_agent_id=root_agent_id,
            agent_type="orchestrator" if run_mode == RunMode.ORCHESTRATOR else "main",
            nodes=path,
            metadata={
                "forked_from_session_id": source_session_id,
                "forked_from_node_id": str(source_node_id),
            },
        )
    except ValueError as exc:
        return ForkSessionResult(
            source_session_id=source_session_id,
            source_node_id=str(source_node_id),
            session_id=target_session_id,
            forked=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message=str(exc)),
        )
    await runtime.store.add_event(
        make_event(
            session_id=target_session_id,
            event_type="session_forked",
            node_id=result.get("active_node_id"),
            payload={
                "source_session_id": source_session_id,
                "source_node_id": str(source_node_id),
                "copied_nodes": result.get("copied_nodes", 0),
            },
        )
    )
    return ForkSessionResult(
        source_session_id=source_session_id,
        source_node_id=str(source_node_id),
        session_id=target_session_id,
        active_node_id=result.get("active_node_id"),
        forked=True,
        copied_nodes=int(result.get("copied_nodes") or 0),
    )


async def delete_session(runtime: Any, session_id: str) -> DeleteSessionResult:
    await runtime._ensure_started()
    assert runtime.store
    if session_id in runtime._session_active or runtime._session_queues.get(session_id):
        return DeleteSessionResult(
            session_id=session_id,
            deleted=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="session has active or queued runs"),
        )
    if await runtime.store.has_active_runs(session_id):
        return DeleteSessionResult(
            session_id=session_id,
            deleted=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="session has active or queued runs"),
        )
    artifacts = await runtime.store.list_artifacts(session_id=session_id)
    deletion_errors: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "")
        path = Path(str(artifact.get("path") or ""))
        try:
            _delete_artifact_path(path)
        except OSError as exc:
            deletion_errors.append({"artifact_id": artifact_id, "path": str(path), "error": str(exc)})
    if deletion_errors:
        return DeleteSessionResult(
            session_id=session_id,
            deleted=False,
            error=ErrorPayload(
                code=ErrorCode.STORAGE_ERROR,
                message=f"failed to delete session artifacts: {session_id}",
                details={"artifacts": deletion_errors},
            ),
        )
    await runtime.store.delete_session(session_id)
    return DeleteSessionResult(session_id=session_id, deleted=True)


async def switch_node(runtime: Any, session_id: str, node_id: str) -> SwitchNodeResult:
    await runtime._ensure_started()
    assert runtime.store
    if session_id in runtime._session_active or runtime._session_queues.get(session_id):
        return SwitchNodeResult(
            session_id=session_id,
            node_id=node_id,
            switched=False,
            error=ErrorPayload(code=ErrorCode.SESSION_ACTIVE, message="cannot switch active node while session has active or queued runs"),
        )
    if not await runtime.store.node_exists(session_id, node_id):
        return SwitchNodeResult(
            session_id=session_id,
            node_id=node_id,
            switched=False,
            error=ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message="node not found in session"),
        )
    await runtime.store.set_active_node(session_id, node_id)
    await runtime.store.add_event(
        make_event(
            session_id=session_id,
            event_type="active_node_switched",
            node_id=node_id,
            payload={"node_id": node_id},
        )
    )
    return SwitchNodeResult(session_id=session_id, node_id=node_id, switched=True)
