from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agent_core.agents.workers import WorkerRuntimeState
from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.workers.executors.codex_mcp import CodexMcpWorkerExecutor


class RecordingRuntime:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def _emit_child_run_event(self, **kwargs) -> None:
        self.events.append((str(kwargs["event_type"]), dict(kwargs["payload"])))

    def text(self) -> str:
        return "".join(str(payload.get("text") or "") for event_type, payload in self.events if event_type == "worker_text_delta")


def _context(*, project: Path, instruction: str, worker_id: str = "codex_mcp_worker") -> WorkerExecutorContext:
    server = Path(__file__).resolve().parents[1] / "fixtures" / "fake_codex_mcp_server.py"
    return WorkerExecutorContext(
        session_id="sess_codex_mcp",
        parent_run_id="run_parent",
        parent_agent_id="agent_orchestrator",
        task_id="task_codex_mcp",
        instruction=instruction,
        worker=WorkerRuntimeState(worker_id=worker_id, pool_id="default", agent_definition_id="codex_mcp_agent"),
        worker_agent_id="agent_worker_codex_mcp",
        worker_run_id="run_worker_codex_mcp",
        worker_start_node=None,  # type: ignore[arg-type]
        worker_stream=None,  # type: ignore[arg-type]
        executor_config={
            "command": [sys.executable, str(server)],
            "cwd": str(project),
            "sandbox": "read-only",
            "approval_policy": "on-request",
        },
    )


@pytest.mark.asyncio
async def test_codex_mcp_worker_streams_result_and_persists_thread(tmp_path: Path) -> None:
    db = HubDatabase(tmp_path / "hub.db")
    await db.open()
    events = HubEventHub()
    permissions = PermissionBridge(db, events)
    runtime = RecordingRuntime()
    executor = CodexMcpWorkerExecutor(db=db, permission_bridge=permissions, project_dir=tmp_path)
    try:
        result = await executor.run(runtime, _context(project=tmp_path, instruction="Reply exactly codex-mcp-ok"))
        mapping = await db.get_external_worker_session(
            core_session_id="sess_codex_mcp",
            worker_id="codex_mcp_worker",
            executor_type="codex_mcp",
        )
    finally:
        await permissions.shutdown()
        await db.close()

    assert result.text == "codex-mcp-ok"
    assert runtime.text() == "codex-mcp-ok"
    assert result.data == {"text": "codex-mcp-ok", "external_thread_id": "thread_fake_1"}
    assert mapping is not None
    assert mapping["external_session_id"] == "thread_fake_1"


@pytest.mark.asyncio
async def test_codex_mcp_worker_uses_codex_reply_when_thread_is_stored(tmp_path: Path) -> None:
    db = HubDatabase(tmp_path / "hub.db")
    await db.open()
    events = HubEventHub()
    permissions = PermissionBridge(db, events)
    executor = CodexMcpWorkerExecutor(db=db, permission_bridge=permissions, project_dir=tmp_path)
    await db.upsert_external_worker_session(
        core_session_id="sess_codex_mcp",
        worker_id="codex_mcp_worker",
        executor_type="codex_mcp",
        external_session_id="thread_existing",
    )
    try:
        result = await executor.run(RecordingRuntime(), _context(project=tmp_path, instruction="Reply exactly codex-mcp-ok"))
        mapping = await db.get_external_worker_session(
            core_session_id="sess_codex_mcp",
            worker_id="codex_mcp_worker",
            executor_type="codex_mcp",
        )
    finally:
        await permissions.shutdown()
        await db.close()

    assert result.text == "codex-mcp-ok"
    assert mapping is not None
    assert mapping["external_session_id"] == "thread_existing"


@pytest.mark.asyncio
async def test_codex_mcp_worker_routes_approval_to_permission_bridge(tmp_path: Path) -> None:
    db = HubDatabase(tmp_path / "hub.db")
    await db.open()
    events = HubEventHub()
    permissions = PermissionBridge(db, events)
    conversation = await db.create_conversation(core_session_id="sess_codex_mcp", title="Codex MCP")
    permissions.bind_session(core_session_id="sess_codex_mcp", conversation_id=conversation.conversation_id)
    runtime = RecordingRuntime()
    executor = CodexMcpWorkerExecutor(db=db, permission_bridge=permissions, project_dir=tmp_path)
    run_task = asyncio.create_task(executor.run(runtime, _context(project=tmp_path, instruction="approval please")))
    try:
        permission_request_id = ""
        for _ in range(100):
            row = await (
                await db.conn.execute(
                    "SELECT permission_request_id FROM permission_requests WHERE conversation_id=? AND status='pending'",
                    (conversation.conversation_id,),
                )
            ).fetchone()
            if row is not None:
                permission_request_id = str(row["permission_request_id"])
                break
            await asyncio.sleep(0.01)
        assert permission_request_id
        await permissions.decide(permission_request_id, "deny")
        result = await asyncio.wait_for(run_task, timeout=5)
        cursor = await db.conn.execute(
            "SELECT tool_name, status FROM permission_requests WHERE conversation_id=? ORDER BY created_at",
            (conversation.conversation_id,),
        )
        permission_rows = [dict(row) for row in await cursor.fetchall()]
    finally:
        if not run_task.done():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        await permissions.shutdown()
        await db.close()

    assert result.text == "approval decision: denied"
    assert any(event_type == "codex_exec_approval_request" for event_type, _ in runtime.events)
    assert permission_rows
    assert permission_rows[0]["tool_name"] == "codex.exec"
    assert permission_rows[0]["status"] == "denied"
