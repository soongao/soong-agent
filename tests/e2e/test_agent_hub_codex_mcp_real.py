from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from agent_core.agents.workers import WorkerRuntimeState
from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.workers.executors.codex_mcp import CodexMcpWorkerExecutor


pytestmark = pytest.mark.skipif(
    os.environ.get("SOONG_AGENT_REQUIRE_CODEX_MCP_E2E") != "1",
    reason="set SOONG_AGENT_REQUIRE_CODEX_MCP_E2E=1 to run real local codex MCP e2e",
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def _emit_child_run_event(self, **kwargs) -> None:
        self.events.append((str(kwargs["event_type"]), dict(kwargs["payload"])))


@pytest.mark.asyncio
async def test_agent_hub_real_codex_mcp_worker_smoke(tmp_path) -> None:
    if shutil.which("codex") is None:
        pytest.skip("codex command is not available")
    project = Path(__file__).resolve().parents[2]
    db = HubDatabase(tmp_path / "hub.db")
    await db.open()
    events = HubEventHub()
    permissions = PermissionBridge(db, events)
    runtime = RecordingRuntime()
    executor = CodexMcpWorkerExecutor(db=db, permission_bridge=permissions, project_dir=project)
    context = WorkerExecutorContext(
        session_id="sess_codex_mcp_smoke",
        parent_run_id="run_parent",
        parent_agent_id="agent_orchestrator",
        task_id="task_codex_mcp_smoke",
        instruction="Reply with exactly: codex-mcp-smoke-ok",
        worker=WorkerRuntimeState(worker_id="codex_mcp_worker", pool_id="default", agent_definition_id="codex_mcp_agent"),
        worker_agent_id="agent_worker_codex_mcp",
        worker_run_id="run_worker_codex_mcp_smoke",
        worker_start_node=None,  # type: ignore[arg-type]
        worker_stream=None,  # type: ignore[arg-type]
        executor_config={
            "sandbox": "read-only",
            "approval_policy": "never",
            "config": {
                "notify": [],
                "check_for_update_on_startup": False,
                "suppress_unstable_features_warning": True,
                "skip_git_repo_check": True,
            },
        },
    )
    try:
        result = await asyncio.wait_for(executor.run(runtime, context), timeout=45)
        mapping = await db.get_external_worker_session(
            core_session_id="sess_codex_mcp_smoke",
            worker_id="codex_mcp_worker",
            executor_type="codex_mcp",
        )
    finally:
        await permissions.shutdown()
        await db.close()

    assert "codex-mcp-smoke-ok" in result.text
    assert mapping is not None
    assert mapping["external_session_id"]
    assert any(event_type == "worker_text_delta" for event_type, _ in runtime.events)
