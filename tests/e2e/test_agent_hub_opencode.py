from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from agent_core import AgentRuntime
from agent_core.tools.execution import ToolExecutionContext
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.events import HubEventHub
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.workers.executors.opencode import OpenCodeWorkerExecutor
from tests.conftest import write_config


pytestmark = pytest.mark.skipif(
    os.environ.get("SOONG_AGENT_REQUIRE_OPENCODE_E2E") != "1",
    reason="set SOONG_AGENT_REQUIRE_OPENCODE_E2E=1 to run real local opencode ACP e2e",
)


@pytest.mark.asyncio
async def test_agent_hub_real_opencode_worker_executor_smoke(tmp_path) -> None:
    if shutil.which("opencode") is None:
        pytest.skip("opencode command is not available")
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (project / "README.md").write_text("opencode smoke project\n", encoding="utf-8")
    write_config(home, provider="ollama", model_name="gemma4")

    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    permissions = PermissionBridge(db, events)
    runtime = AgentRuntime(project_dir=project, home_dir=home, permission_callback=permissions.permission_callback)
    executor = OpenCodeWorkerExecutor(db=db, permission_bridge=permissions, project_dir=project)
    runtime.register_worker_executor("opencode", executor)
    try:
        async with runtime:
            worker = await runtime.create_worker_config(
                {
                    "worker_id": "opencode_worker",
                    "name": "OpenCode Worker",
                    "system_prompt": "External OpenCode worker.",
                    "allowed_tools": ["opencode.acp"],
                    "metadata": {"worker_executor": {"type": "opencode", "config": {}}},
                }
            )
            session_id = "sess_opencode_smoke"
            root_agent_id = runtime._root_agent_id(session_id=session_id, mode="orchestrator")
            assert runtime.store and runtime.config and runtime.artifacts
            await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id=root_agent_id)
            conversation = await db.create_conversation(core_session_id=session_id, title="OpenCode smoke")
            permissions.bind_session(core_session_id=session_id, conversation_id=conversation.conversation_id)
            context = ToolExecutionContext(
                session_id=session_id,
                run_id="run_parent",
                agent_id=root_agent_id,
                agent_role="orchestrator",
                project_dir=project,
                home_dir=home,
                config=runtime.config,
                artifact_manager=runtime.artifacts,
                permission_callback=permissions.permission_callback,
                permission_cache=runtime._permission_caches[session_id],
            )
            runtime.task_service.create_task(
                context,
                {
                    "task_id": "task_smoke",
                    "wal_name": "task_smoke.wal.jsonl",
                    "title": "Smoke",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Answer briefly"}],
                },
            )
            result = await asyncio.wait_for(
                runtime.run_worker_agent(
                    session_id=session_id,
                    parent_run_id="run_parent",
                    parent_agent_id=root_agent_id,
                    task_id="task_smoke",
                    instruction="Reply with exactly: opencode-smoke-ok",
                    worker_agent_id=runtime._worker_agent_id(session_id=session_id, worker_id=worker.worker_id),
                    timeout_ms=60000,
                ),
                timeout=75,
            )
            mapping = await db.get_external_worker_session(
                core_session_id=session_id,
                worker_id="opencode_worker",
                executor_type="opencode",
            )
    finally:
        await permissions.shutdown()
        await db.close()

    assert result["worker_id"] == "opencode_worker"
    assert result["claimed_step_id"] == "s1"
    assert result["step_status"] == "completed"
    assert "opencode-smoke-ok" in result["worker_result"]["text"]
    assert mapping is not None
    assert str(mapping["external_session_id"]).startswith("ses_")

