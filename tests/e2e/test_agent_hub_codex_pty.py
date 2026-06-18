from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from agent_core.agents.workers import WorkerRuntimeState
from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext
from agent_hub.backend.workers.executors.codex_pty import CodexPtyWorkerExecutor
from agent_hub.backend.workers.pty import PtySessionManager


pytestmark = pytest.mark.skipif(
    os.environ.get("SOONG_AGENT_REQUIRE_CODEX_PTY_E2E") != "1",
    reason="set SOONG_AGENT_REQUIRE_CODEX_PTY_E2E=1 to run real local codex PTY e2e",
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def _emit_child_run_event(self, **kwargs) -> None:
        self.events.append((str(kwargs["event_type"]), dict(kwargs["payload"])))

    def text(self) -> str:
        return "".join(str(payload.get("text") or "") for event_type, payload in self.events if event_type == "worker_text_delta")


@pytest.mark.asyncio
async def test_agent_hub_real_codex_pty_worker_smoke(tmp_path) -> None:
    if shutil.which("codex") is None:
        pytest.skip("codex command is not available")
    project = Path(__file__).resolve().parents[2]

    manager = PtySessionManager()
    runtime = RecordingRuntime()
    executor = CodexPtyWorkerExecutor(pty_manager=manager, project_dir=project)
    context = WorkerExecutorContext(
        session_id="sess_codex_pty_smoke",
        parent_run_id="run_parent",
        parent_agent_id="agent_orchestrator",
        task_id="task_codex_pty_smoke",
        instruction="Reply with exactly: codex-pty-smoke-ok",
        worker=WorkerRuntimeState(worker_id="codex_pty_worker", pool_id="default", agent_definition_id="codex_pty_agent"),
        worker_agent_id="agent_worker_codex_pty",
        worker_run_id="run_worker_codex_pty_smoke",
        worker_start_node=None,  # type: ignore[arg-type]
        worker_stream=None,  # type: ignore[arg-type]
        executor_config={
            "sandbox": "read-only",
            "ask_for_approval": "never",
            "args": [
                "--config",
                "notify=[]",
                "--config",
                "check_for_update_on_startup=false",
                "--config",
                "suppress_unstable_features_warning=true",
            ],
        },
    )
    try:
        try:
            result = await asyncio.wait_for(executor.run(runtime, context), timeout=45)
        except asyncio.TimeoutError:
            print("\n--- codex pty captured output ---")
            print(runtime.text())
            print("--- end codex pty captured output ---")
            raise
    finally:
        await manager.close()

    assert "codex-pty-smoke-ok" in result.text
    assert "<<<AGENTHUB_DONE:" not in result.text
    assert any(event_type == "worker_text_delta" for event_type, _ in runtime.events)
