from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from agent_core import AgentRuntime
from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.agents.workers import WorkerPoolRuntime
from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.permissions import PermissionSessionCache
from agent_core.providers import ModelEvent, ModelRequest, ProviderAdapter, ProviderRegistry
from agent_core.providers.base import StopReason
from agent_core.types.content import TextBlock
from agent_core.types.runtime import RunStatus
from agent_core.tasks.service import TaskService
from agent_core.tasks.tools import register_task_tools
from agent_core.tools.agent_tools import register_agent_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.tools import ToolCall
from tests.conftest import write_config


async def make_context(home, project):
    write_config(home, worker_pool=True)
    config, paths = load_runtime_config(project_dir=project)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    service = TaskService()
    registry = ToolRegistry()
    register_task_tools(registry, service)
    register_agent_tools(registry, AgentDefinitionRegistry(), WorkerPoolRuntime(config.agents))
    context = ToolExecutionContext(
        session_id="sess_worker",
        run_id="run_worker",
        agent_id="agent_orchestrator",
        agent_role="orchestrator",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
        services={"task_service": service},
    )
    return registry, context


@pytest.mark.asyncio
async def test_list_workers_and_dispatch_claims_ready_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry, context = await make_context(home, project)
    workers = await registry.execute(ToolCall(tool_call_id="w1", name="agent.list_workers", arguments={}), context)
    assert workers.content[0].data["workers"][0]["worker_id"] == "worker1"  # type: ignore[union-attr]

    await registry.execute(
        ToolCall(
            tool_call_id="t1",
            name="agent.task_create",
            arguments={
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}, {"step_id": "s2", "title": "Step 2"}],
            },
        ),
        context,
    )
    result = await registry.execute(
        ToolCall(
            tool_call_id="d1",
            name="agent.dispatch_worker",
            arguments={"task_id": "task1", "instruction": "do it", "allowed_step_ids": ["s2"]},
        ),
        context,
    )
    assert not result.is_error
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["claimed_step_id"] == "s2"
    assert data["no_step_claimed"] is False


@pytest.mark.asyncio
async def test_dispatch_empty_allowed_step_ids_rejected(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry, context = await make_context(home, project)
    result = await registry.execute(
        ToolCall(
            tool_call_id="d1",
            name="agent.dispatch_worker",
            arguments={"task_id": "task1", "instruction": "do it", "allowed_step_ids": []},
        ),
        context,
    )
    assert result.is_error
    assert result.error and result.error.code.value == "validation_error"


@pytest.mark.asyncio
async def test_dispatch_no_step_claimed(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry, context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="t1",
            name="agent.task_create",
            arguments={
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [
                    {"step_id": "s1", "title": "Step 1"},
                    {"step_id": "s2", "title": "Step 2", "depends_on_step_ids": ["s1"]},
                ],
            },
        ),
        context,
    )
    result = await registry.execute(
        ToolCall(
            tool_call_id="d1",
            name="agent.dispatch_worker",
            arguments={"task_id": "task1", "instruction": "do it", "allowed_step_ids": ["s2"]},
        ),
        context,
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["no_step_claimed"] is True
    assert data["claimed_step_id"] is None


@pytest.mark.asyncio
async def test_runtime_worker_preflight_skips_run_when_no_ready_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = OrchestratorWorkerProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        assert runtime.config and runtime.artifacts
        context = ToolExecutionContext(
            session_id="sess_preflight",
            run_id="run_parent",
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=project,
            home_dir=home,
            config=runtime.config,
            artifact_manager=runtime.artifacts,
            permission_callback=allow,
            permission_cache=PermissionSessionCache(),
        )
        runtime.task_service.create_task(
            context,
            {
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [
                    {"step_id": "s1", "title": "Step 1"},
                    {"step_id": "s2", "title": "Step 2", "depends_on_step_ids": ["s1"]},
                ],
            },
        )
        result = await runtime.run_worker_agent(
            session_id="sess_preflight",
            parent_run_id="run_parent",
            parent_agent_id="agent_orchestrator",
            task_id="task1",
            instruction="do it",
            allowed_step_ids=["s2"],
        )
        replay = await runtime.replay_session("sess_preflight")

    assert result["no_step_claimed"] is True
    assert result["child_run_id"] is None
    assert provider.requests == []
    assert all(event.event_type != "worker_run_started" for event in replay.events)


class OrchestratorWorkerProvider(ProviderAdapter):
    def __init__(self, *, worker_claim_step_id: str = "s1", dispatch_allowed_tools=None, worker_complete_step: bool = True) -> None:
        self.requests: list[ModelRequest] = []
        self.worker_claim_step_id = worker_claim_step_id
        self.dispatch_allowed_tools = dispatch_allowed_tools
        self.worker_complete_step = worker_complete_step

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        run_id = str(request.metadata.get("run_id") or "")
        if run_id.startswith("run_worker"):
            worker_turn = sum(1 for item in self.requests if str(item.metadata.get("run_id") or "") == run_id)
            if worker_turn == 1:
                yield ModelEvent(
                    event_type="model_completed",
                    tool_calls=[
                        ToolCall(
                            tool_call_id="w_claim",
                            name="agent.task_claim_step",
                            arguments={"task_id": "task1", "step_id": self.worker_claim_step_id},
                        )
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
                return
            if worker_turn == 2 and self.worker_complete_step:
                yield ModelEvent(
                    event_type="model_completed",
                    tool_calls=[
                        ToolCall(
                            tool_call_id="w_done",
                            name="agent.task_update_step",
                            arguments={"task_id": "task1", "step_id": "s1", "status": "completed", "result_summary": "worker done"},
                        )
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
                return
            if worker_turn == 2:
                yield ModelEvent(event_type="model_completed", content=[TextBlock(text="worker final without terminal")], stop_reason=StopReason.END_TURN)
                return
            yield ModelEvent(event_type="model_completed", content=[TextBlock(text="worker final")], stop_reason=StopReason.END_TURN)
            return

        main_turn = sum(1 for item in self.requests if not str(item.metadata.get("run_id") or "").startswith("run_worker"))
        if main_turn == 1:
            yield ModelEvent(
                event_type="model_completed",
                tool_calls=[
                    ToolCall(
                        tool_call_id="create",
                        name="agent.task_create",
                        arguments={
                            "task_id": "task1",
                            "wal_name": "task1.wal.jsonl",
                            "title": "Task",
                            "summary": "",
                            "steps": [{"step_id": "s1", "title": "Step 1"}, {"step_id": "s2", "title": "Step 2"}],
                        },
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
            )
            return
        if main_turn == 2:
            yield ModelEvent(
                event_type="model_completed",
                tool_calls=[
                    ToolCall(
                        tool_call_id="dispatch",
                        name="agent.dispatch_worker",
                        arguments={
                            "task_id": "task1",
                            "instruction": "do it",
                            "allowed_step_ids": ["s1"],
                            **({"allowed_tools": self.dispatch_allowed_tools} if self.dispatch_allowed_tools is not None else {}),
                        },
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
            )
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="orchestrator final")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class BlockingWorkerProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.worker_run_id: str | None = None
        self.worker_blocked = asyncio.Event()
        self.release_worker = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        run_id = str(request.metadata.get("run_id") or "")
        if run_id.startswith("run_worker"):
            self.worker_run_id = run_id
            worker_turn = sum(1 for item in self.requests if str(item.metadata.get("run_id") or "") == run_id)
            if worker_turn == 1:
                yield ModelEvent(
                    event_type="model_completed",
                    tool_calls=[
                        ToolCall(
                            tool_call_id="w_claim",
                            name="agent.task_claim_step",
                            arguments={"task_id": "task1", "step_id": "s1"},
                        )
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
                return
            self.worker_blocked.set()
            await self.release_worker.wait()
            yield ModelEvent(event_type="model_completed", content=[TextBlock(text="late worker final")], stop_reason=StopReason.END_TURN)
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="unused")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        self.release_worker.set()


@pytest.mark.asyncio
async def test_runtime_dispatch_worker_runs_worker_agent_and_updates_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = OrchestratorWorkerProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("orchestrate", session_id="sess_runtime_worker", mode="orchestrator")
        events = [event.event_type async for event in handle.events()]
        replay = await runtime.replay_session("sess_runtime_worker")

    assert handle.status == RunStatus.COMPLETED
    replay_events = [event.event_type for event in replay.events]
    assert "worker_run_started" in replay_events
    assert "worker_run_completed" in replay_events
    assert "tool_completed" in events
    worker_requests = [request for request in provider.requests if str(request.metadata.get("run_id") or "").startswith("run_worker")]
    assert worker_requests
    assert {tool.name for tool in worker_requests[0].tools} == {
        "agent.task_get",
        "agent.task_query_steps",
        "agent.task_claim_step",
        "agent.task_update_step",
        "code.read_file",
    }
    assert any(node.node_type == "worker_dispatch" for node in replay.nodes)
    task_get = runtime.task_service.get_task(
        ToolExecutionContext(
            session_id="sess_runtime_worker",
            run_id="inspect",
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=project,
            home_dir=home,
            config=runtime.config,
            artifact_manager=ArtifactManager(home_dir=home),
        ),
        {"task_id": "task1", "include_terminal_steps": True},
    )
    steps = {step["step_id"]: step for step in task_get["task"]["steps"]}
    assert steps["s1"]["status"] == "completed"
    assert steps["s1"]["result_summary"] == "worker done"


@pytest.mark.asyncio
async def test_runtime_worker_cannot_claim_step_outside_dispatch_scope(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = OrchestratorWorkerProvider(worker_claim_step_id="s2")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("orchestrate", session_id="sess_worker_scope", mode="orchestrator")
        _events = [event.event_type async for event in handle.events()]
        task_get = runtime.task_service.get_task(
            ToolExecutionContext(
                session_id="sess_worker_scope",
                run_id="inspect",
                agent_id="agent_orchestrator",
                agent_role="orchestrator",
                project_dir=project,
                home_dir=home,
                config=runtime.config,
                artifact_manager=ArtifactManager(home_dir=home),
            ),
            {"task_id": "task1", "include_terminal_steps": True},
        )

    steps = {step["step_id"]: step for step in task_get["task"]["steps"]}
    assert steps["s1"]["status"] == "ready"
    assert steps["s2"]["status"] == "ready"


@pytest.mark.asyncio
async def test_dispatch_allowed_tools_cannot_expand_worker_effective_set(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = OrchestratorWorkerProvider(dispatch_allowed_tools=["code.write_file"])
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("orchestrate", session_id="sess_worker_bad_tools", mode="orchestrator")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "tool_failed" in events
    worker_requests = [request for request in provider.requests if str(request.metadata.get("run_id") or "").startswith("run_worker")]
    assert worker_requests == []


@pytest.mark.asyncio
async def test_worker_exit_with_unclosed_claim_marks_step_failed(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = OrchestratorWorkerProvider(worker_complete_step=False)
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("orchestrate", session_id="sess_worker_unclosed", mode="orchestrator")
        _events = [event.event_type async for event in handle.events()]
        task_get = runtime.task_service.get_task(
            ToolExecutionContext(
                session_id="sess_worker_unclosed",
                run_id="inspect",
                agent_id="agent_orchestrator",
                agent_role="orchestrator",
                project_dir=project,
                home_dir=home,
                config=runtime.config,
                artifact_manager=ArtifactManager(home_dir=home),
            ),
            {"task_id": "task1", "include_terminal_steps": True},
        )

    step = {step["step_id"]: step for step in task_get["task"]["steps"]}["s1"]
    assert step["status"] == "failed"
    assert step["reason"] == "worker_finished_without_terminal_step_status"
    wal_text = (project / ".soong-agent" / "tasks" / "sess_worker_unclosed" / "task1.wal.jsonl").read_text(encoding="utf-8")
    assert "worker_finished_without_terminal_step_status" in wal_text


@pytest.mark.asyncio
async def test_task_cancel_cancels_active_worker_run_and_preserves_task_terminal_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    provider = BlockingWorkerProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        assert runtime.store and runtime.config and runtime.artifacts
        session_id = "sess_worker_task_cancel"
        parent_run_id = "run_parent"
        await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id="agent_orchestrator")
        await runtime.store.ensure_agent(agent_id="agent_orchestrator", session_id=session_id, agent_type="orchestrator", status="running")
        await runtime.store.create_run(run_id=parent_run_id, session_id=session_id, agent_id="agent_orchestrator", status="running")
        context = ToolExecutionContext(
            session_id=session_id,
            run_id=parent_run_id,
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=project,
            home_dir=home,
            config=runtime.config,
            artifact_manager=runtime.artifacts,
            permission_callback=allow,
            permission_cache=PermissionSessionCache(),
            services={
                "task_service": runtime.task_service,
                "runtime": runtime,
                "agent_definitions": runtime.agent_definitions,
                "context_state": runtime.context_state,
            },
        )
        create = await runtime.tool_registry.execute(
            ToolCall(
                tool_call_id="create",
                name="agent.task_create",
                arguments={
                    "task_id": "task1",
                    "wal_name": "task1.wal.jsonl",
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}],
                },
            ),
            context,
        )
        assert not create.is_error
        worker_task = asyncio.create_task(
            runtime.run_worker_agent(
                session_id=session_id,
                parent_run_id=parent_run_id,
                parent_agent_id="agent_orchestrator",
                task_id="task1",
                instruction="claim and then wait",
                allowed_step_ids=["s1"],
            )
        )
        await asyncio.wait_for(provider.worker_blocked.wait(), timeout=1)
        assert provider.worker_run_id is not None

        cancel = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="cancel", name="agent.task_cancel", arguments={"task_id": "task1", "reason": "stop now"}),
            context,
        )
        assert not cancel.is_error
        cancel_data = cancel.content[0].data  # type: ignore[union-attr]
        assert cancel_data["terminated_worker_run_ids"] == [provider.worker_run_id]
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        task_get = runtime.task_service.get_task(context, {"task_id": "task1", "include_terminal_steps": True})
        step = task_get["task"]["steps"][0]
        assert task_get["task"]["status"] == "cancelled"
        assert step["status"] == "cancelled"
        assert step["reason"] == "task_cancelled"
        replay = await runtime.replay_session(session_id)

    replay_events = [event.event_type for event in replay.events]
    assert "worker_run_cancel_requested" in replay_events
    assert "worker_run_cancelled" in replay_events
