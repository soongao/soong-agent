from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import time

from pydantic import ValidationError
import pytest

from agent_core.errors.codes import ErrorCode
from agent_core.artifacts import ArtifactManager
from agent_core import AgentRuntime
from agent_core.config import load_runtime_config
from agent_core.permissions import PermissionSessionCache
from agent_core.storage.task_wal import TaskWalWriter
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.tasks.service import TaskService
from agent_core.tasks.tools import register_task_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


async def make_context(home, project) -> ToolExecutionContext:
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    return ToolExecutionContext(
        session_id="sess_task",
        run_id="run_task",
        agent_id="agent_orchestrator",
        agent_role="orchestrator",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
    )


@pytest.mark.asyncio
async def test_plan_template_returns_instruction(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    register_task_tools(registry, TaskService())
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="agent.plan_template", arguments={"goal": "build"}),
        await make_context(home, project),
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["node_type"] == "plan_instruction"
    assert "Default Plan Template" in data["content"]
    assert "<project>/.soong-agent/plans" in data["content"]
    assert data["suggested_dir"] == str((project / ".soong-agent" / "plans").resolve())
    assert data["template_id"] == "template.plan.default"
    assert data["template_version"] == "1"


@pytest.mark.asyncio
async def test_plan_template_uses_configured_default_dir(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config_path = home / "config.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n[plan]\ndefault_dir = \"<project>/.soong-agent/plans/custom\"\n", encoding="utf-8")
    config, paths = load_runtime_config(project_dir=project)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess_task",
        run_id="run_task",
        agent_id="agent_orchestrator",
        agent_role="orchestrator",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
    )
    registry = ToolRegistry()
    register_task_tools(registry, TaskService())
    result = await registry.execute(
        ToolCall(tool_call_id="call_cfg", name="agent.plan_template", arguments={"goal": "build"}),
        context,
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["suggested_dir"] == str((project / ".soong-agent" / "plans" / "custom").resolve())


@pytest.mark.asyncio
async def test_runtime_creates_only_project_plan_and_task_dirs(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("ready")

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        handle = await runtime.start("hello")
        _events = [event async for event in handle.events()]

    project_state = project / ".soong-agent"
    assert (project_state / "plans").is_dir()
    assert (project_state / "tasks").is_dir()
    assert not (project / "plans").exists()
    for name in ["hooks", "tools", "agents", "skills", "memory", "rules"]:
        assert not (project_state / name).exists()


@pytest.mark.asyncio
async def test_task_create_get_claim_complete_wal(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    create = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="agent.task_create",
            arguments={
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Task",
                "summary": "Summary",
                "steps": [
                    {"step_id": "s1", "title": "Step 1"},
                    {"step_id": "s2", "title": "Step 2", "depends_on_step_ids": ["s1"]},
                ],
            },
        ),
        context,
    )
    assert not create.is_error
    created = create.content[0].data  # type: ignore[union-attr]
    assert created["task"]["steps"][0]["status"] == "ready"
    assert created["wal_event_id"] == created["wal_event_ids"][-1]
    assert created["wal_event_ids"] == ["task_evt_1", "task_evt_2", "task_evt_3"]
    assert (project / ".soong-agent" / "tasks" / "sess_task" / "task1.wal.jsonl").exists()

    claim = await registry.execute(
        ToolCall(tool_call_id="call2", name="agent.task_claim_step", arguments={"task_id": "task1", "step_id": "s1"}),
        context,
    )
    assert not claim.is_error
    assert claim.content[0].data["wal_event_id"] == "task_evt_4"  # type: ignore[union-attr]
    claimed_step = claim.content[0].data["step"]  # type: ignore[union-attr]
    assert claimed_step["status"] == "claimed"
    assert claimed_step["lease_expires_at"] is not None

    complete_step = await registry.execute(
        ToolCall(
            tool_call_id="call3",
            name="agent.task_update_step",
            arguments={"task_id": "task1", "step_id": "s1", "status": "completed", "result_summary": "done"},
        ),
        context,
    )
    assert not complete_step.is_error
    assert complete_step.content[0].data["wal_event_id"] == "task_evt_6"  # type: ignore[union-attr]

    get = await registry.execute(
        ToolCall(tool_call_id="call4", name="agent.task_get", arguments={"task_id": "task1", "include_terminal_steps": True}),
        context,
    )
    steps = get.content[0].data["task"]["steps"]  # type: ignore[union-attr]
    assert {step["step_id"]: step["status"] for step in steps}["s2"] == "ready"
    wal_events = [json.loads(line) for line in (project / ".soong-agent" / "tasks" / "sess_task" / "task1.wal.jsonl").read_text(encoding="utf-8").splitlines()]
    task_running = next(event for event in wal_events if event["event_type"] == "task_running")
    assert task_running["payload"]["previous_status"] == "pending"
    assert task_running["payload"]["status"] == "running"
    ready_events = [event for event in wal_events if event["event_type"] == "task_step_ready" and event["step_id"] == "s2"]
    assert ready_events[-1]["payload"]["depends_on_step_ids"] == ["s1"]


@pytest.mark.asyncio
async def test_task_dag_write_tools_require_orchestrator_role(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    main_context = dataclasses.replace(context, agent_role="main")

    result = await registry.execute(
        ToolCall(
            tool_call_id="main_create",
            name="agent.task_create",
            arguments={
                "task_id": "task_main_forbidden",
                "wal_name": "task_main_forbidden.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        main_context,
    )

    assert result.is_error
    assert result.error and result.error.code == ErrorCode.PERMISSION_DENIED
    assert not (project / ".soong-agent" / "tasks" / "sess_task" / "task_main_forbidden.wal.jsonl").exists()


@pytest.mark.asyncio
async def test_worker_task_reads_require_dispatch_scope(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    create = await registry.execute(
        ToolCall(
            tool_call_id="create_scope",
            name="agent.task_create",
            arguments={
                "task_id": "task_worker_scope_required",
                "wal_name": "task_worker_scope_required.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    assert not create.is_error
    worker_context = dataclasses.replace(context, agent_id="worker1", run_id="run_worker1", agent_role="worker")

    get = await registry.execute(
        ToolCall(
            tool_call_id="worker_get_no_scope",
            name="agent.task_get",
            arguments={"task_id": "task_worker_scope_required", "include_terminal_steps": True},
        ),
        worker_context,
    )
    query = await registry.execute(
        ToolCall(
            tool_call_id="worker_query_no_scope",
            name="agent.task_query_steps",
            arguments={"task_id": "task_worker_scope_required"},
        ),
        worker_context,
    )

    assert get.is_error
    assert get.error and get.error.code == ErrorCode.PERMISSION_DENIED
    assert query.is_error
    assert query.error and query.error.code == ErrorCode.PERMISSION_DENIED


@pytest.mark.asyncio
async def test_task_update_step_cannot_directly_cancel_claimed_or_running_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create_claimed",
            name="agent.task_create",
            arguments={
                "task_id": "task_claimed_cancel",
                "wal_name": "task_claimed_cancel.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="claim_claimed",
            name="agent.task_claim_step",
            arguments={"task_id": "task_claimed_cancel", "step_id": "s1"},
        ),
        context,
    )

    claimed_cancel = await registry.execute(
        ToolCall(
            tool_call_id="cancel_claimed",
            name="agent.task_update_step",
            arguments={"task_id": "task_claimed_cancel", "step_id": "s1", "status": "cancelled"},
        ),
        context,
    )

    assert claimed_cancel.is_error
    assert claimed_cancel.error and claimed_cancel.error.code == ErrorCode.TASK_NOT_DISPATCHABLE

    await registry.execute(
        ToolCall(
            tool_call_id="create_running",
            name="agent.task_create",
            arguments={
                "task_id": "task_running_cancel",
                "wal_name": "task_running_cancel.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="claim_running",
            name="agent.task_claim_step",
            arguments={"task_id": "task_running_cancel", "step_id": "s1"},
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="start_running",
            name="agent.task_update_step",
            arguments={"task_id": "task_running_cancel", "step_id": "s1", "status": "running"},
        ),
        context,
    )

    running_cancel = await registry.execute(
        ToolCall(
            tool_call_id="cancel_running",
            name="agent.task_update_step",
            arguments={"task_id": "task_running_cancel", "step_id": "s1", "status": "cancelled"},
        ),
        context,
    )

    assert running_cancel.is_error
    assert running_cancel.error and running_cancel.error.code == ErrorCode.TASK_NOT_DISPATCHABLE

    wal_text = (project / ".soong-agent" / "tasks" / "sess_task" / "task_running_cancel.wal.jsonl").read_text(encoding="utf-8")
    assert "task_step_cancelled" not in wal_text


@pytest.mark.asyncio
async def test_task_create_steps_schema_rejects_bad_step_before_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    calls = []

    async def allow(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context.permission_callback = allow
    result = await registry.execute(
        ToolCall(
            tool_call_id="create_bad_step",
            name="agent.task_create",
            arguments={
                "task_id": "task_bad_step",
                "wal_name": "task_bad_step.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1"}],
            },
        ),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code == ErrorCode.VALIDATION_ERROR
    assert "$.steps[0].title is required" in result.error.message
    assert calls == []
    assert not (project / ".soong-agent" / "tasks" / "sess_task" / "task_bad_step.wal.jsonl").exists()


@pytest.mark.asyncio
async def test_task_update_operations_schema_rejects_unknown_fields_before_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    calls = []

    async def allow(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context.permission_callback = allow
    result = await registry.execute(
        ToolCall(
            tool_call_id="update_bad_operation",
            name="agent.task_update",
            arguments={
                "task_id": "task_missing",
                "operations": [{"op": "update_task", "title": "Changed", "system_prompt": "inline"}],
            },
        ),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code == ErrorCode.VALIDATION_ERROR
    assert "unknown field" in result.error.message
    assert calls == []


@pytest.mark.asyncio
async def test_task_list_arguments_schema_rejects_non_string_list_items(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    calls = []

    async def allow(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context.permission_callback = allow
    bad_statuses = await registry.execute(
        ToolCall(
            tool_call_id="query_bad_status",
            name="agent.task_query_steps",
            arguments={"task_id": "task_missing", "statuses": ["ready", {"status": "running"}]},
        ),
        context,
    )
    bad_artifacts = await registry.execute(
        ToolCall(
            tool_call_id="update_bad_artifacts",
            name="agent.task_update_step",
            arguments={
                "task_id": "task_missing",
                "step_id": "s1",
                "artifact_ids": ["artifact_ok", {"artifact_id": "bad"}],
            },
        ),
        context,
    )

    assert bad_statuses.is_error
    assert bad_statuses.error and bad_statuses.error.code == ErrorCode.VALIDATION_ERROR
    assert "$.statuses[1] must be string" in bad_statuses.error.message
    assert bad_artifacts.is_error
    assert bad_artifacts.error and bad_artifacts.error.code == ErrorCode.VALIDATION_ERROR
    assert "$.artifact_ids[1] must be string" in bad_artifacts.error.message
    assert calls == []


@pytest.mark.asyncio
async def test_worker_run_can_claim_only_one_step(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create",
            name="agent.task_create",
            arguments={
                "task_id": "task_one_claim",
                "wal_name": "task_one_claim.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}, {"step_id": "s2", "title": "Step 2"}],
            },
        ),
        context,
    )
    first = await registry.execute(
        ToolCall(tool_call_id="claim1", name="agent.task_claim_step", arguments={"task_id": "task_one_claim", "step_id": "s1"}),
        context,
    )
    second = await registry.execute(
        ToolCall(tool_call_id="claim2", name="agent.task_claim_step", arguments={"task_id": "task_one_claim", "step_id": "s2"}),
        context,
    )

    assert not first.is_error
    assert second.is_error
    assert second.error and second.error.code.value == "step_already_claimed_by_run"
    get = await registry.execute(
        ToolCall(tool_call_id="get", name="agent.task_get", arguments={"task_id": "task_one_claim", "include_terminal_steps": True}),
        context,
    )
    steps = {step["step_id"]: step for step in get.content[0].data["task"]["steps"]}  # type: ignore[union-attr]
    assert steps["s1"]["status"] == "claimed"
    assert steps["s2"]["status"] == "ready"
    wal_text = (project / ".soong-agent" / "tasks" / "sess_task" / "task_one_claim.wal.jsonl").read_text(encoding="utf-8")
    assert wal_text.count("task_step_claimed") == 1


@pytest.mark.asyncio
async def test_concurrent_workers_claim_same_step_conflict_writes_single_claim(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    create = await registry.execute(
        ToolCall(
            tool_call_id="create",
            name="agent.task_create",
            arguments={
                "task_id": "task_claim_conflict",
                "wal_name": "task_claim_conflict.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    assert not create.is_error
    worker1 = dataclasses.replace(context, agent_id="worker1", run_id="run_worker1", agent_role="worker")
    worker2 = dataclasses.replace(context, agent_id="worker2", run_id="run_worker2", agent_role="worker")
    worker1.services = {"worker_scope": {"task_id": "task_claim_conflict"}}
    worker2.services = {"worker_scope": {"task_id": "task_claim_conflict"}}

    async def claim(worker_context):
        return await registry.execute(
            ToolCall(
                tool_call_id=f"claim_{worker_context.agent_id}",
                name="agent.task_claim_step",
                arguments={"task_id": "task_claim_conflict", "step_id": "s1"},
            ),
            worker_context,
        )

    first, second = await asyncio.gather(claim(worker1), claim(worker2))

    success = [result for result in (first, second) if not result.is_error]
    failures = [result for result in (first, second) if result.is_error]
    assert len(success) == 1
    assert len(failures) == 1
    assert failures[0].error and failures[0].error.code == ErrorCode.STEP_ALREADY_CLAIMED
    wal_path = project / ".soong-agent" / "tasks" / "sess_task" / "task_claim_conflict.wal.jsonl"
    wal_events = [json.loads(line)["event_type"] for line in wal_path.read_text(encoding="utf-8").splitlines()]
    assert wal_events.count("task_step_claimed") == 1


def test_task_wal_writer_serializes_concurrent_batches(tmp_path) -> None:
    wal_path = tmp_path / "task.wal.jsonl"
    writers = [TaskWalWriter(wal_path) for _ in range(4)]
    batches = [
        [
            {
                "wal_seq": writer_index * 1000 + seq + 1,
                "session_id": "sess_task",
                "event_id": f"task_evt_{writer_index}_{seq}",
                "event_type": "task_step_updated",
                "actor_agent_id": f"agent_{writer_index}",
                "actor_run_id": f"run_{writer_index}",
                "task_id": "task1",
                "step_id": "s1",
                "payload": {"writer": writer_index, "seq": seq},
                "created_at": "2024-01-01T00:00:00+00:00",
            }
            for seq in range(40)
        ]
        for writer_index in range(len(writers))
    ]

    def append_batch(index: int) -> None:
        writers[index].append_many(batches[index])

    with ThreadPoolExecutor(max_workers=len(writers)) as pool:
        list(pool.map(append_batch, range(len(writers))))

    lines = [json.loads(line) for line in wal_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == sum(len(batch) for batch in batches)
    for writer_index in range(len(writers)):
        positions = [index for index, event in enumerate(lines) if event["payload"]["writer"] == writer_index]
        assert positions == list(range(min(positions), min(positions) + len(batches[writer_index])))
        assert [lines[index]["payload"]["seq"] for index in positions] == list(range(len(batches[writer_index])))


def test_task_wal_writer_validates_records_before_append(tmp_path) -> None:
    wal_path = tmp_path / "task.wal.jsonl"
    writer = TaskWalWriter(wal_path)

    with pytest.raises(ValidationError):
        writer.append_many(
            [
                {
                    "wal_seq": 1,
                    "session_id": "sess_task",
                    "event_id": "task_evt_1",
                    "event_type": "task_created",
                    "actor_agent_id": "agent_orchestrator",
                    "actor_run_id": "run_task",
                    "task_id": "task1",
                    "payload": {},
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "unknown": True,
                }
            ]
        )

    assert not wal_path.exists()


@pytest.mark.asyncio
async def test_task_template_returns_stable_template_metadata(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    register_task_tools(registry, TaskService())
    result = await registry.execute(
        ToolCall(tool_call_id="call_task_tpl", name="agent.task_template", arguments={"goal": "build dag"}),
        await make_context(home, project),
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["node_type"] == "task_instruction"
    assert data["template_id"] == "template.task_dag.default"
    assert data["template_version"] == "1"


@pytest.mark.asyncio
async def test_task_reopen_step_emits_reopened_wal_event(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call_reopen_create",
            name="agent.task_create",
            arguments={
                "task_id": "task_reopen",
                "wal_name": "task_reopen.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="call_reopen_update",
            name="agent.task_update_step",
            arguments={"task_id": "task_reopen", "step_id": "s1", "status": "failed", "reason": "nope"},
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="call_reopen_again",
            name="agent.task_update",
            arguments={
                "task_id": "task_reopen",
                "operations": [{"op": "reopen_step", "step_id": "s1", "reason": "retry"}],
            },
        ),
        context,
    )
    wal_events = [
        json.loads(line)
        for line in (project / ".soong-agent" / "tasks" / "sess_task" / "task_reopen.wal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reopened = [event for event in wal_events if event["event_type"] == "task_step_reopened"]
    assert reopened
    assert reopened[-1]["payload"]["previous_status"] == "failed"
    assert reopened[-1]["payload"]["status"] == "pending"
    assert reopened[-1]["payload"]["reason"] == "retry"


@pytest.mark.asyncio
async def test_task_block_and_reopen_emits_task_reopened_and_replay_restores_running(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call_block_create",
            name="agent.task_create",
            arguments={
                "task_id": "task_blocked",
                "wal_name": "task_blocked.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    blocked = await registry.execute(
        ToolCall(
            tool_call_id="call_block",
            name="agent.task_update",
            arguments={"task_id": "task_blocked", "operations": [{"op": "update_task", "status": "blocked", "reason": "waiting"}]},
        ),
        context,
    )
    assert blocked.content[0].data["task"]["status"] == "blocked"  # type: ignore[union-attr]
    with pytest.raises(Exception) as exc_info:
        service.dispatchable_steps(context, task_id="task_blocked", worker_pool_id="default")
    assert getattr(exc_info.value, "code").value == "task_not_dispatchable"

    reopened = await registry.execute(
        ToolCall(
            tool_call_id="call_reopen_task",
            name="agent.task_update",
            arguments={"task_id": "task_blocked", "operations": [{"op": "update_task", "status": "pending", "reason": "resume"}]},
        ),
        context,
    )
    assert reopened.content[0].data["task"]["status"] == "running"  # type: ignore[union-attr]
    wal_events = [
        json.loads(line)
        for line in (project / ".soong-agent" / "tasks" / "sess_task" / "task_blocked.wal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    task_reopened = [event for event in wal_events if event["event_type"] == "task_reopened"]
    assert task_reopened
    assert task_reopened[-1]["payload"] == {"previous_status": "blocked", "status": "pending", "reason": "resume"}

    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)
    get = await registry2.execute(
        ToolCall(
            tool_call_id="call_get_reopened",
            name="agent.task_get",
            arguments={"task_id": "task_blocked", "include_terminal_steps": True},
        ),
        context,
    )
    assert get.content[0].data["task"]["status"] == "running"  # type: ignore[union-attr]
    assert get.content[0].data["task"]["steps"][0]["status"] == "ready"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_task_step_running_renews_lease_and_blocked_clears_claim(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call1",
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
    claim = await registry.execute(
        ToolCall(tool_call_id="call2", name="agent.task_claim_step", arguments={"task_id": "task1", "step_id": "s1"}),
        context,
    )
    first_lease = claim.content[0].data["step"]["lease_expires_at"]  # type: ignore[union-attr]
    running = await registry.execute(
        ToolCall(
            tool_call_id="call3",
            name="agent.task_update_step",
            arguments={"task_id": "task1", "step_id": "s1", "status": "running"},
        ),
        context,
    )
    second_lease = running.content[0].data["step"]["lease_expires_at"]  # type: ignore[union-attr]
    assert second_lease is not None
    assert second_lease >= first_lease

    blocked = await registry.execute(
        ToolCall(
            tool_call_id="call4",
            name="agent.task_update_step",
            arguments={"task_id": "task1", "step_id": "s1", "status": "blocked", "reason": "needs input"},
        ),
        context,
    )
    step = blocked.content[0].data["step"]  # type: ignore[union-attr]
    assert step["status"] == "blocked"
    assert step["claimed_by_run_id"] is None
    assert step["lease_expires_at"] is None


@pytest.mark.asyncio
async def test_task_wal_step_payloads_match_contract(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)

    async def create(task_id: str, wal_name: str, step_id: str = "s1") -> None:
        await registry.execute(
            ToolCall(
                tool_call_id=f"create_{task_id}",
                name="agent.task_create",
                arguments={
                    "task_id": task_id,
                    "wal_name": wal_name,
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": step_id, "title": "Step 1"}],
                },
            ),
            context,
        )

    await create("task_started", "task_started.wal.jsonl")
    await registry.execute(ToolCall(tool_call_id="claim_started", name="agent.task_claim_step", arguments={"task_id": "task_started", "step_id": "s1"}), context)
    await registry.execute(ToolCall(tool_call_id="start_started", name="agent.task_update_step", arguments={"task_id": "task_started", "step_id": "s1", "status": "running"}), context)

    await create("task_blocked", "task_blocked.wal.jsonl")
    await registry.execute(ToolCall(tool_call_id="claim_blocked", name="agent.task_claim_step", arguments={"task_id": "task_blocked", "step_id": "s1"}), context)
    await registry.execute(
        ToolCall(
            tool_call_id="block_blocked",
            name="agent.task_update_step",
            arguments={"task_id": "task_blocked", "step_id": "s1", "status": "blocked", "reason": "needs input"},
        ),
        context,
    )

    await create("task_completed", "task_completed.wal.jsonl")
    await registry.execute(ToolCall(tool_call_id="claim_completed", name="agent.task_claim_step", arguments={"task_id": "task_completed", "step_id": "s1"}), context)
    await registry.execute(
        ToolCall(
            tool_call_id="complete_completed",
            name="agent.task_update_step",
            arguments={"task_id": "task_completed", "step_id": "s1", "status": "completed", "result_summary": "done", "artifact_ids": ["art1"]},
        ),
        context,
    )

    await create("task_cancelled", "task_cancelled.wal.jsonl")
    await registry.execute(ToolCall(tool_call_id="cancel_cancelled", name="agent.task_cancel", arguments={"task_id": "task_cancelled"}), context)

    await create("task_failed", "task_failed.wal.jsonl")
    await registry.execute(ToolCall(tool_call_id="fail_failed", name="agent.task_fail", arguments={"task_id": "task_failed"}), context)

    events = []
    for wal in (project / ".soong-agent" / "tasks" / "sess_task").glob("*.wal.jsonl"):
        events.extend(json.loads(line) for line in wal.read_text(encoding="utf-8").splitlines())
    latest = {}
    for event in events:
        if event["event_type"].startswith("task_step_"):
            latest[event["event_type"]] = event

    assert set(latest["task_step_started"]["payload"]) == {"previous_status", "status", "lease_expires_at"}
    assert set(latest["task_step_blocked"]["payload"]) == {"previous_status", "reason", "result_summary", "artifact_ids"}
    assert set(latest["task_step_completed"]["payload"]) == {"previous_status", "result_summary", "artifact_ids"}
    assert set(latest["task_step_cancelled"]["payload"]) == {"previous_status", "reason", "result_summary"}
    assert set(latest["task_step_failed"]["payload"]) == {"previous_status", "reason", "result_summary", "artifact_ids"}


@pytest.mark.asyncio
async def test_worker_ready_query_defaults_to_five_steps(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create_worker_ready_limit",
            name="agent.task_create",
            arguments={
                "task_id": "task_worker_ready_limit",
                "wal_name": "task_worker_ready_limit.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": f"s{i}", "title": f"Step {i}"} for i in range(8)],
            },
        ),
        context,
    )
    worker_context = dataclasses.replace(
        context,
        agent_id="worker1",
        run_id="run_worker1",
        agent_role="worker",
        services={"worker_scope": {"task_id": "task_worker_ready_limit"}},
    )

    query = await registry.execute(
        ToolCall(
            tool_call_id="worker_ready_default_limit",
            name="agent.task_query_steps",
            arguments={"task_id": "task_worker_ready_limit", "statuses": ["ready"]},
        ),
        worker_context,
    )

    assert not query.is_error
    assert len(query.content[0].data["steps"]) == 5  # type: ignore[union-attr]
    assert query.content[0].data["truncated"] is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_task_lease_expired_reopens_step_and_replay_restores_ready(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call1",
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
    await registry.execute(
        ToolCall(tool_call_id="call2", name="agent.task_claim_step", arguments={"task_id": "task1", "step_id": "s1"}),
        context,
    )
    record = service._record("sess_task", "task1")
    record.task.steps[0].lease_expires_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    query = await registry.execute(
        ToolCall(
            tool_call_id="call3",
            name="agent.task_query_steps",
            arguments={"task_id": "task1", "statuses": ["ready"], "include_terminal_steps": False},
        ),
        context,
    )
    steps = query.content[0].data["steps"]  # type: ignore[union-attr]
    assert steps[0]["step_id"] == "s1"
    assert steps[0]["status"] == "ready"
    assert steps[0]["claimed_by_run_id"] is None
    wal_text = (project / ".soong-agent" / "tasks" / "sess_task" / "task1.wal.jsonl").read_text(encoding="utf-8")
    assert "task_step_lease_expired" in wal_text
    assert "task_step_ready" in wal_text

    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)
    get = await registry2.execute(
        ToolCall(tool_call_id="call4", name="agent.task_get", arguments={"task_id": "task1", "include_terminal_steps": True}),
        context,
    )
    step = get.content[0].data["task"]["steps"][0]  # type: ignore[union-attr]
    assert step["status"] == "ready"
    assert step["claimed_by_run_id"] is None


@pytest.mark.asyncio
async def test_task_cycle_rejected(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="agent.task_create",
            arguments={
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [
                    {"step_id": "s1", "title": "Step 1", "depends_on_step_ids": ["s2"]},
                    {"step_id": "s2", "title": "Step 2", "depends_on_step_ids": ["s1"]},
                ],
            },
        ),
        context,
    )
    assert result.is_error
    assert result.error and result.error.code.value == "dependency_cycle"


@pytest.mark.asyncio
async def test_task_create_rejects_existing_wal_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    wal_path = project / ".soong-agent" / "tasks" / "sess_task"
    wal_path.mkdir(parents=True, exist_ok=True)
    (wal_path / "dup.wal.jsonl").write_text("existing\n", encoding="utf-8")
    result = await registry.execute(
        ToolCall(
            tool_call_id="call_path_conflict",
            name="agent.task_create",
            arguments={
                "task_id": "task_conflict",
                "wal_name": "dup.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    assert result.is_error
    assert result.error and result.error.code.value == "path_conflict"


@pytest.mark.asyncio
async def test_task_create_wal_append_failure_returns_task_wal_unavailable_and_keeps_memory_empty(
    isolated_dirs, monkeypatch
) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)

    def fail_append_many(_self, _payloads):
        raise OSError("disk unavailable")

    monkeypatch.setattr("agent_core.storage.task_wal.TaskWalWriter.append_many", fail_append_many)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call_wal_unavailable",
            name="agent.task_create",
            arguments={
                "task_id": "task_wal_fail",
                "wal_name": "task_wal_fail.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code.value == "task_wal_unavailable"
    assert service._records == {}
    assert not (project / ".soong-agent" / "tasks" / "sess_task" / "task_wal_fail.wal.jsonl").exists()


@pytest.mark.asyncio
async def test_task_update_is_atomic_when_later_operation_fails(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="agent.task_create",
            arguments={
                "task_id": "task1",
                "wal_name": "task1.wal.jsonl",
                "title": "Original",
                "summary": "",
                "steps": [
                    {"step_id": "s1", "title": "Step 1"},
                    {"step_id": "s2", "title": "Step 2"},
                ],
            },
        ),
        context,
    )
    failed = await registry.execute(
        ToolCall(
            tool_call_id="call2",
            name="agent.task_update",
            arguments={
                "task_id": "task1",
                "operations": [
                    {"op": "update_task", "title": "Mutated"},
                    {"op": "add_dependency", "step_id": "s1", "depends_on_step_id": "s2"},
                    {"op": "add_dependency", "step_id": "s2", "depends_on_step_id": "s1"},
                ],
            },
        ),
        context,
    )
    assert failed.is_error
    assert failed.error and failed.error.code.value == "dependency_cycle"
    current = await registry.execute(
        ToolCall(tool_call_id="call3", name="agent.task_get", arguments={"task_id": "task1", "include_terminal_steps": True}),
        context,
    )
    task = current.content[0].data["task"]  # type: ignore[union-attr]
    assert task["title"] == "Original"
    assert {step["step_id"]: step["depends_on_step_ids"] for step in task["steps"]} == {"s1": [], "s2": []}
    wal_path = project / ".soong-agent" / "tasks" / "sess_task" / "task1.wal.jsonl"
    assert "task_updated" not in wal_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_task_wal_replay_and_terminal_list(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call1",
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
    await registry.execute(
        ToolCall(tool_call_id="call2", name="agent.task_claim_step", arguments={"task_id": "task1", "step_id": "s1"}),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="call3",
            name="agent.task_update_step",
            arguments={"task_id": "task1", "step_id": "s1", "status": "completed"},
        ),
        context,
    )
    await registry.execute(
        ToolCall(tool_call_id="call4", name="agent.task_complete", arguments={"task_id": "task1"}),
        context,
    )
    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)
    hidden = await registry2.execute(
        ToolCall(tool_call_id="call5", name="agent.task_list", arguments={}),
        context,
    )
    assert hidden.content[0].data["tasks"] == []  # type: ignore[union-attr]
    visible = await registry2.execute(
        ToolCall(tool_call_id="call6", name="agent.task_list", arguments={"include_terminal": True}),
        context,
    )
    assert visible.content[0].data["tasks"][0]["status"] == "completed"  # type: ignore[union-attr]
    assert replayed._records == {}
    assert ("sess_task", "task1") in replayed._terminal_records

    terminal_get = await registry2.execute(
        ToolCall(tool_call_id="call7", name="agent.task_get", arguments={"task_id": "task1", "include_terminal_steps": True}),
        context,
    )
    terminal_query = await registry2.execute(
        ToolCall(
            tool_call_id="call7_query",
            name="agent.task_query_steps",
            arguments={"task_id": "task1", "include_terminal_steps": True},
        ),
        context,
    )
    late_update = await registry2.execute(
        ToolCall(
            tool_call_id="call8",
            name="agent.task_update_step",
            arguments={"task_id": "task1", "step_id": "s1", "status": "failed"},
        ),
        context,
    )
    assert not terminal_get.is_error
    assert not terminal_query.is_error
    assert terminal_query.content[0].data["steps"][0]["status"] == "completed"  # type: ignore[union-attr]
    assert late_update.is_error
    assert late_update.error and late_update.error.code.value == "task_terminal"


@pytest.mark.asyncio
async def test_task_wal_replay_failure_marks_task_unavailable(isolated_dirs) -> None:
    home, project = isolated_dirs
    context = await make_context(home, project)
    task_dir = project / ".soong-agent" / "tasks" / "sess_task"
    task_dir.mkdir(parents=True, exist_ok=True)
    bad_wal = task_dir / "bad_task.wal.jsonl"
    bad_wal.write_text("{not json}\n", encoding="utf-8")

    replayed = TaskService()
    replayed.replay_project(project)
    registry = ToolRegistry()
    register_task_tools(registry, replayed)

    listed = await registry.execute(
        ToolCall(tool_call_id="list_bad", name="agent.task_list", arguments={"include_terminal": True}),
        context,
    )
    get = await registry.execute(
        ToolCall(
            tool_call_id="get_bad",
            name="agent.task_get",
            arguments={"task_id": "bad_task", "include_terminal_steps": True},
        ),
        context,
    )

    assert listed.content[0].data["tasks"][0]["task_id"] == "bad_task"  # type: ignore[union-attr]
    assert listed.content[0].data["tasks"][0]["status"] == "unavailable"  # type: ignore[union-attr]
    assert listed.content[0].data["tasks"][0]["error"]["code"] == "task_wal_unavailable"  # type: ignore[union-attr]
    assert get.is_error
    assert get.error and get.error.code.value == "task_wal_unavailable"
    assert get.error.details["wal_path"] == str(bad_wal)


@pytest.mark.asyncio
async def test_runtime_start_ignores_bad_task_wal_and_reports_unavailable(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    task_dir = project / ".soong-agent" / "tasks" / "sess_task"
    task_dir.mkdir(parents=True, exist_ok=True)
    bad_wal = task_dir / "bad_task.wal.jsonl"
    bad_wal.write_text("{not json}\n", encoding="utf-8")
    scripted_ollama.enqueue_text("hello")

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        handle = await runtime.start("hi", session_id="sess_runtime_bad_wal")
        _events = [event async for event in handle.events()]
        context = await make_context(home, project)
        listed = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="list_bad", name="agent.task_list", arguments={"include_terminal": True}),
            dataclasses.replace(
                context,
                services={"runtime": runtime},
            ),
        )
        replay = await runtime.replay_session("sess_task")

    assert listed.content[0].data["tasks"][0]["task_id"] == "bad_task"  # type: ignore[union-attr]
    assert listed.content[0].data["tasks"][0]["status"] == "unavailable"  # type: ignore[union-attr]
    assert listed.content[0].data["tasks"][0]["error"]["details"]["wal_path"] == str(bad_wal)  # type: ignore[union-attr]
    assert replay.task_wal_errors
    assert replay.task_wal_errors[0]["error"]["code"] == "task_wal_unavailable"
    assert replay.task_wal_errors[0]["error"]["details"]["wal_path"] == str(bad_wal)


@pytest.mark.asyncio
async def test_task_terminal_list_orders_by_updated_at_and_pages(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)

    async def create_and_complete(task_id: str) -> None:
        await registry.execute(
            ToolCall(
                tool_call_id=f"create_{task_id}",
                name="agent.task_create",
                arguments={
                    "task_id": task_id,
                    "wal_name": f"{task_id}.wal.jsonl",
                    "title": task_id,
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}],
                },
            ),
            context,
        )
        await registry.execute(
            ToolCall(tool_call_id=f"claim_{task_id}", name="agent.task_claim_step", arguments={"task_id": task_id, "step_id": "s1"}),
            context,
        )
        await registry.execute(
            ToolCall(
                tool_call_id=f"step_{task_id}",
                name="agent.task_update_step",
                arguments={"task_id": task_id, "step_id": "s1", "status": "completed"},
            ),
            context,
        )
        await registry.execute(
            ToolCall(tool_call_id=f"complete_{task_id}", name="agent.task_complete", arguments={"task_id": task_id}),
            context,
        )

    await create_and_complete("task_old")
    time.sleep(0.01)
    await create_and_complete("task_new")

    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)

    first_page = await registry2.execute(
        ToolCall(tool_call_id="list_first", name="agent.task_list", arguments={"include_terminal": True, "limit": 1, "offset": 0}),
        context,
    )
    second_page = await registry2.execute(
        ToolCall(tool_call_id="list_second", name="agent.task_list", arguments={"include_terminal": True, "limit": 1, "offset": 1}),
        context,
    )
    assert first_page.content[0].data["tasks"][0]["task_id"] == "task_new"  # type: ignore[union-attr]
    assert first_page.content[0].data["truncated"] is True  # type: ignore[union-attr]
    assert second_page.content[0].data["tasks"][0]["task_id"] == "task_old"  # type: ignore[union-attr]
    assert second_page.content[0].data["truncated"] is False  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_task_complete_replay_cancels_optional_steps(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="call_create",
            name="agent.task_create",
            arguments={
                "task_id": "task_optional",
                "wal_name": "task_optional.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [
                    {"step_id": "required", "title": "Required"},
                    {"step_id": "optional", "title": "Optional", "required": False},
                ],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="call_complete_step",
            name="agent.task_update_step",
            arguments={"task_id": "task_optional", "step_id": "required", "status": "completed"},
        ),
        context,
    )
    complete = await registry.execute(
        ToolCall(tool_call_id="call_complete_task", name="agent.task_complete", arguments={"task_id": "task_optional"}),
        context,
    )
    assert not complete.is_error
    live_steps = {step["step_id"]: step for step in complete.content[0].data["task"]["steps"]}  # type: ignore[union-attr]
    assert live_steps["optional"]["status"] == "cancelled"
    assert live_steps["optional"]["reason"] == "task_completed"

    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)
    get = await registry2.execute(
        ToolCall(
            tool_call_id="call_get",
            name="agent.task_get",
            arguments={"task_id": "task_optional", "include_terminal_steps": True},
        ),
        context,
    )
    replay_steps = {step["step_id"]: step for step in get.content[0].data["task"]["steps"]}  # type: ignore[union-attr]
    assert replay_steps["optional"]["status"] == "cancelled"
    assert replay_steps["optional"]["reason"] == "task_completed"
    wal_events = [
        json.loads(line)
        for line in (project / ".soong-agent" / "tasks" / "sess_task" / "task_optional.wal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event_type"] == "task_step_cancelled" and event["step_id"] == "optional" for event in wal_events)


@pytest.mark.asyncio
async def test_task_cancel_and_fail_replay_terminate_unfinished_steps(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    for task_id, wal_name, tool_name, expected_status in [
        ("task_cancel", "task_cancel.wal.jsonl", "agent.task_cancel", "cancelled"),
        ("task_fail", "task_fail.wal.jsonl", "agent.task_fail", "failed"),
    ]:
        await registry.execute(
            ToolCall(
                tool_call_id=f"create_{task_id}",
                name="agent.task_create",
                arguments={
                    "task_id": task_id,
                    "wal_name": wal_name,
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}],
                },
            ),
            context,
        )
        await registry.execute(
            ToolCall(tool_call_id=f"terminate_{task_id}", name=tool_name, arguments={"task_id": task_id, "reason": "stop"}),
            context,
        )

    replayed = TaskService()
    replayed.replay_project(project)
    registry2 = ToolRegistry()
    register_task_tools(registry2, replayed)
    for task_id, expected_status in [("task_cancel", "cancelled"), ("task_fail", "failed")]:
        get = await registry2.execute(
            ToolCall(
                tool_call_id=f"get_{task_id}",
                name="agent.task_get",
                arguments={"task_id": task_id, "include_terminal_steps": True},
            ),
            context,
        )
        task = get.content[0].data["task"]  # type: ignore[union-attr]
        assert task["status"] == expected_status
        assert task["steps"][0]["status"] == expected_status
        assert task["steps"][0]["reason"] == f"task_{expected_status}"
        assert task["steps"][0]["result_summary"] == f"task_{expected_status}"
        wal_text = (project / ".soong-agent" / "tasks" / "sess_task" / f"{task_id}.wal.jsonl").read_text(encoding="utf-8")
        assert f"task_step_{expected_status}" in wal_text


@pytest.mark.asyncio
async def test_task_delete_step_status_and_dependent_constraints(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create_delete_task",
            name="agent.task_create",
            arguments={
                "task_id": "task_delete",
                "wal_name": "task_delete.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [
                    {"step_id": "ready", "title": "Ready"},
                    {"step_id": "blocked", "title": "Blocked"},
                    {"step_id": "dep", "title": "Dependent", "depends_on_step_ids": ["blocked"]},
                ],
            },
        ),
        context,
    )
    blocked = await registry.execute(
        ToolCall(
            tool_call_id="block_step",
            name="agent.task_update_step",
            arguments={"task_id": "task_delete", "step_id": "blocked", "status": "blocked"},
        ),
        context,
    )
    assert not blocked.is_error

    blocked_delete = await registry.execute(
        ToolCall(
            tool_call_id="delete_blocked",
            name="agent.task_update",
            arguments={"task_id": "task_delete", "operations": [{"op": "delete_step", "step_id": "blocked"}]},
        ),
        context,
    )
    assert blocked_delete.is_error
    assert blocked_delete.error and blocked_delete.error.code.value == "task_not_dispatchable"

    reopen_and_delete = await registry.execute(
        ToolCall(
            tool_call_id="delete_ready_dependent",
            name="agent.task_update",
            arguments={
                "task_id": "task_delete",
                "operations": [
                    {"op": "reopen_step", "step_id": "blocked"},
                    {"op": "delete_step", "step_id": "blocked"},
                ],
            },
        ),
        context,
    )
    assert reopen_and_delete.is_error
    assert reopen_and_delete.error and reopen_and_delete.error.code.value == "step_has_dependents"

    delete_ready = await registry.execute(
        ToolCall(
            tool_call_id="delete_ready",
            name="agent.task_update",
            arguments={"task_id": "task_delete", "operations": [{"op": "delete_step", "step_id": "ready"}]},
        ),
        context,
    )
    assert not delete_ready.is_error
    step_ids = {step["step_id"] for step in delete_ready.content[0].data["task"]["steps"]}  # type: ignore[union-attr]
    assert "ready" not in step_ids


@pytest.mark.asyncio
async def test_worker_cannot_modify_task_content_fields_via_task_update(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    create = await registry.execute(
        ToolCall(
            tool_call_id="create_worker_content",
            name="agent.task_create",
            arguments={
                "task_id": "task_worker_content",
                "wal_name": "task_worker_content.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1", "summary": "original"}],
            },
        ),
        context,
    )
    assert not create.is_error
    wal_path = project / ".soong-agent" / "tasks" / "sess_task" / "task_worker_content.wal.jsonl"
    before_wal = wal_path.read_text(encoding="utf-8")
    worker_context = dataclasses.replace(
        context,
        agent_id="worker1",
        run_id="run_worker1",
        agent_role="worker",
        services={"worker_scope": {"task_id": "task_worker_content", "allowed_step_ids": ["s1"]}},
    )

    result = await registry.execute(
        ToolCall(
            tool_call_id="worker_update_content",
            name="agent.task_update",
            arguments={
                "task_id": "task_worker_content",
                "operations": [{"op": "update_step", "step_id": "s1", "title": "Changed", "summary": "changed"}],
            },
        ),
        worker_context,
    )
    get = await registry.execute(
        ToolCall(tool_call_id="get_worker_content", name="agent.task_get", arguments={"task_id": "task_worker_content", "include_terminal_steps": True}),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code == ErrorCode.PERMISSION_DENIED
    step = get.content[0].data["task"]["steps"][0]  # type: ignore[union-attr]
    assert step["title"] == "Step 1"
    assert step["summary"] == "original"
    assert wal_path.read_text(encoding="utf-8") == before_wal


@pytest.mark.asyncio
async def test_orchestrator_can_modify_running_step_content(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create_running_content",
            name="agent.task_create",
            arguments={
                "task_id": "task_running_content",
                "wal_name": "task_running_content.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1", "summary": "original"}],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(tool_call_id="claim_running_content", name="agent.task_claim_step", arguments={"task_id": "task_running_content", "step_id": "s1"}),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="start_running_content",
            name="agent.task_update_step",
            arguments={"task_id": "task_running_content", "step_id": "s1", "status": "running"},
        ),
        context,
    )

    result = await registry.execute(
        ToolCall(
            tool_call_id="orchestrator_update_running_content",
            name="agent.task_update",
            arguments={
                "task_id": "task_running_content",
                "operations": [{"op": "update_step", "step_id": "s1", "title": "Changed", "summary": "changed"}],
            },
        ),
        context,
    )

    assert not result.is_error
    step = result.content[0].data["task"]["steps"][0]  # type: ignore[union-attr]
    assert step["status"] == "running"
    assert step["title"] == "Changed"
    assert step["summary"] == "changed"
    assert step["metadata"]["updated_after_dispatch"] is True


@pytest.mark.asyncio
async def test_task_terminal_step_update_does_not_write_wal(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    service = TaskService()
    register_task_tools(registry, service)
    context = await make_context(home, project)
    await registry.execute(
        ToolCall(
            tool_call_id="create_terminal",
            name="agent.task_create",
            arguments={
                "task_id": "task_terminal",
                "wal_name": "task_terminal.wal.jsonl",
                "title": "Task",
                "summary": "",
                "steps": [{"step_id": "s1", "title": "Step 1"}],
            },
        ),
        context,
    )
    await registry.execute(
        ToolCall(
            tool_call_id="complete_step",
            name="agent.task_update_step",
            arguments={"task_id": "task_terminal", "step_id": "s1", "status": "completed"},
        ),
        context,
    )
    await registry.execute(
        ToolCall(tool_call_id="complete_task", name="agent.task_complete", arguments={"task_id": "task_terminal"}),
        context,
    )
    wal_path = project / ".soong-agent" / "tasks" / "sess_task" / "task_terminal.wal.jsonl"
    before = wal_path.read_text(encoding="utf-8")
    late = await registry.execute(
        ToolCall(
            tool_call_id="late_update",
            name="agent.task_update_step",
            arguments={"task_id": "task_terminal", "step_id": "s1", "status": "failed", "reason": "late"},
        ),
        context,
    )
    after = wal_path.read_text(encoding="utf-8")
    assert late.is_error
    assert late.error and late.error.code.value == "task_terminal"
    assert after == before
