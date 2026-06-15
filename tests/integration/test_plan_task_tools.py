from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import time

import pytest

from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.permissions import PermissionSessionCache
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.tasks.service import TaskService
from agent_core.tasks.tools import register_task_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolCall
from tests.conftest import write_config


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
    assert data["template_id"] == "template.plan.default"
    assert data["template_version"] == "1"


@pytest.mark.asyncio
async def test_plan_template_uses_configured_default_dir(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config_path = home / "config.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n[plan]\ndefault_dir = \"<project>/plans/custom\"\n", encoding="utf-8")
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
    assert data["suggested_dir"] == str((project / "plans" / "custom").resolve())


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
    assert (project / ".soong-agent" / "tasks" / "sess_task" / "task1.wal.jsonl").exists()

    claim = await registry.execute(
        ToolCall(tool_call_id="call2", name="agent.task_claim_step", arguments={"task_id": "task1", "step_id": "s1"}),
        context,
    )
    assert not claim.is_error
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
