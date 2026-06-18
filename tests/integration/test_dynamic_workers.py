from __future__ import annotations

import json
import sqlite3
import asyncio

import pytest

from agent_core import AgentRuntime
from agent_core.agents.workers import worker_agent_id_for_session
from agent_core.errors import AgentCoreError
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types import RunDirectives
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_loads_json_agent_and_worker(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    _write_json(
        home / "agents" / "reviewer.json",
        {
            "id": "json_reviewer",
            "name": "JSON Reviewer",
            "description": "Review from JSON",
            "system_prompt": "Review carefully.",
            "suggested_tools": ["code.read_file"],
            "tags": ["review"],
        },
    )
    _write_json(
        home / "workers" / "review_worker.json",
        {
            "worker_id": "review_worker",
            "worker_pool_id": "review",
            "name": "Review Worker",
            "description": "Reviews code",
            "agent_definition_id": "json_reviewer",
            "allowed_tools": ["code.read_file"],
        },
    )

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        workers = await runtime.list_worker_configs()
        worker = next(item for item in workers if item.worker_id == "review_worker")
        assert worker.worker_pool_id == "review"
        assert worker.agent_definition_id == "json_reviewer"
        assert worker.source == "json"
        definition = runtime.agent_definitions.get("json_reviewer")
        assert definition is not None
        assert definition.source == "json"
        assert definition.body == "Review carefully."
        assert runtime.worker_runtime is not None
        assert [item.worker_id for item in runtime.worker_runtime.list_workers("review")] == ["review_worker"]


@pytest.mark.asyncio
async def test_runtime_loads_json_worker_with_inline_agent(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    _write_json(
        home / "workers" / "inline_reviewer.json",
        {
            "worker_id": "inline_reviewer",
            "worker_pool_id": "review",
            "enabled": True,
            "agent": {
                "agent_definition_id": "inline_code_reviewer",
                "name": "Inline Code Reviewer",
                "description": "Reviews code changes.",
                "model": {
                    "provider": "openai",
                    "name": "qwen2.5:7b",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "ollama",
                    "temperature": 0.2,
                },
                "system_prompt": "You are a senior code reviewer.",
                "suggested_tools": ["code.read_file", "code.search", "code.read_file"],
                "tags": ["review", "review"],
            },
            "allowed_tools": ["code.read_file", "code.search"],
        },
    )

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        worker = await runtime.get_worker_config("inline_reviewer")
        assert worker is not None
        assert worker.source == "json"
        assert worker.worker_pool_id == "review"
        assert worker.agent_definition_id == "inline_code_reviewer"
        assert worker.name == "Inline Code Reviewer"
        assert worker.description == "Reviews code changes."
        assert worker.system_prompt == "You are a senior code reviewer."
        assert worker.model == {
            "provider": "openai",
            "name": "qwen2.5:7b",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 0.2,
        }
        assert worker.allowed_tools == ["code.read_file", "code.search"]
        assert worker.metadata["inline_agent"] is True
        assert worker.metadata["inline_agent_definition"]["suggested_tools"] == ["code.read_file", "code.search"]
        assert worker.metadata["inline_agent_definition"]["tags"] == ["review"]

        definition = runtime.agent_definitions.get("inline_code_reviewer")
        assert definition is not None
        assert definition.source == "json"
        assert definition.body == "You are a senior code reviewer."
        assert definition.model_profile == worker.model
        assert definition.suggested_tools == ["code.read_file", "code.search"]
        assert definition.tags == ["worker", "review"]
        assert definition.metadata["worker_id"] == "inline_reviewer"
        assert runtime.worker_runtime is not None
        assert [item.worker_id for item in runtime.worker_runtime.list_workers("review")] == ["inline_reviewer"]


@pytest.mark.asyncio
async def test_runtime_dynamic_worker_crud_and_reload(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        created = await runtime.create_worker_config(
            {
                "worker_id": "hub_worker",
                "worker_pool_id": "hub",
                "name": "Hub Worker",
                "description": "Created by hub",
                "system_prompt": "Handle hub work.",
                "model": {"name": "worker-model", "temperature": 0.0},
                "allowed_tools": ["code.read_file"],
            }
        )
        assert created.worker_id == "hub_worker"
        assert created.agent_definition_id == "worker.hub_worker"
        assert created.source == "dynamic"
        assert runtime.agent_definitions.get("worker.hub_worker").body == "Handle hub work."  # type: ignore[union-attr]
        assert runtime.agent_definitions.get("worker.hub_worker").model_profile["name"] == "worker-model"  # type: ignore[index,union-attr]
        assert runtime.worker_runtime is not None
        assert [item.worker_id for item in runtime.worker_runtime.list_workers("hub")] == ["hub_worker"]

        updated = await runtime.update_worker_config(
            "hub_worker",
            {
                "name": "Updated Hub Worker",
                "system_prompt": "Updated prompt.",
                "allowed_tools": ["code.read_file", "code.search"],
            },
        )
        assert updated.name == "Updated Hub Worker"
        assert updated.allowed_tools == ["code.read_file", "code.search"]
        assert runtime.agent_definitions.get("worker.hub_worker").body == "Updated prompt."  # type: ignore[union-attr]

        disabled = await runtime.disable_worker_config("hub_worker")
        assert disabled.enabled is False
        assert runtime.worker_runtime.list_workers("hub") == []
        assert [item.worker_id for item in await runtime.list_worker_configs()] == ["hub_worker"]
        assert await runtime.list_worker_configs(include_disabled=False) == []

        enabled = await runtime.enable_worker_config("hub_worker")
        assert enabled.enabled is True
        assert [item.worker_id for item in runtime.worker_runtime.list_workers("hub")] == ["hub_worker"]

        deleted = await runtime.soft_delete_worker_config("hub_worker")
        assert deleted.deleted_at is not None
        assert runtime.worker_runtime.list_workers("hub") == []
        assert await runtime.list_worker_configs() == []
        deleted_rows = await runtime.list_worker_configs(include_deleted=True)
        assert deleted_rows[0].worker_id == "hub_worker"
        assert deleted_rows[0].deleted_at is not None


@pytest.mark.asyncio
async def test_dynamic_sqlite_worker_overrides_json_worker(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    _write_json(
        home / "workers" / "shared.json",
        {
            "worker_id": "shared_worker",
            "worker_pool_id": "json",
            "name": "JSON Worker",
            "system_prompt": "JSON prompt.",
            "allowed_tools": ["code.read_file"],
        },
    )

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        assert (await runtime.get_worker_config("shared_worker")).source == "json"  # type: ignore[union-attr]
        await runtime.create_worker_config(
            {
                "worker_id": "shared_worker",
                "worker_pool_id": "dynamic",
                "name": "Dynamic Worker",
                "system_prompt": "Dynamic prompt.",
                "allowed_tools": ["code.search"],
            }
        )
        effective = await runtime.get_worker_config("shared_worker")
        assert effective is not None
        assert effective.source == "dynamic"
        assert effective.worker_pool_id == "dynamic"
        assert effective.allowed_tools == ["code.search"]
        assert runtime.agent_definitions.get("worker.shared_worker").body == "Dynamic prompt."  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_agent_list_workers_shows_dynamic_worker(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), permission_callback=allow) as runtime:
        assert runtime.config and runtime.paths and runtime.artifacts
        await runtime.create_worker_config(
            {
                "worker_id": "catalog_worker",
                "worker_pool_id": "catalog",
                "name": "Catalog Worker",
                "description": "Visible in list",
                "system_prompt": "Catalog prompt.",
                "allowed_tools": ["code.read_file"],
            }
        )
        context = ToolExecutionContext(
            session_id="sess_catalog",
            run_id="run_catalog",
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=runtime.paths.project_dir,
            home_dir=runtime.paths.home_dir,
            config=runtime.config,
            artifact_manager=runtime.artifacts,
            permission_callback=allow,
            services={
                "task_service": runtime.task_service,
                "runtime": runtime,
                "agent_definitions": runtime.agent_definitions,
            },
        )
        result = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="list_workers", name="agent.list_workers", arguments={"worker_pool_id": "catalog"}),
            context,
        )

    assert not result.is_error
    workers = result.content[0].data["workers"]  # type: ignore[union-attr]
    assert len(workers) == 1
    assert workers[0]["worker_id"] == "catalog_worker"
    assert workers[0]["name"] == "Catalog Worker"
    assert workers[0]["description"] == "Visible in list"
    assert workers[0]["allowed_tools"] == ["code.read_file"]


@pytest.mark.asyncio
async def test_dynamic_worker_create_validates_duplicate_active_worker(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        await runtime.create_worker_config({"worker_id": "dup_worker", "system_prompt": "one"})
        with pytest.raises(AgentCoreError) as exc:
            await runtime.create_worker_config({"worker_id": "dup_worker", "system_prompt": "two"})
    assert exc.value.code.value == "config_error"


@pytest.mark.asyncio
async def test_resolve_worker_mention_by_id_name_disabled_deleted_and_ambiguous_state(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        await runtime.create_worker_config(
            {"worker_id": "mention_worker", "name": "Mention Worker", "system_prompt": "Mention prompt."}
        )
        await runtime.create_worker_config(
            {"worker_id": "ambiguous_one", "name": "Ambiguous Worker", "system_prompt": "First ambiguous prompt."}
        )
        await runtime.create_worker_config(
            {"worker_id": "ambiguous_two", "name": "Ambiguous Worker", "system_prompt": "Second ambiguous prompt."}
        )
        await runtime.create_worker_config(
            {"worker_id": "deleted_worker", "name": "Deleted Worker", "system_prompt": "Deleted prompt."}
        )
        by_id = runtime.resolve_worker_mention("@mention_worker", session_id="sess_mentions")
        assert by_id.resolved is True
        assert by_id.worker_id == "mention_worker"
        assert by_id.worker_agent_id == worker_agent_id_for_session(session_id="sess_mentions", worker_id="mention_worker")
        by_name = runtime.resolve_worker_mention("Mention Worker", session_id="sess_mentions")
        assert by_name.resolved is True
        assert by_name.worker_id == "mention_worker"

        await runtime.disable_worker_config("mention_worker")
        disabled = runtime.resolve_worker_mention("mention_worker", session_id="sess_mentions")
        assert disabled.resolved is False
        assert disabled.status == "disabled"
        assert disabled.error_code == "worker_disabled"

        ambiguous = runtime.resolve_worker_mention("Ambiguous Worker", session_id="sess_mentions")
        assert ambiguous.resolved is False
        assert ambiguous.status == "ambiguous"
        assert ambiguous.error_code == "worker_ambiguous"

        await runtime.soft_delete_worker_config("deleted_worker")
        deleted = runtime.resolve_worker_mention("deleted_worker", session_id="sess_mentions")
        assert deleted.resolved is False
        assert deleted.status == "deleted"
        assert deleted.error_code == "worker_deleted"


@pytest.mark.asyncio
async def test_runtime_start_persists_mentioned_worker_directive(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("done")

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        await runtime.create_worker_config({"worker_id": "directed_worker", "name": "Directed Worker", "system_prompt": "Do directed work."})
        handle = await runtime.start(
            "hello",
            session_id="sess_directive_metadata",
            mode="orchestrator",
            directives={"mentioned_worker": {"mention": "@directed_worker"}},
        )
        _events = [event async for event in handle.events()]
        assert handle.status.value == "completed"
        assert handle.directives["mentioned_worker"]["worker_id"] == "directed_worker"
        assert handle.directives["mentioned_worker"]["worker_agent_id"] == worker_agent_id_for_session(
            session_id="sess_directive_metadata", worker_id="directed_worker"
        )
        assert runtime.paths is not None
        conn = sqlite3.connect(runtime.paths.session_db_path)
        row = conn.execute("SELECT metadata_json FROM runs WHERE run_id=?", (handle.run_id,)).fetchone()
        conn.close()
        metadata = json.loads(row[0])
        assert metadata["directives"]["mentioned_worker"]["worker_id"] == "directed_worker"


@pytest.mark.asyncio
async def test_dispatch_worker_uses_mentioned_worker_and_rejects_other_worker(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), permission_callback=allow) as runtime:
        assert runtime.config and runtime.paths and runtime.artifacts
        await runtime.create_worker_config({"worker_id": "selected_worker", "system_prompt": "Selected."})
        await runtime.create_worker_config({"worker_id": "other_worker", "system_prompt": "Other."})
        directive = runtime.resolve_worker_mention("selected_worker", session_id="sess_dispatch_directive").to_directive()
        context = ToolExecutionContext(
            session_id="sess_dispatch_directive",
            run_id="run_dispatch_directive",
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=runtime.paths.project_dir,
            home_dir=runtime.paths.home_dir,
            config=runtime.config,
            artifact_manager=runtime.artifacts,
            permission_callback=allow,
            services={
                "task_service": runtime.task_service,
                "runtime": runtime,
                "agent_definitions": runtime.agent_definitions,
                "run_directives": RunDirectives(mentioned_worker=directive).model_dump(mode="json", exclude_none=True),
            },
        )
        await runtime.store.ensure_session(
            session_id="sess_dispatch_directive",
            cwd=str(project),
            root_agent_id="agent_orchestrator",
        )
        await runtime.store.ensure_agent(
            agent_id="agent_orchestrator",
            session_id="sess_dispatch_directive",
            agent_type="orchestrator",
            status="running",
        )
        await runtime.store.create_run(
            run_id="run_dispatch_directive",
            session_id="sess_dispatch_directive",
            agent_id="agent_orchestrator",
            status="running",
        )
        create = await runtime.tool_registry.execute(
            ToolCall(
                tool_call_id="create_task",
                name="agent.task_create",
                arguments={
                    "task_id": "task_directive",
                    "wal_name": "task_directive.wal.jsonl",
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}],
                },
            ),
            context,
        )
        assert not create.is_error
        result = await runtime.tool_registry.execute(
            ToolCall(
                tool_call_id="dispatch_selected",
                name="agent.dispatch_worker",
                arguments={"task_id": "task_directive", "instruction": "work", "allowed_step_ids": ["s1"]},
            ),
            context,
        )
        assert not result.is_error
        assert result.content[0].data["worker_id"] == "selected_worker"  # type: ignore[union-attr]

        rejected = await runtime.tool_registry.execute(
            ToolCall(
                tool_call_id="dispatch_other",
                name="agent.dispatch_worker",
                arguments={
                    "task_id": "task_directive",
                    "instruction": "work",
                    "worker_agent_id": "other_worker",
                },
            ),
            context,
        )
        assert rejected.is_error
        assert rejected.error and rejected.error.code.value == "worker_not_available"


@pytest.mark.asyncio
async def test_busy_specified_worker_enters_queue_then_runs(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    worker_blocked = asyncio.Event()
    release_worker = asyncio.Event()

    async def wait_for_release() -> None:
        worker_blocked.set()
        await release_worker.wait()

    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="claim1", name="agent.task_claim_step", arguments={"task_id": "task_queue", "step_id": "s1"})]
    )
    scripted_ollama.enqueue_text("first worker done", block=wait_for_release)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="claim2", name="agent.task_claim_step", arguments={"task_id": "task_queue", "step_id": "s2"})]
    )
    scripted_ollama.enqueue_text("second worker done")

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), permission_callback=allow) as runtime:
        assert runtime.config and runtime.paths and runtime.artifacts
        await runtime.create_worker_config({"worker_id": "queue_worker", "system_prompt": "Queue worker."})
        session_id = "sess_worker_queue"
        parent_run_id = "run_parent_queue"
        await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id="agent_orchestrator")
        await runtime.store.ensure_agent(agent_id="agent_orchestrator", session_id=session_id, agent_type="orchestrator", status="running")
        await runtime.store.create_run(run_id=parent_run_id, session_id=session_id, agent_id="agent_orchestrator", status="running")
        context = ToolExecutionContext(
            session_id=session_id,
            run_id=parent_run_id,
            agent_id="agent_orchestrator",
            agent_role="orchestrator",
            project_dir=runtime.paths.project_dir,
            home_dir=runtime.paths.home_dir,
            config=runtime.config,
            artifact_manager=runtime.artifacts,
            permission_callback=allow,
            services={
                "task_service": runtime.task_service,
                "runtime": runtime,
                "agent_definitions": runtime.agent_definitions,
            },
        )
        create = await runtime.tool_registry.execute(
            ToolCall(
                tool_call_id="create_queue_task",
                name="agent.task_create",
                arguments={
                    "task_id": "task_queue",
                    "wal_name": "task_queue.wal.jsonl",
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}, {"step_id": "s2", "title": "Step 2"}],
                },
            ),
            context,
        )
        assert not create.is_error

        first = asyncio.create_task(
            runtime.run_worker_agent(
                session_id=session_id,
                parent_run_id=parent_run_id,
                parent_agent_id="agent_orchestrator",
                task_id="task_queue",
                instruction="first",
                worker_agent_id="queue_worker",
                allowed_step_ids=["s1"],
            )
        )
        await asyncio.wait_for(worker_blocked.wait(), timeout=1)
        second = asyncio.create_task(
            runtime.run_worker_agent(
                session_id=session_id,
                parent_run_id=parent_run_id,
                parent_agent_id="agent_orchestrator",
                task_id="task_queue",
                instruction="second",
                worker_agent_id="queue_worker",
                allowed_step_ids=["s2"],
            )
        )
        for _ in range(20):
            if runtime.list_worker_queue("queue_worker"):
                break
            await asyncio.sleep(0.01)
        queue = runtime.list_worker_queue("queue_worker")
        assert len(queue) == 1
        assert queue[0].worker_id == "queue_worker"
        assert queue[0].status == "queued"

        release_worker.set()
        first_result = await asyncio.wait_for(first, timeout=2)
        second_result = await asyncio.wait_for(second, timeout=2)
        assert first_result["claimed_step_id"] == "s1"
        assert second_result["claimed_step_id"] == "s2"
        assert second_result["selection_reason"] == "queued_worker"
        assert runtime.list_worker_queue("queue_worker") == []


@pytest.mark.asyncio
async def test_worker_queue_limit_returns_worker_queue_full(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        await runtime.create_worker_config({"worker_id": "limited_worker", "system_prompt": "Limited worker."})
        worker = runtime.worker_runtime.select_worker(worker_agent_id="limited_worker", session_id="sess_queue_limit")
        runtime.worker_runtime.mark_busy(worker, task_id="task")
        runtime._worker_queue_limit = 1
        first = asyncio.create_task(
            runtime.run_worker_agent(
                session_id="sess_queue_limit",
                parent_run_id="run_queue_limit",
                parent_agent_id="agent_orchestrator",
                task_id="task",
                instruction="queued",
                worker_agent_id="limited_worker",
            )
        )
        for _ in range(20):
            if runtime.list_worker_queue("limited_worker"):
                break
            await asyncio.sleep(0.01)
        with pytest.raises(AgentCoreError) as exc:
            await runtime.run_worker_agent(
                session_id="sess_queue_limit",
                parent_run_id="run_queue_limit",
                parent_agent_id="agent_orchestrator",
                task_id="task",
                instruction="overflow",
                worker_agent_id="limited_worker",
            )
        assert exc.value.code.value == "worker_queue_full"
        queue_id = runtime.list_worker_queue("limited_worker")[0].queue_id
        assert runtime.cancel_worker_queue_item(queue_id) is True
        with pytest.raises(AgentCoreError) as cancelled:
            await first
        assert cancelled.value.code.value == "cancelled"
        runtime.worker_runtime.mark_idle(worker)
