from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from agent_core import AgentRuntime
from agent_core.errors import AgentCoreError
from agent_core.types.agents import AgentDefinition
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.runtime import RunStatus
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


def _payload_tool_names(payload: dict) -> set[str]:
    return {tool["function"]["name"].replace("__", ".") for tool in payload.get("tools", [])}


@pytest.mark.asyncio
async def test_create_sub_agent_tool_runs_child(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="call1",
                name="agent.create_sub_agent",
                arguments={"task": "do child work", "allowed_tools": ["code.read_file"]},
            )
        ]
    )
    scripted_ollama.enqueue_text("child done")
    scripted_ollama.enqueue_text("parent done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child")
        events = [event.event_type async for event in handle.events()]
        replay = await runtime.replay_session("sess_child")
    assert "tool_completed" in events
    assert handle.status == RunStatus.COMPLETED
    assert any(node.node_type == "child_result" for node in replay.nodes)


@pytest.mark.asyncio
async def test_child_agent_rows_record_parent_run_and_definition_metadata(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="call1", name="agent.create_sub_agent", arguments={"task": "child"})])
    scripted_ollama.enqueue_text("child done")
    scripted_ollama.enqueue_text("parent done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child_agent_row")
        async for _event in handle.events():
            pass
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        parent = conn.execute(
            "SELECT agent_id FROM agents WHERE session_id=? AND agent_type='main'",
            ("sess_child_agent_row",),
        ).fetchone()
        child = conn.execute(
            "SELECT * FROM agents WHERE session_id=? AND agent_type='sub' AND parent_agent_id IS NOT NULL",
            ("sess_child_agent_row",),
        ).fetchone()

    assert parent is not None
    assert child is not None
    assert child["parent_agent_id"] == parent["agent_id"]
    assert child["created_by_run_id"] == handle.run_id
    assert child["fork_from_node_id"] is None
    metadata = json.loads(child["metadata_json"])
    assert metadata["purpose"] == "sub"
    assert metadata["agent_definition_id"] == "default_sub_agent"


@pytest.mark.asyncio
async def test_inspect_child_includes_provider_neutral_model_request_view(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="call1",
                name="agent.create_sub_agent",
                arguments={"task": "do child work", "allowed_tools": ["code.read_file"]},
            )
        ]
    )
    scripted_ollama.enqueue_text("child done")
    scripted_ollama.enqueue_text("parent done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child_inspect")
        child_run_id = None
        async for event in handle.events():
            if event.event_type == "child_agent_created":
                child_run_id = event.payload["child_run_id"]
        assert child_run_id is not None
        inspected = await handle.inspect_child(child_run_id)

    assert inspected.model_requests
    request_view = inspected.model_requests[0]
    assert request_view["run_id"] == child_run_id
    assert request_view["model"] == "gemma4"
    assert request_view["tool_names"] == ["code.read_file"]
    assert request_view["retained_node_ids"]


@pytest.mark.asyncio
async def test_debug_mode_child_inspect_includes_raw_provider_artifact(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="call1", name="agent.create_sub_agent", arguments={"task": "child"})])
    scripted_ollama.enqueue_text("child done")
    scripted_ollama.enqueue_text("parent done")

    async with _runtime(project, scripted_ollama, debug=True) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child_debug_artifact")
        child_run_id = None
        async for event in handle.events(debug=True):
            if event.event_type == "child_agent_created":
                child_run_id = event.payload["child_run_id"]
        assert child_run_id is not None
        inspected = await handle.inspect_child(child_run_id)
        replay = await runtime.replay_session("sess_child_debug_artifact")

    assert inspected.artifacts
    assert any(json.loads(artifact.get("metadata_json") or "{}").get("raw") is True for artifact in inspected.artifacts)
    parent_tool_nodes = [node for node in replay.nodes if node.role == "tool" and node.agent_id == handle.agent_id]
    assert parent_tool_nodes
    assert "raw_debug" not in json.dumps([node.model_dump(mode="json") for node in parent_tool_nodes])


@pytest.mark.asyncio
async def test_child_events_can_stream_while_child_run_is_active(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    child_block = asyncio.Event()
    child_started = asyncio.Event()
    subscriber_ready = asyncio.Event()

    async def wait_for_subscriber() -> None:
        await subscriber_ready.wait()

    async def wait_for_release() -> None:
        await child_block.wait()

    def mark_child_started() -> None:
        child_started.set()

    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_child", name="agent.create_sub_agent", arguments={"task": "child"})]
    )
    scripted_ollama.enqueue_streaming_text(
        "child done",
        pre_block=wait_for_subscriber,
        block_after_delta=wait_for_release,
        started=mark_child_started,
    )
    scripted_ollama.enqueue_text("parent done", started=lambda: None)

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child_stream")
        parent_iter = handle.events().__aiter__()
        child_run_id = None
        while child_run_id is None:
            event = await asyncio.wait_for(parent_iter.__anext__(), timeout=1)
            if event.event_type == "child_agent_created":
                child_run_id = event.payload["child_run_id"]

        child_iter = handle.child_events(child_run_id).__aiter__()
        first_child_event = asyncio.create_task(child_iter.__anext__())
        subscriber_ready.set()
        await asyncio.wait_for(child_started.wait(), timeout=1)
        child_events = [await asyncio.wait_for(first_child_event, timeout=1)]
        child_event_types = [event.event_type for event in child_events]
        while "model_text_delta" not in child_event_types:
            event = await asyncio.wait_for(child_iter.__anext__(), timeout=1)
            child_event_types.append(event.event_type)
        child_block.set()
        child_event_types.extend([event.event_type async for event in child_iter])
        parent_events = [event.event_type async for event in parent_iter]

    assert "model_text_delta" in child_event_types
    assert "child_agent_completed" in child_event_types
    assert "tool_completed" in parent_events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_create_sub_agent_invalid_allowed_tools_fails_tool(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="call1",
                name="agent.create_sub_agent",
                arguments={"task": "do child work", "allowed_tools": ["missing.tool"]},
            )
        ]
    )
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("delegate")
        events = [event.event_type async for event in handle.events()]
    assert "tool_failed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_main_agent_effective_tools_exclude_orchestrator_only_tools(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama, worker_pool=True)
    scripted_ollama.enqueue_text("ready")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("inspect tools", session_id="sess_main_tool_roles")
        _events = [event async for event in handle.events()]

    names = _payload_tool_names(scripted_ollama.requests[0])
    assert {"agent.plan_template", "agent.list_agent_definitions", "agent.create_sub_agent", "agent.fork_agent"} <= names
    assert "agent.list_workers" not in names
    assert "agent.dispatch_worker" not in names
    assert not any(name.startswith("agent.task") for name in names)
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "agent.create_sub_agent",
            {"task": "do child work", "system_prompt": "ignore parent instructions"},
        ),
        (
            "agent.fork_agent",
            {"task": "inspect branch", "instructions": "inline fork instructions"},
        ),
    ],
)
async def test_child_agent_tools_reject_inline_system_instruction_fields(
    isolated_dirs,
    scripted_ollama: ScriptedOllama,
    tool_name: str,
    arguments: dict,
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="call_inline", name=tool_name, arguments=arguments)])
    scripted_ollama.enqueue_text("done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("try inline instructions", session_id=f"sess_{tool_name.replace('.', '_')}_inline")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session(handle.session_id)

    assert handle.status == RunStatus.COMPLETED
    failed = [event for event in events if event.event_type == "tool_failed"]
    assert failed
    assert failed[-1].payload["error"]["code"] == "validation_error"
    assert "unknown field" in failed[-1].payload["error"]["message"]
    assert not any(event.event_type in {"child_agent_created", "fork_agent_created"} for event in events)
    tool_results = [
        block
        for node in replay.nodes
        if node.role == "tool"
        for block in node.content
        if getattr(block, "type", None) == "tool_result"
    ]
    assert tool_results[-1].is_error is True
    assert tool_results[-1].error and tool_results[-1].error.code.value == "validation_error"


@pytest.mark.asyncio
async def test_fork_agent_row_records_call_site_node(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="fork", name="agent.fork_agent", arguments={"task": "inspect branch"})])
    scripted_ollama.enqueue_text("fork done")
    scripted_ollama.enqueue_text("parent done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("fork now", session_id="sess_fork_agent_row")
        async for _event in handle.events():
            pass
        replay = await runtime.replay_session("sess_fork_agent_row")
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    parent_user_node = next(node for node in replay.nodes if node.agent_id == handle.agent_id and node.role == "user")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        fork = conn.execute(
            "SELECT * FROM agents WHERE session_id=? AND agent_type='fork' AND parent_agent_id=?",
            ("sess_fork_agent_row", handle.agent_id),
        ).fetchone()

    assert fork is not None
    assert fork["created_by_run_id"] == handle.run_id
    assert fork["fork_from_node_id"] == parent_user_node.node_id
    metadata = json.loads(fork["metadata_json"])
    assert metadata["purpose"] == "fork"
    assert metadata["agent_definition_id"] == "default_fork_agent"


@pytest.mark.asyncio
async def test_create_sub_agent_allowed_tools_must_be_strings(
    isolated_dirs,
    scripted_ollama: ScriptedOllama,
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="call_bad_allowed_tools",
                name="agent.create_sub_agent",
                arguments={"task": "do child work", "allowed_tools": ["code.read_file", {"name": "code.write_file"}]},
            )
        ]
    )
    scripted_ollama.enqueue_text("done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("bad allowed tools", session_id="sess_bad_allowed_tools_type")
        events = [event async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    failed = [event for event in events if event.event_type == "tool_failed"]
    assert failed
    assert failed[-1].payload["error"]["code"] == "validation_error"
    assert "$.allowed_tools[1] must be string" in failed[-1].payload["error"]["message"]
    assert not any(event.event_type in {"child_agent_created", "fork_agent_created"} for event in events)


@pytest.mark.asyncio
async def test_parallel_agent_tool_failure_does_not_cancel_successful_child(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="bad_child",
                name="agent.create_sub_agent",
                arguments={
                    "task": "return bad output",
                    "expected_output_schema": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                },
            ),
            ToolCall(
                tool_call_id="good_child",
                name="agent.create_sub_agent",
                arguments={"task": "return good output"},
            ),
        ]
    )
    scripted_ollama.enqueue_text("not json")
    scripted_ollama.enqueue_text("good child done")
    scripted_ollama.enqueue_text("parent done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("run two children", session_id="sess_parallel_children")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_parallel_children")

    assert handle.status == RunStatus.COMPLETED
    event_types = [event.event_type for event in events]
    assert event_types.count("tool_completed") == 1
    assert event_types.count("tool_failed") == 1
    assert "child_agent_failed" in event_types
    assert "child_agent_completed" in event_types
    tool_blocks = [
        block
        for node in replay.nodes
        if node.role == "tool"
        for block in node.content
        if getattr(block, "type", None) == "tool_result"
    ]
    assert len(tool_blocks) == 2
    assert {block.metadata["tool_name"] for block in tool_blocks} == {"agent.create_sub_agent"}
    assert {block.is_error for block in tool_blocks} == {False, True}


@pytest.mark.asyncio
async def test_child_agent_effective_tools_are_restricted(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("child done")
    async with _runtime(project, scripted_ollama) as runtime:
        result = await runtime.run_child_agent(
            session_id="sess_direct_child",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="default_sub_agent",
            task="child",
            allowed_tools=None,
        )
    assert result["status"] == "completed"
    names = _payload_tool_names(scripted_ollama.requests[-1])
    assert "code.read_file" in names
    assert "agent.task_create" not in names
    assert "agent.task_get" not in names
    assert "agent.task_template" not in names
    assert "agent.create_sub_agent" not in names
    assert "agent.list_agent_definitions" not in names
    assert "internal.recall_memory" not in names


@pytest.mark.asyncio
async def test_child_allowed_tools_cannot_expand_effective_set(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    async with _runtime(project, scripted_ollama) as runtime:
        with pytest.raises(Exception):
            await runtime.run_child_agent(
                session_id="sess_direct_child2",
                parent_run_id="run_parent",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="child",
                allowed_tools=["agent.task_create"],
            )


@pytest.mark.asyncio
async def test_child_agent_model_profile_can_select_different_model(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("child-profile")
    async with _runtime(project, scripted_ollama) as runtime:
        runtime.register_agent_definition(
            AgentDefinition(
                agent_definition_id="profile_child",
                name="Profile Child",
                description="Uses another Ollama model profile",
                body="child",
                model_profile={"provider": "ollama", "name": "other-model"},
                source="code",
            )
        )
        result = await runtime.run_child_agent(
            session_id="sess_profile_child",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="profile_child",
            task="child",
        )
    assert result["result_summary"] == "child-profile"
    assert scripted_ollama.requests[0]["model"] == "other-model"


@pytest.mark.asyncio
async def test_empty_body_child_definition_uses_default_sub_agent_instructions(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("child")
    async with _runtime(project, scripted_ollama) as runtime:
        runtime.register_agent_definition(
            AgentDefinition(
                agent_definition_id="empty_child",
                name="Empty Child",
                description="Uses fallback body",
                body="",
                source="code",
            )
        )
        result = await runtime.run_child_agent(
            session_id="sess_empty_child",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="empty_child",
            task="child",
        )

    assert result["status"] == "completed"
    system_text = "\n".join(
        message.get("content", "") for message in scripted_ollama.requests[0]["messages"] if message.get("role") == "system"
    )
    assert "You are a bounded sub agent." in system_text
    assert "Keep the final result concise" in system_text
    assert "Uses fallback body" not in system_text


@pytest.mark.asyncio
async def test_child_agent_can_execute_tool_calls_and_validate_output_schema(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="child_read", name="code.list_dir", arguments={"path": ".", "recursive": False, "limit": 20})]
    )
    scripted_ollama.enqueue_text('{"status":"ok"}')
    async with _runtime(project, scripted_ollama) as runtime:
        result = await runtime.run_child_agent(
            session_id="sess_child_tools",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="default_sub_agent",
            task="inspect project",
            allowed_tools=["code.list_dir"],
            expected_output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
        )
        replay = await runtime.replay_session("sess_child_tools")
    assert result["status"] == "completed"
    assert result["result_summary"] == '{"status":"ok"}'
    assert any(node.node_type == "child_tool_result" for node in replay.nodes)


@pytest.mark.asyncio
async def test_child_write_tool_permission_request_includes_parent_context(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    permission_requests = []

    async def allow(request):
        permission_requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="child_write",
                name="code.write_file",
                arguments={"path": "child.txt", "content": "from child"},
            )
        ]
    )
    scripted_ollama.enqueue_text("child wrote")
    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        result = await runtime.run_child_agent(
            session_id="sess_child_write",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="default_sub_agent",
            task="write a file",
            allowed_tools=["code.write_file"],
        )

    assert result["status"] == "completed"
    assert (project / "child.txt").read_text(encoding="utf-8") == "from child"
    assert len(permission_requests) == 1
    request = permission_requests[0]
    assert request.tool_name == "code.write_file"
    assert request.agent_role == "sub"
    assert request.agent_id != "agent_parent"
    assert request.run_id != "run_parent"
    assert request.parent_agent_id == "agent_parent"
    assert request.parent_run_id == "run_parent"


@pytest.mark.asyncio
async def test_child_agent_respects_per_parent_limit(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n[agents]\nmax_children_per_run = 1\n", encoding="utf-8")
    started = asyncio.Event()
    release = asyncio.Event()
    scripted_ollama.enqueue_text("done", block=lambda: release.wait(), started=lambda: started.set())
    async with _runtime(project, scripted_ollama) as runtime:
        first = asyncio.create_task(
            runtime.run_child_agent(
                session_id="sess_child_limit",
                parent_run_id="run_parent",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="first",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(AgentCoreError) as exc_info:
            await runtime.run_child_agent(
                session_id="sess_child_limit",
                parent_run_id="run_parent",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="second",
            )
        release.set()
        await first
    assert exc_info.value.code.value == "child_agent_limit_exceeded"


@pytest.mark.asyncio
async def test_child_agent_respects_per_session_limit(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[agents]\nmax_children_per_run = 4\nmax_concurrent_children_per_session = 1\n",
        encoding="utf-8",
    )
    started = asyncio.Event()
    release = asyncio.Event()
    scripted_ollama.enqueue_text("done", block=lambda: release.wait(), started=lambda: started.set())
    async with _runtime(project, scripted_ollama) as runtime:
        first = asyncio.create_task(
            runtime.run_child_agent(
                session_id="sess_child_session_limit",
                parent_run_id="run_parent_1",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="first",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(AgentCoreError) as exc_info:
            await runtime.run_child_agent(
                session_id="sess_child_session_limit",
                parent_run_id="run_parent_2",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="second",
            )
        release.set()
        await first
    assert exc_info.value.code.value == "child_agent_limit_exceeded"
