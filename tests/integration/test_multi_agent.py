from __future__ import annotations

import asyncio

import pytest

from agent_core import AgentRuntime
from agent_core.errors import AgentCoreError
from agent_core.types.agents import AgentDefinition
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
