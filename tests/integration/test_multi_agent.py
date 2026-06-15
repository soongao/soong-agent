from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio

import pytest

from agent_core import AgentRuntime
from agent_core.errors import AgentCoreError
from agent_core.providers import ModelEvent, ModelRequest, ProviderAdapter, ProviderRegistry
from agent_core.providers.base import StopReason
from agent_core.types.content import TextBlock
from agent_core.types.runtime import RunStatus
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider
from agent_core.types.agents import AgentDefinition


class ChildToolProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        run_id = str(request.metadata.get("run_id") or "")
        if run_id.startswith("run"):
            turn = sum(1 for item in self.requests if str(item.metadata.get("run_id") or "") == run_id)
            if turn == 1:
                yield ModelEvent(
                    event_type="model_completed",
                    tool_calls=[
                        ToolCall(
                            tool_call_id="child_read",
                            name="code.list_dir",
                            arguments={"path": ".", "recursive": False, "limit": 20},
                        )
                    ],
                    stop_reason=StopReason.TOOL_USE,
                )
                return
            yield ModelEvent(
                event_type="model_completed",
                content=[TextBlock(text='{"status":"ok"}')],
                stop_reason=StopReason.END_TURN,
            )
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="main done")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class BlockingChildProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.block = asyncio.Event()
        self.started = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        self.started.set()
        yield ModelEvent(event_type="model_started")
        await self.block.wait()
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="done")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        self.block.set()


@pytest.mark.asyncio
async def test_create_sub_agent_tool_runs_child(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(
        final_text="child done",
        tool_call=ToolCall(
            tool_call_id="call1",
            name="agent.create_sub_agent",
            arguments={"task": "do child work", "allowed_tools": ["code.read_file"]},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("delegate", session_id="sess_child")
        events = [event.event_type async for event in handle.events()]
        replay = await runtime.replay_session("sess_child")
    assert "tool_completed" in events
    assert handle.status == RunStatus.COMPLETED
    assert any(node.node_type == "child_result" for node in replay.nodes)


@pytest.mark.asyncio
async def test_create_sub_agent_invalid_allowed_tools_fails_tool(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(
        final_text="done",
        tool_call=ToolCall(
            tool_call_id="call1",
            name="agent.create_sub_agent",
            arguments={"task": "do child work", "allowed_tools": ["missing.tool"]},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("delegate")
        events = [event.event_type async for event in handle.events()]
    assert "tool_failed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_child_agent_effective_tools_are_restricted(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="child done")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        result = await runtime.run_child_agent(
            session_id="sess_direct_child",
            parent_run_id="run_parent",
            parent_agent_id="agent_parent",
            agent_definition_id="default_sub_agent",
            task="child",
            allowed_tools=None,
        )
    assert result["status"] == "completed"
    child_request = fake.requests[-1]
    names = {tool.name for tool in child_request.tools}
    assert "code.read_file" in names
    assert "agent.task_create" not in names
    assert "agent.task_get" not in names
    assert "agent.task_template" not in names
    assert "agent.create_sub_agent" not in names
    assert "agent.list_agent_definitions" not in names
    assert "internal.recall_memory" not in names


@pytest.mark.asyncio
async def test_child_allowed_tools_cannot_expand_effective_set(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="child done")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
async def test_child_agent_model_profile_can_select_different_provider(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    main_provider = FakeProvider(final_text="main")
    child_provider = FakeProvider(final_text="child-profile")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: main_provider)
    registry.register("other", lambda config: child_provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        runtime.register_agent_definition(
            AgentDefinition(
                agent_definition_id="profile_child",
                name="Profile Child",
                description="Uses another provider",
                body="child",
                model_profile={"provider": "other", "name": "other-model"},
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
    assert not main_provider.requests
    assert child_provider.requests[0].model == "other-model"


@pytest.mark.asyncio
async def test_child_agent_can_execute_tool_calls_and_validate_output_schema(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    provider = ChildToolProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
async def test_child_agent_respects_per_parent_limit(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config_path = home / "config.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n[agents]\nmax_children_per_run = 1\n", encoding="utf-8")
    provider = BlockingChildProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = asyncio.create_task(
            runtime.run_child_agent(
                session_id="sess_child_limit",
                parent_run_id="run_parent",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="first",
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        with pytest.raises(AgentCoreError) as exc_info:
            await runtime.run_child_agent(
                session_id="sess_child_limit",
                parent_run_id="run_parent",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="second",
            )
        provider.block.set()
        await first
    assert exc_info.value.code.value == "child_agent_limit_exceeded"


@pytest.mark.asyncio
async def test_child_agent_respects_per_session_limit(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config_path = home / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[agents]\nmax_children_per_run = 4\nmax_concurrent_children_per_session = 1\n",
        encoding="utf-8",
    )
    provider = BlockingChildProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = asyncio.create_task(
            runtime.run_child_agent(
                session_id="sess_child_session_limit",
                parent_run_id="run_parent_1",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="first",
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        with pytest.raises(AgentCoreError) as exc_info:
            await runtime.run_child_agent(
                session_id="sess_child_session_limit",
                parent_run_id="run_parent_2",
                parent_agent_id="agent_parent",
                agent_definition_id="default_sub_agent",
                task="second",
            )
        provider.block.set()
        await first
    assert exc_info.value.code.value == "child_agent_limit_exceeded"
