from __future__ import annotations

import pytest

from agent_core import AgentRuntime
from agent_core.api.handles import RunHandle
from agent_core.events import EventStream
from agent_core.types import RunMode, RunStatus, TextBlock
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


@pytest.mark.asyncio
async def test_list_branchable_nodes_returns_only_user_messages(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        assert runtime.store
        session_id = "sess_branchable"
        await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id="agent_main")
        first = await runtime.store.add_node(
            session_id=session_id,
            parent_id=None,
            agent_id="agent_main",
            run_id="run1",
            role="user",
            node_type="message",
            content=[TextBlock(text="first user message")],
            make_active=True,
        )
        assistant = await runtime.store.add_node(
            session_id=session_id,
            parent_id=first.node_id,
            agent_id="agent_main",
            run_id="run1",
            role="assistant",
            node_type="message",
            content=[TextBlock(text="assistant message")],
            make_active=True,
        )
        second = await runtime.store.add_node(
            session_id=session_id,
            parent_id=assistant.node_id,
            agent_id="agent_main",
            run_id="run2",
            role="user",
            node_type="message",
            content=[TextBlock(text="second user message")],
            make_active=True,
        )
        await runtime.store.add_node(
            session_id=session_id,
            parent_id=second.node_id,
            agent_id="agent_worker",
            run_id="run_worker",
            role="user",
            node_type="worker_dispatch",
            content=[TextBlock(text="worker dispatch")],
        )

        nodes = await runtime.list_branchable_nodes(session_id)
        assert [node.node_id for node in nodes] == [second.node_id, first.node_id]
        assert all(node.role == "user" and node.node_type == "message" for node in nodes)
        assert nodes[0].content_preview == "second user message"
        assert nodes[0].active is True


@pytest.mark.asyncio
async def test_branchable_node_can_switch_and_fork_session(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        assert runtime.store
        session_id = "sess_branch_switch"
        await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id="agent_main")
        first = await runtime.store.add_node(
            session_id=session_id,
            parent_id=None,
            agent_id="agent_main",
            run_id="run1",
            role="user",
            node_type="message",
            content=[TextBlock(text="fork here")],
            make_active=True,
        )
        second = await runtime.store.add_node(
            session_id=session_id,
            parent_id=first.node_id,
            agent_id="agent_main",
            run_id="run2",
            role="user",
            node_type="message",
            content=[TextBlock(text="current message")],
            make_active=True,
        )

        switched = await runtime.switch_node(session_id, first.node_id)
        assert switched.switched is True
        nodes = await runtime.list_branchable_nodes(session_id)
        active = next(node for node in nodes if node.node_id == first.node_id)
        inactive = next(node for node in nodes if node.node_id == second.node_id)
        assert active.active is True
        assert inactive.active is False

        forked = await runtime.fork_session(session_id, node_id=first.node_id, new_session_id="sess_branch_fork")
        assert forked.forked is True
        assert forked.source_node_id == first.node_id
        assert forked.session_id == "sess_branch_fork"
        copied = await runtime.list_branchable_nodes("sess_branch_fork")
        assert len(copied) == 1
        assert copied[0].content_preview == "fork here"


@pytest.mark.asyncio
async def test_branch_and_fork_reject_active_session(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    async with AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry()) as runtime:
        assert runtime.store
        session_id = "sess_branch_active"
        await runtime.store.ensure_session(session_id=session_id, cwd=str(project), root_agent_id="agent_main")
        node = await runtime.store.add_node(
            session_id=session_id,
            parent_id=None,
            agent_id="agent_main",
            run_id="run1",
            role="user",
            node_type="message",
            content=[TextBlock(text="active message")],
            make_active=True,
        )
        runtime._session_active[session_id] = RunHandle(
            run_id="run_active_branch",
            session_id=session_id,
            agent_id="agent_main",
            status=RunStatus.RUNNING,
            mode=RunMode.NORMAL,
            _runtime=runtime,
            _stream=EventStream(),
        )

        switched = await runtime.switch_node(session_id, node.node_id)
        assert switched.switched is False
        assert switched.error is not None
        assert switched.error.code.value == "session_active"

        forked = await runtime.fork_session(session_id, node_id=node.node_id, new_session_id="sess_branch_active_fork")
        assert forked.forked is False
        assert forked.error is not None
        assert forked.error.code.value == "session_active"
