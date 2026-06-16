from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from agent_core import AgentRuntime
from agent_core.errors.codes import ErrorCode
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


@pytest.mark.asyncio
async def test_runtime_compact_agent_writes_compaction_node(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 64\n\n",
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text("main")
    scripted_ollama.enqueue_text("compact summary")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("remember this context", session_id="sess_compact")
        _events = [event async for event in handle.events()]
        result = await runtime.run_compact_agent(session_id="sess_compact", reason="test")
        replay = await runtime.replay_session("sess_compact")
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    assert result["stale"] is False
    assert result["compaction_node_id"] is not None
    assert any(node.node_type == "compact_input" for node in replay.nodes)
    compaction_nodes = [node for node in replay.nodes if node.node_type == "compaction"]
    assert compaction_nodes
    assert compaction_nodes[-1].content[0].text == "compact summary"  # type: ignore[union-attr]
    assert any(event.event_type == "compact_started" and event.payload["reason"] == "test" for event in replay.events)
    compact_events = [event for event in replay.events if event.event_type == "compact_completed"]
    assert compact_events and compact_events[-1].payload["stale"] is False
    compact_requests = [request for request in scripted_ollama.requests if request.get("model") == "compact-model"]
    assert compact_requests
    assert compact_requests[-1].get("tools") is None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM agents WHERE agent_id=?", (result["compact_agent_id"],)).fetchone()
    assert row is not None
    assert row["agent_type"] == "fork"
    assert row["parent_agent_id"] is not None
    assert row["created_by_run_id"] is not None
    assert row["fork_from_node_id"] is not None
    metadata = json.loads(row["metadata_json"])
    assert metadata["purpose"] == "compact"
    assert metadata["agent_definition_id"] == "default_compact_agent"


@pytest.mark.asyncio
async def test_runtime_compact_agent_marks_stale_when_active_path_changes(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 64\n\n",
        encoding="utf-8",
    )
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()
    scripted_ollama.enqueue_text("main")
    scripted_ollama.enqueue_streaming_text(
        "stale compact summary",
        block_after_delta=lambda: release_compact.wait(),
        started=lambda: compact_started.set(),
    )
    scripted_ollama.enqueue_text("new active answer")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("remember this context", session_id="sess_stale_compact")
        _events = [event async for event in handle.events()]
        compact_task = asyncio.create_task(runtime.run_compact_agent(session_id="sess_stale_compact", reason="test_stale"))
        await asyncio.wait_for(compact_started.wait(), timeout=1)
        next_handle = await runtime.start("new branch", session_id="sess_stale_compact")
        _next_events = [event async for event in next_handle.events()]
        release_compact.set()
        result = await asyncio.wait_for(compact_task, timeout=1)
        replay = await runtime.replay_session("sess_stale_compact")

    assert result["stale"] is True
    assert result["compaction_node_id"] is None
    assert not any(node.node_type == "compaction" for node in replay.nodes)
    compact_events = [event for event in replay.events if event.event_type == "compact_completed"]
    assert compact_events and compact_events[-1].payload["stale"] is True


@pytest.mark.asyncio
async def test_runtime_auto_background_compact_after_completed_run(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("context_window = 8192", "context_window = 80")
    text = text.replace("max_output_tokens = 1024", "max_output_tokens = 16")
    text = text.replace(
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 120',
    )
    text += "\n[compact]\nenabled = true\nreserve_tokens = 0\nkeep_recent_tokens = 5\nauto_background = true\nrecovery_sync = true\nmodel_profile = \"compact\"\nmax_summary_tokens = 64\n\n"
    text += "[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 64\n\n"
    config_path.write_text(text, encoding="utf-8")
    scripted_ollama.enqueue_text("main")
    scripted_ollama.enqueue_text("background compact summary")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("x" * 400, session_id="sess_auto_compact")
        _events = [event async for event in handle.events()]
        for _ in range(20):
            replay = await runtime.replay_session("sess_auto_compact")
            if any(node.node_type == "compaction" for node in replay.nodes):
                break
            await asyncio.sleep(0.01)
        replay = await runtime.replay_session("sess_auto_compact")

    assert "compact_pending" in [event.event_type for event in replay.events]
    assert any(node.node_type == "compaction" for node in replay.nodes)


@pytest.mark.asyncio
async def test_compact_role_has_no_provider_visible_tools(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)

    async with _runtime(project, scripted_ollama) as runtime:
        compact_tools = runtime._effective_tools(agent_role="compact")
        context = runtime._worker_tool_context(
            session_id="sess_compact_tools",
            run_id="run_compact_tools",
            agent_id="agent_compact",
            parent_agent_id="agent_main",
            parent_run_id="run_main",
            worker_scope={},
            allowed_tool_names=set(),
        )
        context.agent_role = "compact"
        context.allowed_tool_names = set()
        context.effective_tool_definitions = {}
        blocked = []
        for name, arguments in [
            ("agent.plan_template", {"goal": "x"}),
            ("agent.task_get", {"task_id": "task1"}),
            ("internal.recall_memory", {"query": "pytest"}),
            ("agent.create_sub_agent", {"task": "child"}),
        ]:
            result = await runtime.tool_registry.execute(
                ToolCall(tool_call_id=f"call_{name.replace('.', '_')}", name=name, arguments=arguments),
                context,
            )
            blocked.append(result)

    assert compact_tools == []
    assert all(result.is_error for result in blocked)
    assert {result.error.code for result in blocked if result.error} == {ErrorCode.TOOL_NOT_AVAILABLE}
