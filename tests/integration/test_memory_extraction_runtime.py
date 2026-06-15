from __future__ import annotations

import json
import re

import pytest

from agent_core import AgentRuntime
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama, text_response


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


def _memory_response_with_source(memory_response: str):
    def responder(payload: dict, _index: int):
        source_text = str(payload["messages"][-1].get("content") or "")
        match = re.search(r'<node id="([^"]+)"', source_text)
        node_id = match.group(1) if match else "node_missing"
        return text_response(memory_response.replace("__NODE_ID__", node_id))

    return responder


@pytest.mark.asyncio
async def test_runtime_memory_extraction_uses_model_profile_and_advances_cursor(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[memory]
enabled = true
extract_every_messages = 1
extract_every_tokens = 12000
idle_seconds = 120
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
extract_model_profile = "memory_extract"

[model_overrides.memory_extract]
name = "memory-model"
max_output_tokens = 256
""",
        encoding="utf-8",
    )
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "user",
                    "filename": "prefs.md",
                    "summary": "Testing preference",
                    "tags": ["test"],
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "likes pytest",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("I like pytest", session_id="sess_memory")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory")
        metadata = await runtime.store.session_metadata("sess_memory")  # type: ignore[union-attr]

    memory_requests = [request for request in scripted_ollama.requests if request.get("model") == "memory-model"]
    assert memory_requests
    assert memory_requests[-1].get("tools") is None
    memory_file = home / "memory" / "user" / "prefs.md"
    assert memory_file.exists()
    memory_text = memory_file.read_text(encoding="utf-8")
    assert "likes pytest" in memory_text
    assert "source_node_ids:" in memory_text
    assert "source_session_id: sess_memory" in memory_text
    assert metadata["memory_scan_node_seq"] >= 1
    event_types = [event.event_type for event in replay.events]
    assert "memory_extraction_started" in event_types
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed
    assert completed[-1].payload["scan_cursor"]["node_seq"] == metadata["memory_scan_node_seq"]
    assert completed[-1].agent_id is None
    assert completed[-1].run_id is None


@pytest.mark.asyncio
async def test_runtime_memory_extraction_failure_does_not_advance_cursor(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[memory]
enabled = true
extract_every_messages = 1
extract_every_tokens = 12000
idle_seconds = 120
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
""",
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue_text(
        '{"memories":[{"decision":"new","category":"user","filename":"bad.md","summary":"Bad","content":"missing source"}]}'
    )

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("remember this", session_id="sess_memory_fail")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_fail")
        metadata = await runtime.store.session_metadata("sess_memory_fail")  # type: ignore[union-attr]

    assert "memory_scan_node_seq" not in metadata
    assert not (home / "memory" / "user" / "bad.md").exists()
    failed = [event for event in replay.events if event.event_type == "memory_extraction_failed"]
    assert failed
    assert failed[-1].agent_id is None
    assert failed[-1].run_id is None


@pytest.mark.asyncio
async def test_recall_memory_uses_selector_model_and_deduplicates_context(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[memory]
enabled = true
extract_every_messages = 99
extract_every_tokens = 12000
idle_seconds = 120
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
recall_model_profile = "memory_recall"

[model_overrides.memory_recall]
name = "recall-model"
max_output_tokens = 128
""",
        encoding="utf-8",
    )
    memory_dir = home / "memory" / "user"
    memory_dir.mkdir(parents=True)
    (memory_dir / "prefs.md").write_text(
        "---\nid: mem_prefs\ncategory: user\nsummary: Likes pytest\n---\nlikes pytest\n",
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text('{"selected_paths":["user/prefs.md"]}')

    from agent_core.types.tools import ToolCall

    async with _runtime(project, scripted_ollama) as runtime:
        await runtime._ensure_started()
        context = runtime._worker_tool_context(
            session_id="sess_recall",
            run_id="run_recall",
            agent_id="agent_main",
            parent_agent_id="agent_main",
            parent_run_id="run_parent",
            worker_scope={},
            allowed_tool_names={"internal.recall_memory"},
        )
        context.agent_role = "main"
        context.allowed_tool_names = {"internal.recall_memory"}
        context.effective_tool_definitions = {
            tool.name: tool for tool in runtime._effective_tools(agent_role="main") if tool.name == "internal.recall_memory"
        }
        first = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="m1", name="internal.recall_memory", arguments={"query": "pytest"}),
            context,
        )
        second = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="m2", name="internal.recall_memory", arguments={"query": "pytest"}),
            context,
        )

    recall_requests = [request for request in scripted_ollama.requests if request.get("model") == "recall-model"]
    assert recall_requests
    assert first.content[0].data["selected_by_model"] is True  # type: ignore[union-attr]
    assert first.content[0].data["matches"][0]["id"] == "mem_prefs"  # type: ignore[union-attr]
    assert second.content[0].data["already_recalled"] is True  # type: ignore[union-attr]
