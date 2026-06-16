from __future__ import annotations

import asyncio
import json
import re

import pytest

from agent_core import AgentRuntime
from agent_core.errors import AgentCoreError
from agent_core.api.runtime import _synthetic_context_nodes_from_tool_results
from agent_core.types.content import JsonBlock
from agent_core.types.tools import ToolResult
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


def test_repeated_memory_recall_result_does_not_create_synthetic_context_node() -> None:
    result = ToolResult(
        tool_call_id="m2",
        tool_name="internal.recall_memory",
        content=[
            JsonBlock(
                data={
                    "node_type": "memory_context",
                    "already_recalled": True,
                    "content": "<memory>likes pytest</memory>",
                }
            )
        ],
    )

    assert _synthetic_context_nodes_from_tool_results([result]) == []


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
    assert memory_requests[-1]["format"]["properties"]["memories"]["items"]["properties"]["category"]["enum"] == [
        "user",
        "feedback",
        "reference",
    ]
    memory_system_prompt = memory_requests[-1]["messages"][0]["content"]
    assert '"user|feedback|reference"' not in memory_system_prompt
    assert '"user", "feedback", or "reference"' in memory_system_prompt
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
async def test_memory_extraction_explicit_intent_triggers_immediately(
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
""",
        encoding="utf-8",
    )
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "user",
                    "filename": "explicit.md",
                    "summary": "Explicit preference",
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "prefers terse answers",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("记住：我喜欢简洁回答", session_id="sess_memory_explicit")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_explicit")

    assert (home / "memory" / "user" / "explicit.md").exists()
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed[-1].payload["reason"] == "explicit"


@pytest.mark.asyncio
async def test_memory_extraction_plain_message_waits_for_idle(
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
""",
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text("main done")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("ordinary status update", session_id="sess_memory_wait")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_wait")

    assert [request for request in scripted_ollama.requests if request.get("metadata", {}).get("purpose") == "memory_extraction"] == []
    assert not [event for event in replay.events if event.event_type.startswith("memory_extraction_")]


@pytest.mark.asyncio
async def test_memory_extraction_idle_triggers_after_delay(
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
idle_seconds = 0
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
""",
        encoding="utf-8",
    )
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "user",
                    "filename": "idle.md",
                    "summary": "Idle memory",
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "idle extracted memory",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("ordinary idle input", session_id="sess_memory_idle")
        _events = [event async for event in handle.events()]
        for _ in range(20):
            if (home / "memory" / "user" / "idle.md").exists():
                break
            await asyncio.sleep(0.01)
        replay = await runtime.replay_session("sess_memory_idle")

    assert (home / "memory" / "user" / "idle.md").exists()
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed[-1].payload["reason"] == "idle"


@pytest.mark.asyncio
async def test_memory_extraction_token_backlog_triggers(
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
extract_every_tokens = 2
idle_seconds = 120
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
""",
        encoding="utf-8",
    )
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "reference",
                    "filename": "token.md",
                    "summary": "Token backlog",
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "token backlog extracted memory",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("ordinary long input with enough characters", session_id="sess_memory_token")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_token")

    assert (home / "memory" / "reference" / "token.md").exists()
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed[-1].payload["reason"] == "token_backlog"


@pytest.mark.asyncio
async def test_runtime_memory_extraction_uses_configured_user_memory_dir(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[memory]
enabled = true
memory_dir = "${SOONG_AGENT_HOME}/memory-alt"
extract_every_messages = 1
extract_every_tokens = 12000
idle_seconds = 120
catalog_max_tokens = 4000
recall_top_k = 5
memory_context_token_budget = 6000
""",
        encoding="utf-8",
    )
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "user",
                    "filename": "alt.md",
                    "summary": "Alt memory",
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "stored in alternate user memory dir",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("Remember alternate dir", session_id="sess_memory_alt")
        _events = [event async for event in handle.events()]

    assert (home / "memory-alt" / "user" / "alt.md").exists()
    assert (home / "memory-alt" / "MEMORY.md").exists()
    assert not (home / "memory" / "user" / "alt.md").exists()


@pytest.mark.asyncio
async def test_runtime_rejects_project_memory_dir(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[memory]
enabled = true
memory_dir = "<project>/.soong-agent/memory"
""",
        encoding="utf-8",
    )

    with pytest.raises(AgentCoreError):
        async with _runtime(project, scripted_ollama) as runtime:
            await runtime._ensure_started()

    assert not (project / ".soong-agent" / "memory").exists()


@pytest.mark.asyncio
async def test_memory_extraction_does_not_consume_child_concurrency(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """

[agents]
max_concurrent_children_per_session = 0

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
    memory_response = json.dumps(
        {
            "memories": [
                {
                    "decision": "new",
                    "category": "user",
                    "filename": "child-limit.md",
                    "summary": "Child limit check",
                    "tags": ["test"],
                    "source_node_ids": ["__NODE_ID__"],
                    "content": "memory extraction is not a child agent",
                }
            ]
        }
    )
    scripted_ollama.enqueue_text("main done")
    scripted_ollama.enqueue(_memory_response_with_source(memory_response))

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("Remember that extraction is independent", session_id="sess_memory_child_limit")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_child_limit")

    assert (home / "memory" / "user" / "child-limit.md").exists()
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed
    assert completed[-1].agent_id is None
    assert completed[-1].run_id is None
    assert all(event.event_type != "child_agent_limit_exceeded" for event in replay.events)


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
async def test_memory_extraction_failed_event_redacts_sensitive_message(
    isolated_dirs, monkeypatch, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    monkeypatch.setenv("SOONG_MEMORY_TEST_SECRET", "memory-secret-value")
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
        json.dumps(
            {
                "memories": [
                    {
                        "decision": "new",
                        "category": "user",
                        "filename": "leak.md",
                        "summary": "Leak",
                        "content": "should not be written",
                        "source_node_ids": ["api_key=memory-secret-value"],
                    }
                ]
            }
        )
    )

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("remember this", session_id="sess_memory_fail_redact")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_memory_fail_redact", include_sensitive=True)

    failed = [event for event in replay.events if event.event_type == "memory_extraction_failed"]
    assert failed
    assert "memory-secret-value" not in failed[-1].payload["message"]
    assert "[REDACTED]" in failed[-1].payload["message"]


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


@pytest.mark.asyncio
async def test_recall_memory_selector_failure_returns_tool_error_not_run_failure(
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
""",
        encoding="utf-8",
    )
    memory_dir = home / "memory" / "user"
    memory_dir.mkdir(parents=True)
    (memory_dir / "prefs.md").write_text(
        "---\nid: mem_prefs\ncategory: user\nsummary: Likes pytest\n---\nlikes pytest\n",
        encoding="utf-8",
    )
    scripted_ollama.enqueue_failure_after_delta("selector partial")

    from agent_core.types.tools import ToolCall

    async with _runtime(project, scripted_ollama) as runtime:
        await runtime._ensure_started()
        context = runtime._worker_tool_context(
            session_id="sess_recall_fail",
            run_id="run_recall_fail",
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
        result = await runtime.tool_registry.execute(
            ToolCall(tool_call_id="m1", name="internal.recall_memory", arguments={"query": "pytest"}),
            context,
        )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code.value in {"memory_recall_failed", "provider_error"}
