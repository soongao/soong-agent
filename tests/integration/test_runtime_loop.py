from __future__ import annotations

import asyncio
import json
import time

import pytest

from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from agent_core.providers.ollama import OllamaProvider
from agent_core.types.content import TextBlock
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.runtime import RunStatus
from agent_core.types.tools import ToolCall, ToolDefinition
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama, ScriptedOllamaResponse


class NoToolsOllamaProvider(OllamaProvider):
    supports_tools = False


class RecordingProvider(OllamaProvider):
    def __init__(self, config, scripted_ollama: ScriptedOllama, seen_configs: list) -> None:
        seen_configs.append(config)
        super().__init__(config)
        self.base_url = scripted_ollama.base_url
        self._client = scripted_ollama.provider_registry().create("ollama", config)._client


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


def _payload_texts(payload: dict) -> list[str]:
    return [str(message.get("content") or "") for message in payload.get("messages", []) if message.get("content")]


def _payload_tool_names(payload: dict) -> set[str]:
    return {tool["function"]["name"].replace("__", ".") for tool in payload.get("tools", [])}


def _payload_system_text(payload: dict) -> str:
    return "\n".join(str(message.get("content") or "") for message in payload.get("messages", []) if message.get("role") == "system")


@pytest.mark.asyncio
async def test_runtime_simple_run(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi")
        events = [event.event_type async for event in handle.events()]
    assert "loop_started" in events
    assert "run_completed" in events
    assert "loop_completed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_events_is_single_consumer(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi")
        first_events = [event.event_type async for event in handle.events()]
        with pytest.raises(RuntimeError, match="single-consumer"):
            _second_events = [event async for event in handle.events()]

    assert "loop_completed" in first_events


@pytest.mark.asyncio
async def test_model_text_delta_is_realtime_only_not_persisted(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_realtime_delta")
        events = [event async for event in handle.events(debug=True)]
        replay = await runtime.replay_session("sess_realtime_delta")

    assert any(event.event_type == "model_text_delta" and event.seq is None for event in events)
    assert all(event.event_type != "model_text_delta" for event in replay.events)
    assert any(event.event_type == "model_completed" for event in replay.events)


@pytest.mark.asyncio
async def test_runtime_register_provider_uses_custom_factory_config(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, provider="custom", base_url=scripted_ollama.base_url, model_name="custom-model")
    scripted_ollama.enqueue_text("custom provider ok")
    seen_configs = []

    runtime = AgentRuntime(project_dir=project)
    runtime.register_provider(
        "custom",
        lambda config: RecordingProvider(config, scripted_ollama, seen_configs),
    )
    async with runtime:
        handle = await runtime.start("hi", session_id="sess_custom_provider")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "loop_completed" in events
    assert len(seen_configs) == 1
    assert seen_configs[0].provider == "custom"
    assert seen_configs[0].name == "custom-model"
    assert scripted_ollama.requests[0]["model"] == "custom-model"


@pytest.mark.asyncio
async def test_provider_without_tool_support_fails_before_model_call(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    registry = ProviderRegistry()
    registry.register("ollama", lambda config: NoToolsOllamaProvider(config))
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("hi")
        events = [event async for event in handle.events()]

    assert handle.status == RunStatus.FAILED
    assert scripted_ollama.requests == []
    failed = [event for event in events if event.event_type == "loop_failed"]
    assert failed
    assert failed[-1].payload["code"] == "unsupported_capability"


@pytest.mark.asyncio
async def test_provider_failure_persists_partial_without_making_it_active(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_failure_after_delta("partial text")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("will fail", session_id="sess_partial")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_partial")
        partial = [node for node in replay.nodes if node.role == "assistant" and node.metadata.get("partial")]
        assert handle.status == RunStatus.FAILED
        assert partial
        assert partial[-1].metadata["failed"] is True

    scripted_ollama.enqueue_text("ok")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("after fail", session_id="sess_partial")
        _events = [event async for event in handle.events()]

    texts = _payload_texts(scripted_ollama.requests[-1])
    assert "will fail" in texts
    assert "after fail" in texts
    assert "partial text" not in texts


@pytest.mark.asyncio
async def test_runtime_tool_call(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    (project / "a.txt").write_text("hello", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"})]
    )
    scripted_ollama.enqueue_text("read")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]
    assert "tool_started" in events
    assert "tool_completed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_empty_model_response_after_tool_result_is_retried_once(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    (project / "a.txt").write_text("hello", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"})]
    )
    scripted_ollama.enqueue(ScriptedOllamaResponse(lines=[{"message": {}, "done": True, "prompt_eval_count": 1, "eval_count": 0}]))
    scripted_ollama.enqueue_text("read after retry")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("read file", session_id="sess_empty_tool_retry")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_empty_tool_retry")

    assert handle.status == RunStatus.COMPLETED
    assert "model_empty_response_recovered" in [event.event_type for event in events]
    assert len(scripted_ollama.requests) == 3
    assert any(
        getattr(block, "text", "") == "read after retry"
        for node in replay.nodes
        if node.role == "assistant"
        for block in node.content
    )


@pytest.mark.asyncio
async def test_loaded_instruction_context_is_scoped_to_session(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    instruction_dir = project / "pkg"
    instruction_dir.mkdir()
    (instruction_dir / "CLAUDE.md").write_text("session scoped instruction body\n", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="read_instruction", name="code.read_file", arguments={"path": "pkg/CLAUDE.md"})]
    )
    scripted_ollama.enqueue_text("loaded instruction")
    scripted_ollama.enqueue_text("other session")

    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("load instructions", session_id="sess_instruction_a")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("separate session", session_id="sess_instruction_b")
        _second_events = [event async for event in second.events()]

    second_system = _payload_system_text(scripted_ollama.requests[2])
    assert "session scoped instruction body" not in second_system
    assert "session scoped instruction body" not in "\n".join(_payload_texts(scripted_ollama.requests[2]))


@pytest.mark.asyncio
async def test_root_claude_md_is_auto_loaded_without_reading_linked_rules(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    rules_dir = project / "rules"
    rules_dir.mkdir()
    (project / "CLAUDE.md").write_text("project entry instruction\nsee rules/detail.md\n", encoding="utf-8")
    (rules_dir / "detail.md").write_text("linked rule should not auto load\n", encoding="utf-8")
    scripted_ollama.enqueue_text("ok")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("check instructions", session_id="sess_auto_claude")
        _events = [event async for event in handle.events()]

    system_text = _payload_system_text(scripted_ollama.requests[0])
    assert "project entry instruction" in system_text
    assert "linked rule should not auto load" not in system_text


@pytest.mark.asyncio
async def test_run_command_timeout_returns_tool_error_and_event_code(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)

    async def allow(_request):
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="call_timeout",
                name="code.run_command",
                arguments={"argv": ["python3", "-c", "import time; time.sleep(1)"], "timeout_ms": 50},
            )
        ]
    )
    scripted_ollama.enqueue_text("handled timeout")
    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        handle = await runtime.start("run slow command", session_id="sess_timeout")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_timeout")

    failed = [event for event in events if event.event_type == "tool_failed"]
    assert handle.status == RunStatus.COMPLETED
    assert failed
    assert failed[-1].payload["error"]["code"] == "timeout"
    tool_results = [
        block
        for node in replay.nodes
        if node.role == "tool"
        for block in node.content
        if getattr(block, "type", None) == "tool_result"
    ]
    assert tool_results[0].is_error is True
    assert tool_results[0].error and tool_results[0].error.code.value == "timeout"


@pytest.mark.asyncio
async def test_write_permission_failure_stops_later_write_tool_calls(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)

    async def callback(_request):
        raise RuntimeError("permission UI unavailable")

    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(tool_call_id="call1", name="code.write_file", arguments={"path": "first.txt", "content": "one"}),
            ToolCall(tool_call_id="call2", name="code.write_file", arguments={"path": "second.txt", "content": "two"}),
        ]
    )
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=callback) as runtime:
        handle = await runtime.start("write files", session_id="sess_permission_failure")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_permission_failure")

    assert handle.status == RunStatus.COMPLETED
    assert "permission_failed" in [event.event_type for event in events]
    assert "tool_started" in [event.event_type for event in events]
    assert not (project / "first.txt").exists()
    assert not (project / "second.txt").exists()
    tool_started_names = [event.payload["name"] for event in events if event.event_type == "tool_started"]
    assert tool_started_names == ["code.write_file"]
    tool_results = [
        block
        for node in replay.nodes
        if node.role == "tool"
        for block in node.content
        if getattr(block, "type", None) == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "ollama_tool_0"


@pytest.mark.asyncio
async def test_allow_for_session_permission_cache_is_not_persisted(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    requests = []

    async def allow_for_session(request):
        requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_FOR_SESSION)

    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="write_first", name="code.write_file", arguments={"path": "cached.txt", "content": "one"})]
    )
    scripted_ollama.enqueue_text("first done")
    async with _runtime(project, scripted_ollama, permission_callback=allow_for_session) as runtime:
        first = await runtime.start("write once", session_id="sess_permission_cache")
        _first_events = [event async for event in first.events()]

    assert len(requests) == 1

    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="write_second",
                name="code.write_file",
                arguments={"path": "cached.txt", "content": "two", "overwrite": True},
            )
        ]
    )
    scripted_ollama.enqueue_text("second done")
    async with _runtime(project, scripted_ollama, permission_callback=allow_for_session) as runtime:
        second = await runtime.start("write after restart", session_id="sess_permission_cache")
        _second_events = [event async for event in second.events()]

    assert len(requests) == 2
    assert requests[0].target_scope == requests[1].target_scope == str((project / "cached.txt").resolve())
    assert (project / "cached.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_plan_template_tool_persists_synthetic_instruction_node(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_plan", name="agent.plan_template", arguments={"goal": "build a feature"})]
    )
    scripted_ollama.enqueue_text("planned")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("make a plan", session_id="sess_plan_instruction")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_plan_instruction")

    assert handle.status == RunStatus.COMPLETED
    instruction_nodes = [node for node in replay.nodes if node.node_type == "plan_instruction"]
    assert len(instruction_nodes) == 1
    assert instruction_nodes[0].role == "user"
    assert instruction_nodes[0].metadata["synthetic"] is True
    assert instruction_nodes[0].metadata["source"] == "agent.plan_template"
    assert instruction_nodes[0].metadata["goal"] == "build a feature"
    assert instruction_nodes[0].metadata["template_id"] == "template.plan.default"
    assert instruction_nodes[0].metadata["template_version"] == "1"
    texts = _payload_texts(scripted_ollama.requests[1])
    assert any("Default Plan Template" in text for text in texts)
    assert any("build a feature" in text for text in texts)


@pytest.mark.asyncio
async def test_plan_template_then_write_plan_file_uses_write_permission(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    permission_requests = []

    async def allow(request):
        permission_requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_plan", name="agent.plan_template", arguments={"goal": "build a feature"})]
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="write_plan",
                name="code.write_file",
                arguments={"path": ".soong-agent/plans/plan.md", "content": "# Plan\n\n- build"},
            )
        ]
    )
    scripted_ollama.enqueue_text("plan written")
    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        handle = await runtime.start("make and write a plan", session_id="sess_plan_write")
        events = [event.event_type async for event in handle.events()]

    plan_path = project / ".soong-agent" / "plans" / "plan.md"
    assert handle.status == RunStatus.COMPLETED
    assert "tool_completed" in events
    assert plan_path.read_text(encoding="utf-8") == "# Plan\n\n- build"
    assert len(permission_requests) == 1
    assert permission_requests[0].tool_name == "code.write_file"
    assert permission_requests[0].target_scope == str(plan_path.resolve())


@pytest.mark.asyncio
async def test_load_skill_tool_persists_synthetic_context_node(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    skill_dir = home / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_skill", name="internal.load_skill", arguments={"name": "review"})]
    )
    scripted_ollama.enqueue_text("loaded")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("use review skill", session_id="sess_skill_context")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_skill_context")

    assert handle.status == RunStatus.COMPLETED
    context_nodes = [node for node in replay.nodes if node.node_type == "skill_context"]
    assert len(context_nodes) == 1
    assert context_nodes[0].role == "user"
    assert context_nodes[0].metadata["synthetic"] is True
    assert context_nodes[0].metadata["source"] == "internal.load_skill"
    assert context_nodes[0].metadata["name"] == "review"
    text = "\n".join(_payload_texts(scripted_ollama.requests[1]))
    assert '<skill name="review">' in text
    assert "Skill body" in text


@pytest.mark.asyncio
async def test_runtime_explicit_load_skill_persists_session_context(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    skills = home / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    scripted_ollama.enqueue_text("used skill")
    async with _runtime(project, scripted_ollama) as runtime:
        catalog = await runtime.list_skills()
        result = await runtime.load_skill("sess_explicit_skill", "review")
        repeated = await runtime.load_skill("sess_explicit_skill", "review")
        handle = await runtime.start("now use it", session_id="sess_explicit_skill")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_explicit_skill")

    assert [skill.name for skill in catalog] == ["review"]
    assert result.loaded is True
    assert result.already_loaded is False
    assert result.node_id
    assert repeated.loaded is True
    assert repeated.already_loaded is True
    context_nodes = [node for node in replay.nodes if node.node_type == "skill_context"]
    assert len(context_nodes) == 1
    assert context_nodes[0].metadata["source"] == "runtime.load_skill"
    assert context_nodes[0].metadata["name"] == "review"
    text = "\n".join(_payload_texts(scripted_ollama.requests[0]))
    assert '<skill name="review">' in text
    assert "Skill body" in text


@pytest.mark.asyncio
async def test_repeated_load_skill_does_not_duplicate_synthetic_context_node(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    skills = home / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_skill_1", name="internal.load_skill", arguments={"name": "review"})]
    )
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call_skill_2", name="internal.load_skill", arguments={"name": "review"})]
    )
    scripted_ollama.enqueue_text("loaded once")

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("load review twice", session_id="sess_skill_duplicate")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_skill_duplicate")

    assert handle.status == RunStatus.COMPLETED
    context_nodes = [node for node in replay.nodes if node.node_type == "skill_context"]
    assert len(context_nodes) == 1


@pytest.mark.asyncio
async def test_task_tool_refreshes_task_board_context_and_context_report(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama, worker_pool=True)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="task_create",
                name="agent.task_create",
                arguments={
                    "task_id": "task1",
                    "wal_name": "task1.wal.jsonl",
                    "title": "Task",
                    "summary": "Task summary",
                    "steps": [
                        {"step_id": "s1", "title": "Step 1", "summary": "Do first"},
                        {"step_id": "s2", "title": "Step 2", "depends_on_step_ids": ["s1"]},
                    ],
                },
            )
        ]
    )
    scripted_ollama.enqueue_text("done")

    async def allow(_request):
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        handle = await runtime.start("create task", session_id="sess_task_board", mode="orchestrator")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_task_board")

    assert handle.status == RunStatus.COMPLETED
    wal_text = (project / ".soong-agent" / "tasks" / "sess_task_board" / "task1.wal.jsonl").read_text(encoding="utf-8")
    assert "task_running" in wal_text
    text = "\n".join(_payload_texts(scripted_ollama.requests[1]))
    assert "<task_board>" in text
    assert "task1 [running]" in text
    assert "s1 [ready]" in text
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events
    assert context_events[-1].payload["estimated_input_tokens"] >= 0
    assert any(item["node_type"] == "task_board" for item in context_events[-1].payload["synthetic_messages"])


@pytest.mark.asyncio
async def test_next_run_includes_session_active_path(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first question", session_id="sess_history")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second question", session_id="sess_history")
        _second_events = [event async for event in second.events()]

    texts = _payload_texts(scripted_ollama.requests[1])
    assert "first question" in texts
    assert "answer" in texts
    assert "second question" in texts


@pytest.mark.asyncio
async def test_context_budget_trims_old_messages_but_keeps_latest_user(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 12',
        ),
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first question has enough text to be trimmed", session_id="sess_trim")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second question", session_id="sess_trim")
        _second_events = [event async for event in second.events()]
        replay = await runtime.replay_session("sess_trim")

    texts = _payload_texts(scripted_ollama.requests[1])
    assert "second question" in texts
    assert "first question has enough text to be trimmed" not in texts
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events[-1].payload["trimmed_node_ids"]
    assert context_events[-1].payload["non_system_budget"] == 12
    assert context_events[-1].payload["tokens_before_trim"] >= context_events[-1].payload["estimated_input_tokens"]


@pytest.mark.asyncio
async def test_prompt_too_long_fails_without_provider_call_when_latest_user_exceeds_budget(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 2',
        ),
        encoding="utf-8",
    )
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("this latest prompt is too long for the configured context budget", session_id="sess_prompt_too_long")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_prompt_too_long")

    assert handle.status == RunStatus.FAILED
    assert scripted_ollama.requests == []
    failed = [event for event in events if event.event_type == "loop_failed"]
    assert failed and failed[-1].payload["end_reason"] == "prompt_too_long"
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events[-1].payload["too_long"] is True
    assert context_events[-1].payload["non_system_tokens_after_trim"] > context_events[-1].payload["non_system_budget"]


@pytest.mark.asyncio
async def test_recovery_compact_runs_when_history_exceeds_budget(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 24',
    )
    text += "\n[compact]\nenabled = true\nauto_background = false\nrecovery_sync = true\nmodel_profile = \"compact\"\nmax_summary_tokens = 32\n\n"
    text += "[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 32\n\n"
    config_path.write_text(text, encoding="utf-8")
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("old context", session_id="sess_recovery_compact")
        _first_events = [event async for event in first.events()]
        assert runtime.store
        active_node_id = await runtime.store.active_node_id("sess_recovery_compact")
        await runtime.store.add_node(
            session_id="sess_recovery_compact",
            parent_id=active_node_id,
            agent_id="agent_main",
            run_id=first.run_id,
            role="user",
            node_type="plan_instruction",
            content=[TextBlock(text="protected synthetic context " * 30)],
            metadata={"synthetic": True, "source": "test"},
            make_active=True,
        )
        second = await runtime.start("new question", session_id="sess_recovery_compact")
        second_events = [event async for event in second.events()]
        replay = await runtime.replay_session("sess_recovery_compact")

    assert second.status == RunStatus.COMPLETED
    assert "loop_completed" in [event.event_type for event in second_events]
    assert any(event.event_type == "compact_pending" and event.payload.get("reason") == "recovery_sync" for event in replay.events)
    assert any(node.node_type == "compaction" for node in replay.nodes)
    recovery_context_events = [
        event for event in replay.events if event.event_type == "context_built" and event.payload.get("recovery_compact")
    ]
    assert recovery_context_events
    compact_requests = [request for request in scripted_ollama.requests if request.get("model") == "compact-model"]
    assert compact_requests
    texts = _payload_texts(scripted_ollama.requests[-1])
    assert any("<compaction" in text and "answer" in text for text in texts)
    assert "new question" in texts


@pytest.mark.asyncio
async def test_dynamic_system_budget_trims_large_dynamic_blocks(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama, memory_enabled=True)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\ndynamic_system_budget = 4',
        ),
        encoding="utf-8",
    )
    memory_dir = home / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("large memory catalog " * 20, encoding="utf-8")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_system_trim")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_system_trim")

    assert "large memory catalog" not in _payload_system_text(scripted_ollama.requests[0])
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    trimmed = context_events[-1].payload["trimmed_system_blocks"]
    assert any(block["block_id"] == "memory.catalog" for block in trimmed)


@pytest.mark.asyncio
async def test_switch_node_changes_next_run_active_path(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first branch", session_id="sess_switch")
        _first_events = [event async for event in first.events()]
        replay = await runtime.replay_session("sess_switch")
        first_user = next(node for node in replay.nodes if node.role == "user")
        second = await runtime.start("second branch", session_id="sess_switch")
        _second_events = [event async for event in second.events()]
        result = await runtime.switch_node("sess_switch", first_user.node_id)
        assert result.switched is True
        third = await runtime.start("third from first", session_id="sess_switch")
        _third_events = [event async for event in third.events()]

    texts = _payload_texts(scripted_ollama.requests[-1])
    assert "first branch" in texts
    assert "third from first" in texts
    assert "second branch" not in texts


@pytest.mark.asyncio
async def test_list_and_fork_session_from_active_path(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("first answer")
    scripted_ollama.enqueue_text("second answer")
    scripted_ollama.enqueue_text("fork answer")

    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first path", session_id="sess_fork_source")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second path", session_id="sess_fork_source")
        _second_events = [event async for event in second.events()]

        sessions = await runtime.list_sessions(limit=5)
        assert any(session.session_id == "sess_fork_source" for session in sessions)

        nodes = await runtime.list_session_nodes("sess_fork_source", limit=10)
        assert any(node.active and "second answer" in node.content_preview for node in nodes)

        result = await runtime.fork_session("sess_fork_source", mode="normal")
        assert result.forked is True
        assert result.session_id is not None
        assert result.copied_nodes >= 4

        forked_replay = await runtime.replay_session(result.session_id)
        assert forked_replay.nodes
        assert all(node.run_id is None for node in forked_replay.nodes)
        forked = await runtime.start("continue fork", session_id=result.session_id)
        _forked_events = [event async for event in forked.events()]

    texts = _payload_texts(scripted_ollama.requests[-1])
    assert "first path" in texts
    assert "second path" in texts
    assert "continue fork" in texts


@pytest.mark.asyncio
async def test_context_messages_use_latest_compaction_summary(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    scripted_ollama.enqueue_text("answer")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("old context", session_id="sess_compacted_context")
        _first_events = [event async for event in first.events()]
        await runtime.run_compact_agent(session_id="sess_compacted_context", reason="test")
        second = await runtime.start("new question", session_id="sess_compacted_context")
        _second_events = [event async for event in second.events()]

    texts = _payload_texts(scripted_ollama.requests[-1])
    assert any("<compaction" in text and "answer" in text for text in texts)
    assert "new question" in texts


@pytest.mark.asyncio
async def test_session_start_and_user_prompt_submit_hooks_observe_only(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("ok")
    log_path = home / "hook_events.jsonl"
    script = home / "observe_hook.py"
    script.write_text(
        "import json, pathlib, sys\n"
        f"path=pathlib.Path({str(log_path)!r})\n"
        "payload=json.load(sys.stdin)\n"
        "with path.open('a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(payload, ensure_ascii=False)+'\\n')\n"
        "print(json.dumps({'decision':'deny','reason':'observe only'}))\n",
        encoding="utf-8",
    )
    (home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [{"matcher": {}, "hooks": [{"type": "command", "command": ["python3", str(script)]}]}],
                    "UserPromptSubmit": [{"matcher": {}, "hooks": [{"type": "command", "command": ["python3", str(script)]}]}],
                }
            }
        ),
        encoding="utf-8",
    )

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hello hook", session_id="sess_hooks")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_hooks")

    assert handle.status == RunStatus.COMPLETED
    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [payload["event_type"] for payload in payloads] == ["SessionStart", "UserPromptSubmit"]
    assert payloads[1]["prompt"] == "hello hook"
    hook_events = [event for event in replay.events if event.event_type.endswith("_hook_observed")]
    assert {event.event_type for event in hook_events} == {
        "session_started_hook_observed",
        "user_prompt_submitted_hook_observed",
    }


@pytest.mark.asyncio
async def test_session_start_hook_only_runs_for_new_session(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("ok")
    scripted_ollama.enqueue_text("ok")
    log_path = home / "hook_events.jsonl"
    script = home / "observe_hook.py"
    script.write_text(
        "import json, pathlib, sys\n"
        f"path=pathlib.Path({str(log_path)!r})\n"
        "payload=json.load(sys.stdin)\n"
        "with path.open('a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(payload, ensure_ascii=False)+'\\n')\n"
        "print(json.dumps({'decision':'allow'}))\n",
        encoding="utf-8",
    )
    (home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [{"matcher": {}, "hooks": [{"type": "command", "command": ["python3", str(script)]}]}],
                }
            }
        ),
        encoding="utf-8",
    )
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first", session_id="sess_start_once")
        _first = [event async for event in first.events()]
        second = await runtime.start("second", session_id="sess_start_once")
        _second = [event async for event in second.events()]

    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [payload["event_type"] for payload in payloads] == ["SessionStart"]


@pytest.mark.asyncio
async def test_stop_hook_deny_prevents_completion_and_continues_next_turn(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("max_turns = 128", "max_turns = 8"),
        encoding="utf-8",
    )
    for index in range(8):
        scripted_ollama.enqueue_text("first final" if index == 0 else "second final")
    (home / "hooks.json").write_text(
        """
{
  "hooks": {
    "Stop": [
      {
        "matcher": {},
        "hooks": [
          {"decision": "deny", "reason": "final answer missing check"}
        ]
      }
    ]
  }
}
""".strip(),
        encoding="utf-8",
    )
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("finish")
        events = [event.event_type async for event in handle.events()]
        replay = await runtime.replay_session(handle.session_id)

    assert handle.status == RunStatus.FAILED
    assert "stop_hook_prevented" in events
    assert len(scripted_ollama.requests) == 8
    assert any(node.node_type == "hook_context" for node in replay.nodes)


@pytest.mark.asyncio
async def test_tools_disabled_hidden_and_unavailable(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama, disabled_tools=["code.read_file"])
    (project / "a.txt").write_text("hello", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"})]
    )
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "tool_failed" in events
    assert "code.read_file" not in _payload_tool_names(scripted_ollama.requests[0])


@pytest.mark.asyncio
async def test_tool_override_changes_permission_and_tags(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(
        home,
        scripted_ollama,
        tool_overrides={"code.read_file": {"permission": "write", "tags": ["dangerous"], "description": "overridden read"}},
    )
    (project / "a.txt").write_text("hello", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"})]
    )
    scripted_ollama.enqueue_text("done")
    permission_requests = []

    async def allow(request):
        permission_requests.append(request)
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "tool_completed" in events
    exposed = {tool["function"]["name"].replace("__", "."): tool for tool in scripted_ollama.requests[0]["tools"]}
    assert exposed["code.read_file"]["function"]["description"] == "overridden read"
    assert permission_requests
    assert permission_requests[0].permission == "write"
    assert "dangerous" in permission_requests[0].tags


@pytest.mark.asyncio
async def test_readonly_tool_calls_run_in_parallel(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(tool_call_id="r1", name="test.read", arguments={"name": "a"}),
            ToolCall(tool_call_id="r2", name="test.read", arguments={"name": "b"}),
        ]
    )
    scripted_ollama.enqueue_text("done")
    starts: list[tuple[str, float]] = []
    release = asyncio.Event()

    async def read_handler(_context, args):
        starts.append((args["name"], time.monotonic()))
        if len(starts) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return {"name": args["name"]}

    async with _runtime(project, scripted_ollama) as runtime:
        runtime.register_tool(
            ToolDefinition(
                name="test.read",
                description="test read",
                permission="readonly",
                tags={"readonly"},
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            ),
            read_handler,
        )
        handle = await runtime.start("run reads")
        events = [event async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    event_types = [event.event_type for event in events]
    assert event_types.count("tool_completed") == 2
    assert {name for name, _ts in starts} == {"a", "b"}
    assert abs(starts[0][1] - starts[1][1]) < 0.5


@pytest.mark.asyncio
async def test_write_tool_failure_stops_following_tool_calls(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(tool_call_id="w1", name="test.write", arguments={"name": "first"}),
            ToolCall(tool_call_id="r1", name="test.read_after", arguments={"name": "after"}),
        ]
    )
    scripted_ollama.enqueue_text("done")
    calls: list[str] = []

    async def write_handler(_context, args):
        calls.append(args["name"])
        raise RuntimeError("write failed")

    async def read_handler(_context, args):
        calls.append(args["name"])
        return {"name": args["name"]}

    async def allow(_request):
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with _runtime(project, scripted_ollama, permission_callback=allow) as runtime:
        runtime.register_tool(
            ToolDefinition(
                name="test.write",
                description="test write",
                permission="write",
                tags={"write"},
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            ),
            write_handler,
        )
        runtime.register_tool(
            ToolDefinition(
                name="test.read_after",
                description="test read after",
                permission="readonly",
                tags={"readonly"},
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            ),
            read_handler,
        )
        handle = await runtime.start("run write then read")
        events = [event async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert calls == ["first"]
    event_types = [event.event_type for event in events]
    assert event_types.count("tool_failed") == 1
    assert "after" not in calls
