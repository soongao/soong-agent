from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import time

import pytest

from agent_core import AgentRuntime
from agent_core.providers import ModelEvent, ModelRequest, ProviderAdapter, ProviderRegistry
from agent_core.providers.base import StopReason
from agent_core.types.content import TextBlock
from agent_core.types.runtime import RunStatus
from agent_core.types.tools import ToolCall, ToolDefinition
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider


class StopHookProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        text = "first final" if len(self.requests) == 1 else "second final"
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text=text)], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class MultiToolProvider(ProviderAdapter):
    def __init__(self, calls: list[ToolCall]) -> None:
        self.calls = calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        if len(self.requests) == 1:
            yield ModelEvent(event_type="model_completed", tool_calls=self.calls, stop_reason=StopReason.TOOL_USE)
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="done")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class SequentialMultiToolProvider(ProviderAdapter):
    def __init__(self, calls: list[ToolCall]) -> None:
        self.calls = calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        if len(self.requests) == 1:
            yield ModelEvent(event_type="model_completed", tool_calls=self.calls, stop_reason=StopReason.TOOL_USE)
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="done")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class TaskCreateProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        if len(self.requests) == 1:
            yield ModelEvent(
                event_type="model_completed",
                tool_calls=[
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
                ],
                stop_reason=StopReason.TOOL_USE,
            )
            return
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="done")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class NoToolsProvider(ProviderAdapter):
    supports_tools = False

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_completed", content=[TextBlock(text="should not run")], stop_reason=StopReason.END_TURN)

    async def close(self) -> None:
        return None


class FailingAfterDeltaProvider(ProviderAdapter):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        yield ModelEvent(event_type="model_text_delta", text_delta="partial text")
        from agent_core.types.common import ErrorPayload
        from agent_core.errors import ErrorCode

        yield ModelEvent(event_type="model_failed", error=ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="boom"))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_simple_run(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="hello")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("hi")
        events = [event.event_type async for event in handle.events()]
    assert "loop_started" in events
    assert "run_completed" in events
    assert "loop_completed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_provider_without_tool_support_fails_before_model_call(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    provider = NoToolsProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("hi")
        events = [event async for event in handle.events()]

    assert handle.status == RunStatus.FAILED
    assert provider.requests == []
    failed = [event for event in events if event.event_type == "loop_failed"]
    assert failed
    assert failed[-1].payload["code"] == "unsupported_capability"


@pytest.mark.asyncio
async def test_provider_failure_persists_partial_without_making_it_active(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    failing = FailingAfterDeltaProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: failing)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("will fail", session_id="sess_partial")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_partial")
        partial = [node for node in replay.nodes if node.role == "assistant" and node.metadata.get("partial")]
        assert handle.status == RunStatus.FAILED
        assert partial
        assert partial[-1].metadata["failed"] is True

    ok = FakeProvider(final_text="ok")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: ok)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("after fail", session_id="sess_partial")
        _events = [event async for event in handle.events()]

    texts = [
        getattr(block, "text", "")
        for message in ok.requests[-1].messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "will fail" in texts
    assert "after fail" in texts
    assert "partial text" not in texts


@pytest.mark.asyncio
async def test_runtime_tool_call(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    (project / "a.txt").write_text("hello", encoding="utf-8")
    fake = FakeProvider(
        final_text="read",
        tool_call=ToolCall(
            tool_call_id="call1",
            name="code.read_file",
            arguments={"path": "a.txt"},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]
    assert "tool_started" in events
    assert "tool_completed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_command_timeout_returns_tool_error_and_event_code(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)

    async def allow(_request):
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    provider = FakeProvider(
        final_text="handled timeout",
        tool_call=ToolCall(
            tool_call_id="call_timeout",
            name="code.run_command",
            arguments={"argv": ["python3", "-c", "import time; time.sleep(1)"], "timeout_ms": 50},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
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
async def test_write_permission_failure_stops_later_write_tool_calls(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)

    async def callback(_request):
        raise RuntimeError("permission UI unavailable")

    provider = SequentialMultiToolProvider(
        [
            ToolCall(tool_call_id="call1", name="code.write_file", arguments={"path": "first.txt", "content": "one"}),
            ToolCall(tool_call_id="call2", name="code.write_file", arguments={"path": "second.txt", "content": "two"}),
        ]
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=callback) as runtime:
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
    assert tool_results[0].tool_call_id == "call1"


@pytest.mark.asyncio
async def test_plan_template_tool_persists_synthetic_instruction_node(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(
        final_text="planned",
        tool_call=ToolCall(
            tool_call_id="call_plan",
            name="agent.plan_template",
            arguments={"goal": "build a feature"},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
    second_request = fake.requests[1]
    instruction_messages = [message for message in second_request.messages if message.node_type == "plan_instruction"]
    assert len(instruction_messages) == 1
    texts = [
        getattr(block, "text", "")
        for message in instruction_messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "Default Plan Template" in "\n".join(texts)
    assert "build a feature" in "\n".join(texts)


@pytest.mark.asyncio
async def test_load_skill_tool_persists_synthetic_context_node(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    skills = home / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    fake = FakeProvider(
        final_text="loaded",
        tool_call=ToolCall(
            tool_call_id="call_skill",
            name="internal.load_skill",
            arguments={"name": "review"},
        ),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
    second_request = fake.requests[1]
    skill_messages = [message for message in second_request.messages if message.node_type == "skill_context"]
    assert len(skill_messages) == 1
    text = "\n".join(
        getattr(block, "text", "")
        for message in skill_messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    assert '<skill name="review">' in text
    assert "Skill body" in text


@pytest.mark.asyncio
async def test_task_tool_refreshes_task_board_context_and_context_report(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    provider = TaskCreateProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async def allow(_request):
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("create task", session_id="sess_task_board")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_task_board")

    assert handle.status == RunStatus.COMPLETED
    wal_text = (project / ".soong-agent" / "tasks" / "sess_task_board" / "task1.wal.jsonl").read_text(encoding="utf-8")
    assert "task_running" in wal_text
    second_request = provider.requests[1]
    task_board_messages = [message for message in second_request.messages if message.node_type == "task_board"]
    assert len(task_board_messages) == 1
    text = "\n".join(
        getattr(block, "text", "")
        for message in task_board_messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    assert "<task_board>" in text
    assert "task1 [running]" in text
    assert "s1 [ready]" in text
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events
    assert context_events[-1].payload["estimated_input_tokens"] >= 0
    assert any(item["node_type"] == "task_board" for item in context_events[-1].payload["synthetic_messages"])


@pytest.mark.asyncio
async def test_next_run_includes_session_active_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = await runtime.start("first question", session_id="sess_history")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second question", session_id="sess_history")
        _second_events = [event async for event in second.events()]

    second_request = fake.requests[1]
    texts = [
        getattr(block, "text", "")
        for message in second_request.messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "first question" in texts
    assert "answer" in texts
    assert "second question" in texts


@pytest.mark.asyncio
async def test_context_budget_trims_old_messages_but_keeps_latest_user(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = write_config(home)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 12',
        ),
        encoding="utf-8",
    )
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = await runtime.start("first question has enough text to be trimmed", session_id="sess_trim")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second question", session_id="sess_trim")
        _second_events = [event async for event in second.events()]
        replay = await runtime.replay_session("sess_trim")

    second_request = fake.requests[1]
    texts = [
        getattr(block, "text", "")
        for message in second_request.messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "second question" in texts
    assert "first question has enough text to be trimmed" not in texts
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events[-1].payload["trimmed_node_ids"]
    assert context_events[-1].payload["non_system_budget"] == 12
    assert context_events[-1].payload["tokens_before_trim"] >= context_events[-1].payload["estimated_input_tokens"]


@pytest.mark.asyncio
async def test_prompt_too_long_fails_without_provider_call_when_latest_user_exceeds_budget(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = write_config(home)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
            'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 2',
        ),
        encoding="utf-8",
    )
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("this latest prompt is too long for the configured context budget", session_id="sess_prompt_too_long")
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_prompt_too_long")

    assert handle.status == RunStatus.FAILED
    assert fake.requests == []
    failed = [event for event in events if event.event_type == "loop_failed"]
    assert failed and failed[-1].payload["end_reason"] == "prompt_too_long"
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    assert context_events[-1].payload["too_long"] is True
    assert context_events[-1].payload["non_system_tokens_after_trim"] > context_events[-1].payload["non_system_budget"]


@pytest.mark.asyncio
async def test_dynamic_system_budget_trims_large_dynamic_blocks(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = write_config(home)
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
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("hi", session_id="sess_system_trim")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_system_trim")

    assert all(block.block_id != "memory.catalog" for block in fake.requests[0].system)
    context_events = [event for event in replay.events if event.event_type == "context_built"]
    trimmed = context_events[-1].payload["trimmed_system_blocks"]
    assert any(block["block_id"] == "memory.catalog" for block in trimmed)


@pytest.mark.asyncio
async def test_switch_node_changes_next_run_active_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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

    third_request = fake.requests[-1]
    texts = [
        getattr(block, "text", "")
        for message in third_request.messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "first branch" in texts
    assert "third from first" in texts
    assert "second branch" not in texts


@pytest.mark.asyncio
async def test_context_messages_use_latest_compaction_summary(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    fake = FakeProvider(final_text="answer")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = await runtime.start("old context", session_id="sess_compacted_context")
        _first_events = [event async for event in first.events()]
        await runtime.run_compact_agent(session_id="sess_compacted_context", reason="test")
        second = await runtime.start("new question", session_id="sess_compacted_context")
        _second_events = [event async for event in second.events()]

    second_request = fake.requests[-1]
    texts = [
        getattr(block, "text", "")
        for message in second_request.messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert any("<compaction" in text and "answer" in text for text in texts)
    assert "new question" in texts


@pytest.mark.asyncio
async def test_session_start_and_user_prompt_submit_hooks_observe_only(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
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
    fake = FakeProvider(final_text="ok")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)

    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
async def test_session_start_hook_only_runs_for_new_session(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
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
    fake = FakeProvider(final_text="ok")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        first = await runtime.start("first", session_id="sess_start_once")
        _first = [event async for event in first.events()]
        second = await runtime.start("second", session_id="sess_start_once")
        _second = [event async for event in second.events()]

    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [payload["event_type"] for payload in payloads] == ["SessionStart"]


@pytest.mark.asyncio
async def test_stop_hook_deny_prevents_completion_and_continues_next_turn(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
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
    provider = StopHookProvider()
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("finish")
        events = [event.event_type async for event in handle.events()]
        replay = await runtime.replay_session(handle.session_id)

    assert handle.status == RunStatus.FAILED
    assert "stop_hook_prevented" in events
    assert len(provider.requests) == 8
    assert any(node.node_type == "hook_context" for node in replay.nodes)


@pytest.mark.asyncio
async def test_tools_disabled_hidden_and_unavailable(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, disabled_tools=["code.read_file"])
    (project / "a.txt").write_text("hello", encoding="utf-8")
    fake = FakeProvider(
        final_text="done",
        tool_call=ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"}),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "tool_failed" in events
    assert "code.read_file" not in {tool.name for tool in fake.requests[0].tools}


@pytest.mark.asyncio
async def test_tool_override_changes_permission_and_tags(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(
        home,
        tool_overrides={"code.read_file": {"permission": "write", "tags": ["dangerous"], "description": "overridden read"}},
    )
    (project / "a.txt").write_text("hello", encoding="utf-8")
    fake = FakeProvider(
        final_text="done",
        tool_call=ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": "a.txt"}),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    permission_requests = []

    async def allow(request):
        permission_requests.append(request)
        from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind

        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
        handle = await runtime.start("read file")
        events = [event.event_type async for event in handle.events()]

    assert handle.status == RunStatus.COMPLETED
    assert "tool_completed" in events
    exposed = {tool.name: tool for tool in fake.requests[0].tools}
    assert exposed["code.read_file"].description == "overridden read"
    assert exposed["code.read_file"].permission == "write"
    assert "dangerous" in exposed["code.read_file"].tags
    assert permission_requests
    assert permission_requests[0].permission == "write"


@pytest.mark.asyncio
async def test_readonly_tool_calls_run_in_parallel(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    provider = MultiToolProvider(
        [
            ToolCall(tool_call_id="r1", name="test.read", arguments={"name": "a"}),
            ToolCall(tool_call_id="r2", name="test.read", arguments={"name": "b"}),
        ]
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    starts: list[tuple[str, float]] = []
    release = asyncio.Event()

    async def read_handler(_context, args):
        starts.append((args["name"], time.monotonic()))
        if len(starts) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return {"name": args["name"]}

    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
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
async def test_write_tool_failure_stops_following_tool_calls(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    provider = MultiToolProvider(
        [
            ToolCall(tool_call_id="w1", name="test.write", arguments={"name": "first"}),
            ToolCall(tool_call_id="r1", name="test.read_after", arguments={"name": "after"}),
        ]
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
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

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=allow) as runtime:
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
