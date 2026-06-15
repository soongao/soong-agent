from __future__ import annotations

import pytest
from pydantic import ValidationError

import agent_core
from agent_core import AgentRuntime, PermissionDecision, RuntimeEvent, ToolDefinition
from agent_core.errors import ErrorCode
from agent_core.tools.registry import ToolRegistry
from agent_core.types.permissions import PermissionDecisionKind
from agent_core.types.runtime import UserMessage
from agent_core.api.handles import RunHandle
from agent_core.events import EventStream, make_event
from agent_core.types.runtime import RunMode, RunStatus


def test_public_api_exports() -> None:
    assert AgentRuntime
    assert "AgentRuntime" in agent_core.__all__
    assert ToolDefinition
    assert RuntimeEvent


def test_pydantic_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        PermissionDecision(decision=PermissionDecisionKind.DENY, unexpected=True)  # type: ignore[call-arg]


def test_error_code_contract_subset() -> None:
    values = {item.value for item in ErrorCode}
    for expected in [
        "validation_error",
        "config_error",
        "permission_denied",
        "tool_not_available",
        "path_conflict",
        "write_outside_allowed_roots",
        "task_terminal",
        "memory_write_failed",
    ]:
        assert expected in values


def test_user_message_from_text() -> None:
    message = UserMessage.from_text("hello")
    assert message.content[0].text == "hello"  # type: ignore[index,union-attr]


@pytest.mark.asyncio
async def test_run_events_hide_debug_by_default() -> None:
    stream = EventStream()
    handle = RunHandle(
        run_id="run_test",
        session_id="sess_test",
        agent_id="agent_main",
        status=RunStatus.RUNNING,
        mode=RunMode.NORMAL,
        _runtime=None,  # type: ignore[arg-type]
        _stream=stream,
    )
    await stream.put(make_event(session_id="sess_test", event_type="debug_event", level="debug"))
    await stream.put(make_event(session_id="sess_test", event_type="info_event"))
    await stream.close()

    events = [event.event_type async for event in handle.events()]
    assert events == ["info_event"]


def test_tool_schema_subset_rejects_ref_and_combinators() -> None:
    registry = ToolRegistry()

    async def handler(_context, _args):
        return None

    with pytest.raises(ValueError, match="unsupported tool schema keyword"):
        registry.register_tool(
            ToolDefinition(
                name="bad.ref",
                description="bad",
                input_schema={"type": "object", "properties": {"x": {"$ref": "#/defs/X"}}},
                permission="readonly",
            ),
            handler,
        )


@pytest.mark.asyncio
async def test_tool_arguments_validated_before_handler(isolated_dirs) -> None:
    home, project = isolated_dirs
    from agent_core.artifacts import ArtifactManager
    from agent_core.config import load_runtime_config
    from agent_core.permissions import PermissionSessionCache
    from agent_core.tools.execution import ToolExecutionContext
    from tests.conftest import write_config

    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    registry = ToolRegistry()
    called = False

    async def handler(_context, _args):
        nonlocal called
        called = True
        return "ok"

    registry.register_tool(
        ToolDefinition(
            name="test.echo",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["text"],
            },
            permission="readonly",
        ),
        handler,
    )
    context = ToolExecutionContext(
        session_id="sess_schema",
        run_id="run_schema",
        agent_id="agent_main",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_cache=PermissionSessionCache(),
    )
    from agent_core.types.tools import ToolCall

    missing = await registry.execute(ToolCall(tool_call_id="c1", name="test.echo", arguments={}), context)
    wrong = await registry.execute(ToolCall(tool_call_id="c2", name="test.echo", arguments={"text": 1}), context)
    ok = await registry.execute(ToolCall(tool_call_id="c3", name="test.echo", arguments={"text": "hi", "count": 1}), context)
    extra = await registry.execute(ToolCall(tool_call_id="c4", name="test.echo", arguments={"text": "hi", "unknown": True}), context)
    assert missing.is_error and missing.error and missing.error.code.value == "validation_error"
    assert wrong.is_error and wrong.error and wrong.error.code.value == "validation_error"
    assert not ok.is_error
    assert extra.is_error and extra.error and extra.error.code.value == "validation_error"
    assert "unknown field" in extra.error.message
    assert called is True

    registry.register_tool(
        ToolDefinition(
            name="test.flex",
            description="flex",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": True,
            },
            permission="readonly",
        ),
        handler,
    )
    flex = await registry.execute(ToolCall(tool_call_id="c5", name="test.flex", arguments={"text": "hi", "unknown": True}), context)
    assert not flex.is_error
    with pytest.raises(ValueError, match="unsupported tool schema keyword"):
        registry.register_tool(
            ToolDefinition(
                name="bad.oneof",
                description="bad",
                input_schema={"type": "object", "oneOf": [{"type": "object"}]},
                permission="readonly",
            ),
            handler,
        )
