from __future__ import annotations

import pytest
from pydantic import ValidationError

import agent_core
from agent_core import AgentDefinition, AgentRuntime, PermissionDecision, PermissionRequest, RuntimeEvent, ToolDefinition, ToolResult
from agent_core.errors import ErrorCode
from agent_core.tools.registry import ToolRegistry
from agent_core.types.content import TextBlock
from agent_core.types.permissions import PermissionDecisionKind
from agent_core.types.runtime import UserMessage
from agent_core.api.handles import RunHandle
from agent_core.events import EventStream, make_event
from agent_core.types.runtime import RunMode, RunStatus


def test_public_api_exports() -> None:
    assert set(agent_core.__all__) == {
        "AgentDefinition",
        "AgentRuntime",
        "PermissionDecision",
        "PermissionRequest",
        "RunHandle",
        "RuntimeEvent",
        "ToolDefinition",
        "ToolResult",
        "UserMessage",
    }
    for name in agent_core.__all__:
        assert getattr(agent_core, name)


def test_pydantic_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        PermissionDecision(decision=PermissionDecisionKind.DENY, unexpected=True)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UserMessage(content=[TextBlock(text="hello")], unexpected=True),
        lambda: RuntimeEvent(
            event_id="evt_test",
            session_id="sess_test",
            event_type="test",
            created_at="2024-01-01T00:00:00Z",
            unexpected=True,
        ),
        lambda: PermissionRequest(
            request_id="perm_test",
            session_id="sess_test",
            agent_id="agent_test",
            run_id="run_test",
            agent_role="main",
            tool_name="code.read_file",
            permission="readonly",
            args_summary="{}",
            cwd="/tmp",
            unexpected=True,
        ),
        lambda: PermissionDecision(decision=PermissionDecisionKind.DENY, unexpected=True),
        lambda: ToolDefinition(
            name="test.tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
            permission="readonly",
            unexpected=True,
        ),
        lambda: ToolResult(tool_call_id="call_test", tool_name="test.tool", unexpected=True),
        lambda: AgentDefinition(
            agent_definition_id="agent_test",
            name="Agent",
            description="desc",
            source="code",
            unexpected=True,
        ),
    ],
)
def test_public_boundary_types_forbid_extra_fields(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_error_payload_default_boundary_fields() -> None:
    from agent_core.types.common import ErrorPayload

    payload = ErrorPayload(code=ErrorCode.INTERNAL_ERROR, message="boom")
    assert payload.type == "error"
    assert payload.retryable is False
    assert payload.details == {}
    assert payload.redacted is True


def test_agent_core_error_carries_retryable_details_and_cause() -> None:
    from agent_core.errors import AgentCoreError

    cause = RuntimeError("root")
    error = AgentCoreError(ErrorCode.PROVIDER_TIMEOUT, "timeout", retryable=True, details={"attempt": 1}, cause=cause)

    assert error.code == ErrorCode.PROVIDER_TIMEOUT
    assert error.retryable is True
    assert error.details == {"attempt": 1}
    assert error.cause is cause


def test_error_code_contract_subset() -> None:
    values = {item.value for item in ErrorCode}
    expected_values = {
        "validation_error",
        "schema_error",
        "config_error",
        "permission_denied",
        "tool_not_available",
        "timeout",
        "cancelled",
        "internal_error",
        "provider_error",
        "unsupported_capability",
        "provider_auth_failed",
        "provider_rate_limited",
        "provider_timeout",
        "storage_error",
        "migration_failed",
        "session_active",
        "path_conflict",
        "file_not_found",
        "text_not_found",
        "ambiguous_edit",
        "patch_apply_failed",
        "patch_path_mismatch",
        "write_outside_allowed_roots",
        "invalid_agent_definition",
        "duplicate_agent_definition",
        "invalid_agent_override",
        "child_agent_limit_exceeded",
        "worker_busy",
        "worker_queue_full",
        "worker_not_available",
        "worker_pool_busy",
        "task_not_found",
        "task_terminal",
        "task_not_dispatchable",
        "dependency_cycle",
        "step_not_found",
        "step_not_ready",
        "step_already_claimed",
        "step_already_claimed_by_run",
        "step_has_dependents",
        "no_step_claimed",
        "task_wal_unavailable",
        "memory_recall_failed",
        "memory_write_failed",
        "skill_not_found",
        "skill_load_failed",
    }
    assert values == expected_values


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
