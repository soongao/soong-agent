from __future__ import annotations

from typing import Any, Literal

from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.tools import _run_scheduled_tool_calls
from agent_core.api.runtime_helpers.views import _summary_from_step, _tool_event_payload
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.events import EventStream
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types import RunMode, RunStatus, ToolCall


async def execute_child_tool_calls(
    runtime: Any,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    agent_role: Literal["sub", "fork"],
    parent_agent_id: str,
    parent_run_id: str,
    calls: list[ToolCall],
    allowed_tool_names: set[str],
    stream: EventStream | None = None,
) -> list[Any]:
    context = child_tool_context(
        runtime,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        agent_role=agent_role,
        parent_agent_id=parent_agent_id,
        parent_run_id=parent_run_id,
        allowed_tool_names=allowed_tool_names,
    )

    async def run_one(call: ToolCall):
        await runtime._emit_child_run_event(
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="tool_started",
            tool_call_id=call.tool_call_id,
            payload={"name": call.name},
        )
        result = await runtime.tool_registry.execute(call, context)
        child_handle = RunHandle(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            status=RunStatus.RUNNING,
            mode=RunMode.NORMAL,
            _runtime=runtime,
            _stream=EventStream(),
        )
        await runtime._persist_result_artifacts(child_handle, call, result)
        event_type = _tool_completion_event_type(result)
        await runtime._emit_child_run_event(
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type=event_type,
            tool_call_id=call.tool_call_id,
            payload=_tool_event_payload(call.name, result),
        )
        return result

    return await _run_scheduled_tool_calls(
        calls,
        run_one,
        [tool for tool in runtime._effective_tools(agent_role=agent_role) if tool.name in allowed_tool_names],
    )


async def execute_worker_tool_calls(
    runtime: Any,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    parent_agent_id: str,
    parent_run_id: str,
    calls: list[ToolCall],
    worker_scope: dict[str, Any],
    allowed_tool_names: set[str],
    stream: EventStream | None = None,
) -> list[Any]:
    context = worker_tool_context(
        runtime,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        parent_run_id=parent_run_id,
        worker_scope=worker_scope,
        allowed_tool_names=allowed_tool_names,
    )

    async def run_one(call: ToolCall):
        await runtime._emit_child_run_event(
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="tool_started",
            tool_call_id=call.tool_call_id,
            payload={"name": call.name},
        )
        result = await runtime.tool_registry.execute(call, context)
        worker_handle = RunHandle(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            status=RunStatus.RUNNING,
            mode=RunMode.ORCHESTRATOR,
            _runtime=runtime,
            _stream=EventStream(),
        )
        await runtime._persist_result_artifacts(worker_handle, call, result)
        event_type = _tool_completion_event_type(result)
        await runtime._emit_child_run_event(
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type=event_type,
            tool_call_id=call.tool_call_id,
            payload=_tool_event_payload(call.name, result),
        )
        return result

    return await _run_scheduled_tool_calls(
        calls,
        run_one,
        [tool for tool in runtime._effective_tools(agent_role="worker") if tool.name in allowed_tool_names],
    )


def worker_tool_context(
    runtime: Any,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    parent_agent_id: str,
    parent_run_id: str,
    worker_scope: dict[str, Any],
    allowed_tool_names: set[str] | None = None,
) -> ToolExecutionContext:
    assert runtime.paths and runtime.config and runtime.artifacts
    return ToolExecutionContext(
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        agent_role="worker",
        project_dir=runtime.paths.project_dir,
        home_dir=runtime.paths.home_dir,
        config=runtime.config,
        artifact_manager=runtime.artifacts,
        permission_callback=runtime.permission_callback,
        permission_cache=runtime._permission_caches[session_id],
        parent_agent_id=parent_agent_id,
        parent_run_id=parent_run_id,
        allowed_tool_names=allowed_tool_names or {tool.name for tool in runtime._effective_tools(agent_role="worker")},
        effective_tool_definitions={tool.name: tool for tool in runtime._effective_tools(agent_role="worker")},
        debug=runtime.debug,
        services={
            "task_service": runtime.task_service,
            "agent_definitions": runtime.agent_definitions,
            "context_state": runtime._context_state_for_session(session_id),
            "runtime": runtime,
            "worker_scope": worker_scope,
        },
        hooks=runtime._hooks,
    )


def child_tool_context(
    runtime: Any,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    agent_role: Literal["sub", "fork"],
    parent_agent_id: str,
    parent_run_id: str,
    allowed_tool_names: set[str] | None = None,
) -> ToolExecutionContext:
    assert runtime.paths and runtime.config and runtime.artifacts
    effective_tools = runtime._effective_tools(agent_role=agent_role)
    allowed_names = allowed_tool_names or {tool.name for tool in effective_tools}
    return ToolExecutionContext(
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        agent_role=agent_role,
        project_dir=runtime.paths.project_dir,
        home_dir=runtime.paths.home_dir,
        config=runtime.config,
        artifact_manager=runtime.artifacts,
        permission_callback=runtime.permission_callback,
        permission_cache=runtime._permission_caches[session_id],
        parent_agent_id=parent_agent_id,
        parent_run_id=parent_run_id,
        allowed_tool_names=allowed_names,
        effective_tool_definitions={tool.name: tool for tool in effective_tools if tool.name in allowed_names},
        debug=runtime.debug,
        services={
            "task_service": runtime.task_service,
            "agent_definitions": runtime.agent_definitions,
            "context_state": runtime._context_state_for_session(session_id),
            "runtime": runtime,
        },
        hooks=runtime._hooks,
    )


def worker_step_summary(runtime: Any, *, session_id: str, task_id: str, worker_run_id: str) -> dict[str, Any]:
    try:
        step = runtime.task_service.claimed_step_for_run(session_id, task_id, worker_run_id)
    except AgentCoreError:
        step = None
    if step is None:
        return {
            "claimed_step_id": None,
            "step_status": None,
            "step_result_summary": None,
            "no_step_claimed": True,
        }
    return _summary_from_step(step.model_dump(mode="json"))


def _tool_completion_event_type(result: Any) -> str:
    if not getattr(result, "is_error", False):
        return "tool_completed"
    if getattr(result, "metadata", {}).get("permission_failed"):
        return "permission_failed"
    if result.error and result.error.code == ErrorCode.PERMISSION_DENIED:
        return "tool_denied"
    return "tool_failed"
