from __future__ import annotations

import asyncio
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.api.runtime_helpers.views import _tool_event_payload
from agent_core.errors.codes import ErrorCode
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types import AgentDefinition, ErrorPayload, RunMode, ToolCall, ToolDefinition
from agent_core.types.tools import error_tool_result


async def _run_scheduled_tool_calls(calls: list[ToolCall], run_one: Any, definitions: list[ToolDefinition]) -> list[Any]:
    by_name = {definition.name: definition for definition in definitions}
    results: list[Any] = []
    batch: list[ToolCall] = []

    async def flush_readonly_batch() -> None:
        nonlocal batch
        if not batch:
            return
        results.extend(await asyncio.gather(*(run_one(call) for call in batch)))
        batch = []

    for call in calls:
        definition = by_name.get(call.name)
        is_write = definition is None or definition.permission == "write" or "write" in definition.tags
        if not is_write:
            batch.append(call)
            continue
        await flush_readonly_batch()
        result = await run_one(call)
        results.append(result)
        if getattr(result, "is_error", False):
            break
    await flush_readonly_batch()
    return results


def _agent_definition_body_with_default(
    definitions: AgentDefinitionRegistry,
    definition: AgentDefinition,
    *,
    default_id: str,
) -> str:
    if definition.body:
        return definition.body
    default_definition = definitions.get(default_id)
    return default_definition.body if default_definition is not None else ""


async def execute_tool_calls(runtime: Any, handle: RunHandle, calls: list[ToolCall]):
    assert runtime.paths and runtime.config and runtime.artifacts
    effective_tools = runtime._effective_tools(agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main")
    context = ToolExecutionContext(
        session_id=handle.session_id,
        run_id=handle.run_id,
        agent_id=handle.agent_id,
        agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main",
        project_dir=runtime.paths.project_dir,
        home_dir=runtime.paths.home_dir,
        config=runtime.config,
        artifact_manager=runtime.artifacts,
        permission_callback=runtime.permission_callback,
        permission_cache=runtime._permission_caches[handle.session_id],
        allowed_tool_names={tool.name for tool in effective_tools},
        effective_tool_definitions={tool.name: tool for tool in effective_tools},
        debug=runtime.debug,
        run_handle=handle,
        services={
            "task_service": runtime.task_service,
            "agent_definitions": runtime.agent_definitions,
            "context_state": runtime._context_state_for_session(handle.session_id),
            "runtime": runtime,
        },
        hooks=runtime._hooks,
    )

    async def run_one(call: ToolCall):
        await runtime._emit(handle, "tool_started", tool_call_id=call.tool_call_id, payload={"name": call.name})
        try:
            result = await runtime.tool_registry.execute(call, context)
        except asyncio.CancelledError:
            if handle._task is not None and handle._task.cancelling():
                raise
            result = error_tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                error=ErrorPayload(code=ErrorCode.CANCELLED, message="tool execution cancelled"),
            )
        await runtime._persist_result_artifacts(handle, call, result)
        event_type = "tool_completed"
        if result.is_error:
            if getattr(result, "metadata", {}).get("permission_failed"):
                event_type = "permission_failed"
            elif result.error and result.error.code == ErrorCode.PERMISSION_DENIED:
                event_type = "tool_denied"
            else:
                event_type = "tool_failed"
        await runtime._emit(
            handle,
            event_type,
            tool_call_id=call.tool_call_id,
            payload=_tool_event_payload(call.name, result),
        )
        return result

    return await _run_scheduled_tool_calls(calls, run_one, effective_tools)
