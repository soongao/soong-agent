from __future__ import annotations

import asyncio
from typing import Any, Literal

from agent_core.agents.child import ChildAgentManager
from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.agents.tools import execute_child_tool_calls
from agent_core.api.runtime_helpers.context import _context_build_report
from agent_core.api.runtime_helpers.model import (
    _child_prompt,
    _child_timeout_seconds,
    _collect_model_completion,
    _ensure_provider_supports_request,
    _validate_expected_output_schema,
)
from agent_core.api.runtime_helpers.tools import _agent_definition_body_with_default
from agent_core.config.loader import resolve_model_config
from agent_core.context import build_system_blocks
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.providers import ModelMessage, ModelRequest, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.storage import new_id
from agent_core.types import ErrorPayload, RunMode, RunStatus, TextBlock, ToolCallBlock, ToolResultBlock


async def run_child_agent(
    runtime: Any,
    *,
    session_id: str,
    parent_run_id: str,
    parent_agent_id: str,
    agent_definition_id: str,
    task: str,
    mode: Literal["sub", "fork"] = "sub",
    constraints: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    expected_output_schema: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    parent_handle: RunHandle | None = None,
) -> dict[str, Any]:
    self = runtime
    await self._ensure_started()
    assert self.store and self.paths and self.config and self._provider
    await self._ensure_mcp_tools(session_id=session_id, agent_id=parent_agent_id, run_id=parent_run_id)
    definition = self.agent_definitions.get(agent_definition_id)
    if definition is None:
        raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"agent definition not found: {agent_definition_id}")
    if allowed_tools is not None:
        unknown = [name for name in allowed_tools if self.tool_registry.get(name) is None]
        if unknown:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains unavailable tools: {unknown}")
        effective_names = {tool.name for tool in self._effective_tools(agent_role=mode)}
        excluded = [name for name in allowed_tools if name not in effective_names]
        if excluded:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains tools outside effective set: {excluded}")
    child_manager = self._child_managers.setdefault(
        parent_run_id,
        ChildAgentManager(max_children_per_run=self.config.agents.max_children_per_run),
    )
    if not child_manager.can_start():
        raise AgentCoreError(
            ErrorCode.CHILD_AGENT_LIMIT_EXCEEDED,
            f"parent run exceeds max_children_per_run={self.config.agents.max_children_per_run}",
        )
    if self._session_child_counts[session_id] >= self.config.agents.max_concurrent_children_per_session:
        raise AgentCoreError(
            ErrorCode.CHILD_AGENT_LIMIT_EXCEEDED,
            f"session exceeds max_concurrent_children_per_session={self.config.agents.max_concurrent_children_per_session}",
        )
    child_manager.started()
    self._session_child_counts[session_id] += 1
    agent_id = new_id(f"agent_{mode}")
    run_id = new_id("run")
    child_stream = self._open_child_run_stream(run_id)
    parent_id = await self.store.active_node_id(session_id)
    fork_from_node_id = parent_id if mode == "fork" else None
    await self.store.ensure_agent(
        agent_id=agent_id,
        session_id=session_id,
        agent_type=mode,
        status="running",
        parent_agent_id=parent_agent_id,
        created_by_run_id=parent_run_id,
        fork_from_node_id=fork_from_node_id,
        metadata={
            "agent_definition_id": agent_definition_id,
            "purpose": mode,
        },
    )
    await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
    await self._emit_child_run_event(
        stream=child_stream,
        mirror_handle=parent_handle,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        event_type="child_agent_created" if mode == "sub" else "fork_agent_created",
        payload={
            "parent_agent_id": parent_agent_id,
            "parent_run_id": parent_run_id,
            "agent_definition_id": agent_definition_id,
            "child_agent_id": agent_id,
            "child_run_id": run_id,
            "fork_from_node_id": fork_from_node_id,
        },
    )
    task_node = await self.store.add_node(
        session_id=session_id,
        parent_id=parent_id,
        agent_id=agent_id,
        run_id=run_id,
        role="user",
        node_type="message",
        content=[TextBlock(text=task)],
        metadata={
            "parent_run_id": parent_run_id,
            "parent_agent_id": parent_agent_id,
            "agent_definition_id": agent_definition_id,
            "constraints": constraints,
            "expected_output_schema": expected_output_schema,
        },
        make_active=False,
    )
    tools = self._effective_tools(agent_role=mode)
    base_tool_names = {tool.name for tool in tools}
    if allowed_tools is not None:
        allowed_set = set(allowed_tools)
        tools = [tool for tool in tools if tool.name in allowed_set]
        base_tool_names &= allowed_set
    child_prompt = _child_prompt(
        task=task,
        constraints=constraints,
        expected_output_schema=expected_output_schema,
    )
    model_config = resolve_model_config(self.config, definition.model_profile)
    provider = self._provider_for_model(model_config)
    messages = [
        ModelMessage(
            role=ModelRole.USER,
            content=[TextBlock(text=child_prompt)],
            node_type="message",
            metadata={"node_id": task_node.node_id},
        )
    ]
    end_node_id = task_node.node_id
    result_text = ""
    timeout_seconds = _child_timeout_seconds(self.config, timeout_ms)
    try:
        async with asyncio.timeout(timeout_seconds):
            for _turn in range(self.config.runtime.max_turns):
                system_blocks = build_system_blocks(
                    home_dir=self.paths.home_dir,
                    project_dir=self.paths.project_dir,
                    context_state=self._context_state_for_session(session_id),
                    memory_enabled=self.config.memory.enabled,
                    memory_dir_template=self.config.memory.memory_dir,
                ) + [
                    SystemBlock(
                        block_id=f"agent_definition.{agent_definition_id}",
                        source="agent_definition",
                        content=_agent_definition_body_with_default(
                            self.agent_definitions,
                            definition,
                            default_id=self.config.agents.default_sub_agent_definition
                            if mode == "sub"
                            else self.config.agents.default_fork_agent_definition,
                        ),
                        priority=900,
                        dynamic=True,
                        metadata={
                            "agent_definition_id": agent_definition_id,
                            "fallback_default_id": None
                            if definition.body
                            else (
                                self.config.agents.default_sub_agent_definition
                                if mode == "sub"
                                else self.config.agents.default_fork_agent_definition
                            ),
                        },
                    )
                ]
                await self._emit_child_run_event(
                    stream=child_stream,
                    mirror_handle=parent_handle,
                    session_id=session_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    event_type="context_built",
                    payload=_context_build_report(messages, system_blocks, tools) | {"model": model_config.name},
                )
                request = ModelRequest(
                    model=model_config.name,
                    system=system_blocks,
                    messages=messages,
                    tools=tools,
                    temperature=model_config.temperature,
                    max_output_tokens=model_config.max_output_tokens,
                    metadata={"session_id": session_id, "run_id": run_id, "parent_run_id": parent_run_id},
                )
                _ensure_provider_supports_request(provider, request)
                completed, text_parts = await _collect_model_completion(
                    provider,
                    request,
                    provider_failure_message=f"{mode} provider failed",
                    on_model_event=lambda event: self._emit_child_model_event(
                        stream=child_stream,
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        event=event,
                    ),
                    on_completed=lambda event: self._persist_run_debug_artifact(
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        model_event=event,
                        node_id=None,
                    ),
                )
                assistant_content = list(completed.content)
                turn_text = ""
                for block in completed.content:
                    if getattr(block, "type", None) == "text":
                        turn_text += getattr(block, "text", "")
                if not turn_text:
                    turn_text = "".join(text_parts)
                if turn_text:
                    result_text = turn_text
                for call in completed.tool_calls:
                    assistant_content.append(
                        ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata)
                    )
                assistant_node = await self.store.add_node(
                    session_id=session_id,
                    parent_id=end_node_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    role="assistant",
                    node_type="child_message",
                    content=assistant_content,
                    metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None, "mode": mode},
                    make_active=False,
                )
                end_node_id = assistant_node.node_id
                messages.append(
                    ModelMessage(
                        role=ModelRole.ASSISTANT,
                        content=assistant_content,
                        node_type="child_message",
                        metadata={"node_id": assistant_node.node_id},
                    )
                )
                if not completed.tool_calls:
                    _validate_expected_output_schema(result_text, expected_output_schema)
                    break
                tool_results = await execute_child_tool_calls(
                    self,
                    session_id=session_id,
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_role=mode,
                    parent_agent_id=parent_agent_id,
                    parent_run_id=parent_run_id,
                    calls=completed.tool_calls,
                    allowed_tool_names=base_tool_names,
                    stream=child_stream,
                )
                tool_content = [
                    ToolResultBlock(
                        tool_call_id=result.tool_call_id,
                        is_error=result.is_error,
                        content=result.content,
                        error=result.error,
                        metadata={**result.metadata, "tool_name": result.tool_name},
                    )
                    for result in tool_results
                ]
                tool_node = await self.store.add_node(
                    session_id=session_id,
                    parent_id=end_node_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    role="tool",
                    node_type="child_tool_result",
                    content=tool_content,
                    metadata={"mode": mode},
                    make_active=False,
                )
                end_node_id = tool_node.node_id
                messages.append(
                    ModelMessage(
                        role=ModelRole.TOOL,
                        content=tool_content,
                        node_type="child_tool_result",
                        metadata={"node_id": tool_node.node_id},
                    )
                )
            else:
                raise AgentCoreError(ErrorCode.INTERNAL_ERROR, f"{mode} agent max turns exceeded")
    except TimeoutError as exc:
        child_manager.finished()
        self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            start_node_id=task_node.node_id,
            end_node_id=end_node_id,
            end_reason="failed",
            error={"code": ErrorCode.TIMEOUT.value, "message": f"{mode} agent timed out", "reason": "timeout"},
        )
        await self._emit_child_run_event(
            stream=child_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
            level="error",
            node_id=end_node_id,
            payload={"code": ErrorCode.TIMEOUT.value, "message": f"{mode} agent timed out", "child_run_id": run_id},
        )
        await self._close_child_run_stream(run_id)
        raise AgentCoreError(ErrorCode.TIMEOUT, f"{mode} agent timed out") from exc
    except AgentCoreError as exc:
        child_manager.finished()
        self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            start_node_id=task_node.node_id,
            end_node_id=end_node_id,
            end_reason="failed",
            error={"code": exc.code.value, "message": exc.message},
        )
        await self._emit_child_run_event(
            stream=child_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
            level="error",
            node_id=end_node_id,
            payload={"code": exc.code.value, "message": exc.message, "child_run_id": run_id},
        )
        await self._close_child_run_stream(run_id)
        raise
    except Exception as exc:
        child_manager.finished()
        self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            start_node_id=task_node.node_id,
            end_node_id=end_node_id,
            end_reason="failed",
            error={"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc)},
        )
        await self._emit_child_run_event(
            stream=child_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="child_agent_failed" if mode == "sub" else "fork_agent_failed",
            level="error",
            node_id=end_node_id,
            payload={"code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc), "child_run_id": run_id},
        )
        await self._close_child_run_stream(run_id)
        raise
    result_node = await self.store.add_node(
        session_id=session_id,
        parent_id=end_node_id,
        agent_id=agent_id,
        run_id=run_id,
        role="assistant",
        node_type="child_result",
        content=[TextBlock(text=result_text)],
        metadata={
            "agent_definition_id": agent_definition_id,
            "mode": mode,
            "constraints": constraints,
            "expected_output_schema": expected_output_schema,
        },
        make_active=False,
    )
    await self.store.update_run(
        run_id=run_id,
        status=RunStatus.COMPLETED.value,
        start_node_id=task_node.node_id,
        end_node_id=result_node.node_id,
        end_reason="completed",
    )
    await self.store.update_agent(
        agent_id=agent_id,
        status=RunStatus.COMPLETED.value,
        result={
            "result_summary": result_text,
            "child_run_id": run_id,
            "agent_definition_id": agent_definition_id,
        },
    )
    await self._emit_child_run_event(
        stream=child_stream,
        mirror_handle=parent_handle,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        event_type="child_agent_completed" if mode == "sub" else "fork_agent_completed",
        node_id=result_node.node_id,
        payload={"result_summary": result_text, "child_agent_id": agent_id, "child_run_id": run_id},
    )
    await self._close_child_run_stream(run_id)
    child_manager.finished()
    self._session_child_counts[session_id] = max(0, self._session_child_counts[session_id] - 1)
    return {
        "child_run_id": run_id,
        "child_agent_id": agent_id,
        "agent_definition_id": agent_definition_id,
        "result_summary": result_text,
        "status": "completed",
    }
