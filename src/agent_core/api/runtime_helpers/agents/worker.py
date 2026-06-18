from __future__ import annotations

import asyncio
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.agents.tools import (
    execute_worker_tool_calls,
    worker_tool_context,
    worker_step_summary,
)
from agent_core.api.runtime_helpers.agents.worker_lifecycle import fail_unclosed_worker_step
from agent_core.api.runtime_helpers.agents.worker_executor import (
    WorkerExecutorContext,
    worker_executor_config,
)
from agent_core.api.runtime_helpers.context import _apply_context_budget, _context_build_report
from agent_core.api.runtime_helpers.model import (
    _child_timeout_seconds,
    _collect_model_completion,
    _ensure_provider_supports_request,
    _validate_expected_output_schema,
    _worker_prompt,
)
from agent_core.api.runtime_helpers.tools import _agent_definition_body_with_default
from agent_core.api.runtime_helpers.views import _summary_from_step
from agent_core.config.loader import resolve_model_config
from agent_core.context import build_context_messages, build_system_blocks
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.providers import ModelMessage, ModelRequest, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.storage import new_id
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types import ErrorPayload, Node, RunStatus, TextBlock, ToolCallBlock, ToolResultBlock


WORKER_CONTEXT_NODE_TYPES = {"worker_dispatch", "worker_message", "worker_tool_result"}


async def run_worker_agent(
    runtime: Any,
    *,
    session_id: str,
    parent_run_id: str,
    parent_agent_id: str,
    task_id: str,
    instruction: str,
    worker_pool_id: str | None = None,
    worker_agent_id: str | None = None,
    allowed_step_ids: list[str] | None = None,
    dispatch_context: str | None = None,
    constraints: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    expected_output_schema: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    parent_handle: RunHandle | None = None,
) -> dict[str, Any]:
    self = runtime
    await self._ensure_started()
    assert self.store and self.paths and self.config and self._provider and self.worker_runtime and self.artifacts
    if allowed_step_ids == []:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "allowed_step_ids cannot be empty")
    step_scope = list(dict.fromkeys(str(item) for item in allowed_step_ids)) if allowed_step_ids is not None else None
    selection_worker_agent_id = worker_agent_id
    queued = False
    while True:
        try:
            worker = self.worker_runtime.select_worker(
                worker_pool_id=worker_pool_id,
                worker_agent_id=selection_worker_agent_id,
                session_id=session_id,
            )
            break
        except AgentCoreError as exc:
            if exc.code != ErrorCode.WORKER_BUSY or selection_worker_agent_id is None:
                raise
            queued = True
            selection_worker_agent_id = await _wait_for_worker_queue_turn(
                self,
                session_id=session_id,
                parent_run_id=parent_run_id,
                parent_agent_id=parent_agent_id,
                task_id=task_id,
                worker_pool_id=worker_pool_id,
                worker_agent_id=selection_worker_agent_id,
                parent_handle=parent_handle,
            )
    definition = self.agent_definitions.get(worker.agent_definition_id)
    if definition is None:
        raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"worker agent definition not found: {worker.agent_definition_id}")
    executor_config = worker_executor_config(definition.metadata)
    preflight_context = ToolExecutionContext(
        session_id=session_id,
        run_id=parent_run_id,
        agent_id=parent_agent_id,
        agent_role="orchestrator",
        project_dir=self.paths.project_dir,
        home_dir=self.paths.home_dir,
        config=self.config,
        artifact_manager=self.artifacts,
        permission_callback=self.permission_callback,
        permission_cache=self._permission_caches[session_id],
    )
    dispatchable_steps = self.task_service.dispatchable_steps(
        preflight_context,
        task_id=task_id,
        worker_pool_id=worker.pool_id,
        allowed_step_ids=step_scope,
    )
    agent_id = self._worker_agent_id(session_id=session_id, worker_id=worker.worker_id)
    if not dispatchable_steps:
        return {
            "worker_agent_id": agent_id,
            "worker_id": worker.worker_id,
            "child_run_id": None,
            "stream_id": None,
            "selection_reason": _worker_selection_reason(worker_agent_id=worker_agent_id, queued=queued),
            "worker_result": None,
            "claimed_step_id": None,
            "step_status": None,
            "step_result_summary": None,
            "no_step_claimed": True,
        }
    base_tools = self._effective_tools(agent_role="worker")
    base_names = {tool.name for tool in base_tools}
    if executor_config is None:
        worker_allowed = worker.allowed_tools
        if worker_allowed is not None:
            unknown = [name for name in worker_allowed if self.tool_registry.get(name) is None]
            if unknown:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"worker allowed_tools contains unavailable tools: {unknown}")
            excluded = [name for name in worker_allowed if name not in base_names]
            if excluded:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"worker allowed_tools contains tools outside effective set: {excluded}")
            base_names &= set(worker_allowed)
        if allowed_tools is not None:
            unknown = [name for name in allowed_tools if self.tool_registry.get(name) is None]
            if unknown:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains unavailable tools: {unknown}")
            excluded = [name for name in allowed_tools if name not in base_names]
            if excluded:
                raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"allowed_tools contains tools outside worker effective set: {excluded}")
            base_names &= set(allowed_tools)
    tools = [tool for tool in base_tools if tool.name in base_names]

    run_id = new_id("run_worker")
    worker_stream = self._open_child_run_stream(run_id)
    self.worker_runtime.mark_busy(worker, task_id=task_id, run_id=run_id, step_id=dispatchable_steps[0].step_id)
    current_task = asyncio.current_task()
    if current_task is not None:
        self._worker_run_tasks[run_id] = current_task
        self._worker_run_meta[run_id] = {
            "session_id": session_id,
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "worker_agent_id": agent_id,
        }
    await self.store.ensure_agent(
        agent_id=agent_id,
        session_id=session_id,
        agent_type="sub",
        status="running",
        parent_agent_id=parent_agent_id,
        created_by_run_id=parent_run_id,
        metadata={
            "purpose": "worker",
            "worker_id": worker.worker_id,
            "worker_pool_id": worker.pool_id,
            "agent_definition_id": worker.agent_definition_id,
        },
    )
    await self.store.create_run(run_id=run_id, session_id=session_id, agent_id=agent_id, status=RunStatus.RUNNING.value)
    parent_id = await self.store.active_node_id(session_id)
    worker_prompt = _worker_prompt(
        instruction=instruction,
        task_id=task_id,
        worker_pool_id=worker.pool_id,
        allowed_step_ids=step_scope,
        dispatch_context=dispatch_context,
        constraints=constraints,
        expected_output_schema=expected_output_schema,
    )
    start_node = await self.store.add_node(
        session_id=session_id,
        parent_id=parent_id,
        agent_id=agent_id,
        run_id=run_id,
        role="user",
        node_type="worker_dispatch",
        content=[TextBlock(text=worker_prompt)],
        metadata={
            "parent_agent_id": parent_agent_id,
            "parent_run_id": parent_run_id,
            "task_id": task_id,
            "worker_pool_id": worker.pool_id,
            "allowed_step_ids": step_scope,
        },
        make_active=False,
    )
    await self.store.update_run(run_id=run_id, status=RunStatus.RUNNING.value, start_node_id=start_node.node_id)
    await self._emit_child_run_event(
        stream=worker_stream,
        mirror_handle=parent_handle,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        event_type="worker_run_started",
        node_id=start_node.node_id,
        payload={
            "parent_agent_id": parent_agent_id,
            "parent_run_id": parent_run_id,
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "worker_agent_id": agent_id,
            "child_run_id": run_id,
            "stream_id": run_id,
        },
    )
    worker_scope = {"task_id": task_id, "allowed_step_ids": step_scope, "worker_pool_id": worker.pool_id}
    messages = await _worker_context_messages(
        self,
        session_id=session_id,
        worker_agent_id=agent_id,
        start_node=start_node,
    )
    end_node_id: str | None = start_node.node_id
    final_text = ""
    worker_result: dict[str, Any] | None = None
    error_payload: ErrorPayload | None = None
    try:
        if executor_config is not None:
            executor = self._worker_executors.get(executor_config.type)
            if executor is None:
                raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"worker executor not registered: {executor_config.type}")
            step_context = worker_tool_context(
                self,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                parent_agent_id=parent_agent_id,
                parent_run_id=parent_run_id,
                worker_scope=worker_scope,
                allowed_tool_names=set(),
            )
            claimed = self.task_service.claim_step(step_context, {"task_id": task_id, "step_id": dispatchable_steps[0].step_id})
            self.task_service.update_step(
                step_context,
                {"task_id": task_id, "step_id": claimed["step"]["step_id"], "status": "running"},
            )
            async with asyncio.timeout(_child_timeout_seconds(self.config, timeout_ms)):
                result = await executor.run(
                    self,
                    WorkerExecutorContext(
                        session_id=session_id,
                        parent_run_id=parent_run_id,
                        parent_agent_id=parent_agent_id,
                        task_id=task_id,
                        instruction=instruction,
                        dispatch_context=dispatch_context,
                        constraints=constraints,
                        allowed_step_ids=step_scope,
                        allowed_tools=allowed_tools,
                        expected_output_schema=expected_output_schema,
                        timeout_ms=timeout_ms,
                        worker=worker,
                        worker_agent_id=agent_id,
                        worker_run_id=run_id,
                        worker_start_node=start_node,
                        worker_stream=worker_stream,
                        executor_config=executor_config.config,
                        agent_definition=definition,
                        parent_handle=parent_handle,
                    ),
                )
            final_text = result.text
            worker_result = result.as_worker_result()
            assistant_node = await self.store.add_node(
                session_id=session_id,
                parent_id=end_node_id,
                agent_id=agent_id,
                run_id=run_id,
                role="assistant",
                node_type="worker_message",
                content=[TextBlock(text=final_text)],
                metadata={"executor_type": executor_config.type, **(result.metadata or {})},
                make_active=False,
            )
            end_node_id = assistant_node.node_id
            _validate_expected_output_schema(final_text, expected_output_schema)
            self.task_service.update_step(
                step_context,
                {
                    "task_id": task_id,
                    "step_id": claimed["step"]["step_id"],
                    "status": "completed",
                    "result_summary": final_text.strip()[:500] or "external worker completed",
                },
            )
        else:
            model_config = resolve_model_config(self.config, definition.model_profile)
            async with asyncio.timeout(_child_timeout_seconds(self.config, timeout_ms)):
                for _turn in range(self.config.runtime.max_turns):
                    system_blocks = build_system_blocks(
                        home_dir=self.paths.home_dir,
                        project_dir=self.paths.project_dir,
                        context_state=self._context_state_for_session(session_id),
                        memory_enabled=self.config.memory.enabled,
                        memory_dir_template=self.config.memory.memory_dir,
                    ) + [
                        SystemBlock(
                            block_id=f"agent_definition.{worker.agent_definition_id}",
                            source="agent_definition",
                            content=_agent_definition_body_with_default(
                                self.agent_definitions,
                                definition,
                                default_id="default_worker_agent",
                            ),
                            priority=900,
                            dynamic=True,
                            metadata={
                                "agent_definition_id": worker.agent_definition_id,
                                "fallback_default_id": None if definition.body else "default_worker_agent",
                            },
                        )
                    ]
                    context_bundle = _apply_context_budget(
                        messages=messages,
                        system_blocks=system_blocks,
                        context_config=self.config.context,
                        model_config=model_config,
                    )
                    await self._emit_child_run_event(
                        stream=worker_stream,
                        mirror_handle=parent_handle,
                        session_id=session_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        event_type="context_built",
                        payload=_context_build_report(
                            context_bundle["messages"],
                            system_blocks,
                            tools,
                            trimmed_node_ids=context_bundle["trimmed_node_ids"],
                            budget=context_bundle["budget"],
                            tokens_before_trim=context_bundle["tokens_before_trim"],
                            tokens_after_trim=context_bundle["tokens_after_trim"],
                            non_system_tokens_before_trim=context_bundle["non_system_tokens_before_trim"],
                            non_system_tokens_after_trim=context_bundle["non_system_tokens_after_trim"],
                            too_long=context_bundle["too_long"],
                        )
                        | {"model": model_config.name},
                    )
                    if context_bundle["too_long"]:
                        raise AgentCoreError(
                            ErrorCode.VALIDATION_ERROR,
                            "prompt_too_long",
                            details={
                                "end_reason": "prompt_too_long",
                                "estimated_input_tokens": context_bundle["tokens_after_trim"],
                                "non_system_budget": context_bundle["budget"],
                            },
                        )
                    request = ModelRequest(
                        model=model_config.name,
                        system=system_blocks,
                        messages=context_bundle["messages"],
                        tools=tools,
                        temperature=model_config.temperature,
                        max_output_tokens=model_config.max_output_tokens,
                        metadata={"session_id": session_id, "run_id": run_id, "parent_run_id": parent_run_id, "worker_id": worker.worker_id},
                    )
                    provider = self._provider_for_model(model_config)
                    _ensure_provider_supports_request(provider, request)
                    async def on_worker_model_event(event: Any) -> None:
                        await self._emit_child_model_event(
                            stream=worker_stream,
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            event=event,
                        )
                        if event.event_type == "model_text_delta" and event.text_delta:
                            await self._emit_child_run_event(
                                stream=worker_stream,
                                mirror_handle=parent_handle,
                                session_id=session_id,
                                agent_id=agent_id,
                                run_id=run_id,
                                event_type="worker_text_delta",
                                payload={
                                    "text": event.text_delta,
                                    "worker_id": worker.worker_id,
                                    "worker_agent_id": agent_id,
                                    "child_run_id": run_id,
                                },
                            )

                    completed, text_parts = await _collect_model_completion(
                        provider,
                        request,
                        provider_failure_message="worker provider failed",
                        on_model_event=on_worker_model_event,
                        on_completed=lambda event: self._persist_run_debug_artifact(
                            session_id=session_id,
                            agent_id=agent_id,
                            run_id=run_id,
                            model_event=event,
                            node_id=None,
                        ),
                    )
                    final_text += "".join(text_parts)
                    assistant_content = list(completed.content)
                    for block in completed.content:
                        if getattr(block, "type", None) == "text":
                            final_text = getattr(block, "text", final_text)
                    for call in completed.tool_calls:
                        assistant_content.append(ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata))
                    assistant_node = await self.store.add_node(
                        session_id=session_id,
                        parent_id=end_node_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        role="assistant",
                        node_type="worker_message",
                        content=assistant_content,
                        metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None},
                        make_active=False,
                    )
                    end_node_id = assistant_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.ASSISTANT,
                            content=assistant_content,
                            node_type="worker_message",
                            metadata={"node_id": assistant_node.node_id},
                        )
                    )
                    if not completed.tool_calls:
                        _validate_expected_output_schema(final_text, expected_output_schema)
                        worker_result = {"text": final_text}
                        break
                    tool_results = await execute_worker_tool_calls(
                        self,
                        session_id=session_id,
                        run_id=run_id,
                        agent_id=agent_id,
                        parent_agent_id=parent_agent_id,
                        parent_run_id=parent_run_id,
                        calls=completed.tool_calls,
                        worker_scope=worker_scope,
                        allowed_tool_names=base_names,
                        stream=worker_stream,
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
                        node_type="worker_tool_result",
                        content=tool_content,
                        metadata={},
                        make_active=False,
                    )
                    end_node_id = tool_node.node_id
                    messages.append(
                        ModelMessage(
                            role=ModelRole.TOOL,
                            content=tool_content,
                            node_type="worker_tool_result",
                            metadata={"node_id": tool_node.node_id},
                        )
                    )
                else:
                    raise AgentCoreError(ErrorCode.INTERNAL_ERROR, "worker max turns exceeded")
        if error_payload:
            raise AgentCoreError(error_payload.code, error_payload.message, details=error_payload.details)
        fallback = fail_unclosed_worker_step(
            self,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            task_id=task_id,
            reason="worker_finished_without_terminal_step_status",
        )
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.COMPLETED.value,
            end_node_id=end_node_id,
            end_reason="completed",
        )
        summary = worker_step_summary(self, session_id=session_id, task_id=task_id, worker_run_id=run_id)
        if fallback is not None:
            summary = _summary_from_step(fallback["step"])
        await self._emit_child_run_event(
            stream=worker_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="worker_run_completed",
            node_id=end_node_id,
            payload={
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "summary": summary,
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
                "child_run_id": run_id,
                "stream_id": run_id,
                "worker_result": worker_result or {"text": final_text},
            },
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status="idle",
            result={
                "last_run_id": run_id,
                "task_id": task_id,
                "summary": summary,
                "worker_result": worker_result or {"text": final_text},
            },
        )
        return {
            "worker_agent_id": agent_id,
            "worker_id": worker.worker_id,
            "child_run_id": run_id,
            "worker_start_node_id": start_node.node_id,
            "worker_end_node_id": end_node_id,
            "stream_id": run_id,
            "selection_reason": _worker_selection_reason(worker_agent_id=worker_agent_id, queued=queued),
            "worker_result": worker_result or {"text": final_text},
            **summary,
        }
    except TimeoutError as exc:
        fail_unclosed_worker_step(
            self,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            task_id=task_id,
            reason="worker_timeout",
        )
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            end_node_id=end_node_id,
            end_reason="failed",
            error={"code": ErrorCode.TIMEOUT.value, "message": "worker agent timed out", "reason": "timeout"},
        )
        await self._emit_child_run_event(
            stream=worker_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="worker_run_failed",
            level="error",
            node_id=end_node_id,
            payload={
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "code": ErrorCode.TIMEOUT.value,
                "message": "worker agent timed out",
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
                "child_run_id": run_id,
                "stream_id": run_id,
            },
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status=RunStatus.FAILED.value,
            result={"last_run_id": run_id, "task_id": task_id, "error": "worker agent timed out"},
        )
        raise AgentCoreError(ErrorCode.TIMEOUT, "worker agent timed out") from exc
    except asyncio.CancelledError:
        fail_unclosed_worker_step(
            self,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            task_id=task_id,
            reason="worker_cancelled",
        )
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.CANCELLED.value,
            end_node_id=end_node_id,
            end_reason="aborted_tools",
            error={"code": ErrorCode.CANCELLED.value, "message": "worker agent cancelled", "reason": "cancelled"},
        )
        await self._emit_child_run_event(
            stream=worker_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="worker_run_cancelled",
            node_id=end_node_id,
            payload={
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
                "child_run_id": run_id,
                "stream_id": run_id,
            },
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status=RunStatus.CANCELLED.value,
            result={"last_run_id": run_id, "task_id": task_id, "cancelled": True},
        )
        raise
    except Exception as exc:
        fail_unclosed_worker_step(
            self,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            parent_run_id=parent_run_id,
            worker_scope=worker_scope,
            task_id=task_id,
            reason="worker_failed",
        )
        code = getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
        message_text = getattr(exc, "message", str(exc))
        await self.store.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            end_node_id=end_node_id,
            end_reason="failed",
            error={"code": str(code), "message": message_text},
        )
        await self._emit_child_run_event(
            stream=worker_stream,
            mirror_handle=parent_handle,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="worker_run_failed",
            level="error",
            node_id=end_node_id,
            payload={
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "code": str(code),
                "message": message_text,
                "worker_id": worker.worker_id,
                "worker_agent_id": agent_id,
                "child_run_id": run_id,
                "stream_id": run_id,
            },
        )
        await self.store.update_agent(
            agent_id=agent_id,
            status=RunStatus.FAILED.value,
            result={"last_run_id": run_id, "task_id": task_id, "error": message_text},
        )
        raise
    finally:
        self._worker_run_tasks.pop(run_id, None)
        self._worker_run_meta.pop(run_id, None)
        self.worker_runtime.mark_idle(worker)
        _wake_next_worker_queue_item(self, worker.worker_id)
        await self._close_child_run_stream(run_id)


async def _wait_for_worker_queue_turn(
    runtime: Any,
    *,
    session_id: str,
    parent_run_id: str,
    parent_agent_id: str,
    task_id: str,
    worker_pool_id: str | None,
    worker_agent_id: str,
    parent_handle: RunHandle | None,
) -> str:
    worker = runtime.worker_runtime._workers.get(worker_agent_id) or runtime.worker_runtime._worker_by_agent_id(  # noqa: SLF001
        worker_agent_id,
        session_id=session_id,
    )
    if worker is None:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"worker not available: {worker_agent_id}")
    if worker_pool_id is not None and worker.pool_id != worker_pool_id:
        raise AgentCoreError(ErrorCode.WORKER_NOT_AVAILABLE, f"worker {worker_agent_id} is not in pool {worker_pool_id}")
    queue = runtime._worker_queues[worker.worker_id]
    if len(queue) >= runtime._worker_queue_limit:
        raise AgentCoreError(ErrorCode.WORKER_QUEUE_FULL, "worker queue full", details={"worker_id": worker.worker_id})
    from agent_core.types.common import utc_iso

    future = asyncio.get_running_loop().create_future()
    queue_id = new_id("worker_queue")
    now = utc_iso()
    item = {
        "queue_id": queue_id,
        "worker_id": worker.worker_id,
        "worker_agent_id": worker_agent_id,
        "session_id": session_id,
        "parent_run_id": parent_run_id,
        "parent_agent_id": parent_agent_id,
        "task_id": task_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "future": future,
        "cancelled": False,
    }
    queue.append(item)
    await runtime._emit_child_run_event(
        stream=None,
        mirror_handle=parent_handle,
        session_id=session_id,
        agent_id=parent_agent_id,
        run_id=parent_run_id,
        event_type="worker_queued",
        payload={
            "queue_id": queue_id,
            "worker_id": worker.worker_id,
            "worker_agent_id": worker_agent_id,
            "task_id": task_id,
            "parent_run_id": parent_run_id,
            "position": len(queue),
        },
    )
    if worker.status == "idle":
        _wake_next_worker_queue_item(runtime, worker.worker_id)
    try:
        return await future
    except asyncio.CancelledError:
        runtime.cancel_worker_queue_item(queue_id)
        raise


async def _worker_context_messages(runtime: Any, *, session_id: str, worker_agent_id: str, start_node: Node) -> list[ModelMessage]:
    active_path = await _active_path(runtime, session_id)
    context_nodes = await _worker_history_nodes_from_active_path(runtime, session_id=session_id, worker_agent_id=worker_agent_id, active_path=active_path)
    context_nodes.append(start_node)
    return build_context_messages(context_nodes)


async def _active_path(runtime: Any, session_id: str) -> list[Node]:
    active_node_id = await runtime.store.active_node_id(session_id)
    if not active_node_id:
        return []
    return await runtime.store.get_session_node_path(session_id, active_node_id)


async def _worker_history_nodes_from_active_path(runtime: Any, *, session_id: str, worker_agent_id: str, active_path: list[Node]) -> list[Node]:
    end_node_ids = _worker_end_node_ids_from_active_path(active_path)
    by_id: dict[str, Node] = {}
    for end_node_id in end_node_ids:
        path = await runtime.store.get_session_node_path(session_id, end_node_id)
        for node in path:
            if node.agent_id == worker_agent_id and node.node_type in WORKER_CONTEXT_NODE_TYPES:
                by_id[node.node_id] = node
    return list(by_id.values())


def _worker_end_node_ids_from_active_path(active_path: list[Node]) -> list[str]:
    node_ids: list[str] = []
    for node in active_path:
        for block in node.content:
            if getattr(block, "type", None) != "tool_result":
                continue
            for item in getattr(block, "content", []):
                data = getattr(item, "data", None)
                if isinstance(data, dict) and data.get("worker_end_node_id"):
                    node_ids.append(str(data["worker_end_node_id"]))
    return list(dict.fromkeys(node_ids))


def _wake_next_worker_queue_item(runtime: Any, worker_id: str) -> None:
    queue = runtime._worker_queues.get(worker_id)
    if not queue:
        return
    from agent_core.types.common import utc_iso

    while queue:
        item = queue.popleft()
        if item.get("cancelled"):
            continue
        item["status"] = "dequeued"
        item["updated_at"] = utc_iso()
        future = item.get("future")
        if future is not None and not future.done():
            future.set_result(item["worker_agent_id"])
        break


def _worker_selection_reason(*, worker_agent_id: str | None, queued: bool) -> str:
    if queued:
        return "queued_worker"
    return "first_idle" if worker_agent_id is None else "specified_worker"
