from __future__ import annotations

import asyncio
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.context import (
    _apply_context_budget,
    _apply_system_block_budget,
    _context_build_report,
)
from agent_core.api.runtime_helpers.model import _ensure_provider_supports_request
from agent_core.api.runtime_helpers.views import (
    _content_has_text,
    _last_message_is_tool_result,
    _synthetic_context_nodes_from_tool_results,
    _task_board_context_message,
)
from agent_core.context import build_context_messages, build_system_blocks
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.providers import ModelMessage, ModelRequest
from agent_core.providers.base import ModelRole
from agent_core.types import ErrorPayload, RunMode, RunStatus, TextBlock, ToolCallBlock, ToolResultBlock, UserMessage


async def run_main_loop(runtime: Any, handle: RunHandle, message: str | UserMessage) -> None:
    self = runtime
    assert self.store and self.paths and self.config and self._provider and self.artifacts
    self._cancel_memory_idle_task(handle.session_id)
    handle.status = RunStatus.RUNNING
    await self.store.update_run(run_id=handle.run_id, status=RunStatus.RUNNING.value)
    await self._emit(handle, "loop_started")
    await self._ensure_mcp_tools(handle=handle)
    user = message if isinstance(message, UserMessage) else UserMessage.from_text(message)
    parent_id = await self.store.active_node_id(handle.session_id)
    user_content = user.content if isinstance(user.content, list) else [TextBlock(text=user.content)]
    prompt_text = "\n".join(getattr(block, "text", "") for block in user_content if getattr(block, "type", None) == "text")
    await self._run_observe_hook(
        event_type="user_prompt_submitted",
        session_id=handle.session_id,
        agent_id=handle.agent_id,
        run_id=handle.run_id,
        payload={
            "event_type": "UserPromptSubmit",
            "session_id": handle.session_id,
            "agent_id": handle.agent_id,
            "run_id": handle.run_id,
            "prompt": prompt_text[:12000],
            "metadata": user.metadata,
        },
    )
    user_node = await self.store.add_node(
        session_id=handle.session_id,
        parent_id=parent_id,
        agent_id=handle.agent_id,
        run_id=handle.run_id,
        role="user",
        node_type="message",
        content=user_content,
        metadata=user.metadata,
        make_active=True,
    )
    await self.store.update_run(run_id=handle.run_id, status=RunStatus.RUNNING.value, start_node_id=user_node.node_id)
    await self._emit(handle, "message_created", node_id=user_node.node_id, payload={"role": "user"})
    messages = build_context_messages(await self.store.get_node_path(user_node.node_id))
    task_board_message = _task_board_context_message(self.task_service, handle.session_id)
    if task_board_message is not None:
        messages.append(task_board_message)
    end_node_id: str | None = None
    partial_text_parts: list[str] = []
    model_parent_node_id: str | None = None
    empty_tool_result_retries = 0
    try:
        for _turn in range(self.config.runtime.max_turns):
            partial_text_parts = []
            system_blocks = build_system_blocks(
                home_dir=self.paths.home_dir,
                project_dir=self.paths.project_dir,
                context_state=self._context_state_for_session(handle.session_id),
                memory_enabled=self.config.memory.enabled,
                memory_dir_template=self.config.memory.memory_dir,
            )
            system_blocks.extend(_directive_system_blocks(handle))
            system_budget = _apply_system_block_budget(system_blocks, self.config.context.dynamic_system_budget)
            system_blocks = system_budget["system_blocks"]
            tools = self._effective_tools(agent_role="orchestrator" if handle.mode == RunMode.ORCHESTRATOR else "main")
            context_bundle = _apply_context_budget(
                messages=messages,
                system_blocks=system_blocks,
                context_config=self.config.context,
                model_config=self.config.model,
            )
            await self._emit(
                handle,
                "context_built",
                payload=_context_build_report(
                    context_bundle["messages"],
                    system_blocks,
                    tools,
                    trimmed_node_ids=context_bundle["trimmed_node_ids"],
                    trimmed_system_blocks=system_budget["trimmed_system_blocks"],
                    budget=context_bundle["budget"],
                    tokens_before_trim=context_bundle["tokens_before_trim"],
                    tokens_after_trim=context_bundle["tokens_after_trim"],
                    non_system_tokens_before_trim=context_bundle["non_system_tokens_before_trim"],
                    non_system_tokens_after_trim=context_bundle["non_system_tokens_after_trim"],
                    too_long=context_bundle["too_long"],
                )
                | {"model": self.config.model.name},
            )
            if context_bundle["too_long"]:
                recovered = await self._try_recovery_compact(
                    handle,
                    user_node=user_node,
                    messages=messages,
                    system_blocks=system_blocks,
                    context_config=self.config.context,
                    tools=tools,
                )
                if recovered is not None:
                    messages = recovered["messages"]
                    context_bundle = recovered["context_bundle"]
                    end_node_id = recovered["end_node_id"]
                    await self._emit(
                        handle,
                        "context_built",
                        payload=_context_build_report(
                            context_bundle["messages"],
                            system_blocks,
                            tools,
                            trimmed_node_ids=context_bundle["trimmed_node_ids"],
                            trimmed_system_blocks=system_budget["trimmed_system_blocks"],
                            budget=context_bundle["budget"],
                            tokens_before_trim=context_bundle["tokens_before_trim"],
                            tokens_after_trim=context_bundle["tokens_after_trim"],
                            non_system_tokens_before_trim=context_bundle["non_system_tokens_before_trim"],
                            non_system_tokens_after_trim=context_bundle["non_system_tokens_after_trim"],
                            too_long=context_bundle["too_long"],
                        )
                        | {"recovery_compact": True, "model": self.config.model.name},
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
                model=self.config.model.name,
                system=system_blocks,
                messages=context_bundle["messages"],
                tools=tools,
                temperature=self.config.model.temperature,
                max_output_tokens=self.config.model.max_output_tokens,
                metadata={"session_id": handle.session_id, "run_id": handle.run_id},
            )
            _ensure_provider_supports_request(self._provider, request)
            completed = None
            model_parent_node_id = user_node.node_id if end_node_id is None else end_node_id
            async for model_event in self._provider.stream(request):
                if model_event.event_type == "model_started":
                    await self._emit(handle, "model_started", payload=model_event.metadata)
                elif model_event.event_type == "model_text_delta":
                    partial_text_parts.append(model_event.text_delta or "")
                    await self._emit_realtime(handle, "model_text_delta", payload={"text": model_event.text_delta or ""})
                elif model_event.event_type == "model_failed":
                    error = model_event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="provider failed")
                    if partial_text_parts:
                        partial_node = await self.store.add_node(
                            session_id=handle.session_id,
                            parent_id=user_node.node_id if end_node_id is None else end_node_id,
                            agent_id=handle.agent_id,
                            run_id=handle.run_id,
                            role="assistant",
                            node_type="message",
                            content=[TextBlock(text="".join(partial_text_parts))],
                            metadata={"partial": True, "failed": True, "error": error.model_dump(mode="json")},
                            make_active=False,
                        )
                        end_node_id = partial_node.node_id
                    raise AgentCoreError(error.code, error.message, details=error.details)
                elif model_event.event_type == "model_completed":
                    completed = model_event
                    break
            if completed is None:
                raise AgentCoreError(ErrorCode.PROVIDER_ERROR, "provider stream ended without model_completed")
            assistant_content = list(completed.content)
            for call in completed.tool_calls:
                assistant_content.append(
                    ToolCallBlock(tool_call_id=call.tool_call_id, name=call.name, arguments=call.arguments, metadata=call.metadata)
                )
            recover_empty_tool_result = (
                not completed.tool_calls
                and not _content_has_text(assistant_content)
                and _last_message_is_tool_result(messages)
                and empty_tool_result_retries < 1
            )
            assistant_node = await self.store.add_node(
                session_id=handle.session_id,
                parent_id=user_node.node_id if end_node_id is None else end_node_id,
                agent_id=handle.agent_id,
                run_id=handle.run_id,
                role="assistant",
                node_type="message",
                content=assistant_content,
                metadata={"stop_reason": completed.stop_reason.value if completed.stop_reason else None},
                make_active=not completed.tool_calls and not recover_empty_tool_result,
            )
            end_node_id = assistant_node.node_id
            await self._persist_provider_debug_artifact(handle, completed, node_id=assistant_node.node_id)
            await self._emit(handle, "model_completed", node_id=assistant_node.node_id)
            messages.append(
                ModelMessage(
                    role=ModelRole.ASSISTANT,
                    content=assistant_content,
                    node_type="message",
                    metadata={"node_id": assistant_node.node_id},
                )
            )
            if not completed.tool_calls:
                if recover_empty_tool_result:
                    empty_tool_result_retries += 1
                    retry_text = (
                        "The previous tool results are available in context. "
                        "Provide the final answer now, using those tool results."
                    )
                    retry_node = await self.store.add_node(
                        session_id=handle.session_id,
                        parent_id=end_node_id,
                        agent_id=handle.agent_id,
                        run_id=handle.run_id,
                        role="user",
                        node_type="empty_tool_result_recovery",
                        content=[TextBlock(text=retry_text)],
                        metadata={"synthetic": True, "reason": "empty_model_response_after_tool_result"},
                        make_active=False,
                    )
                    messages.append(
                        ModelMessage(
                            role=ModelRole.USER,
                            content=[TextBlock(text=retry_text)],
                            node_type="empty_tool_result_recovery",
                            metadata={"node_id": retry_node.node_id, "synthetic": True},
                        )
                    )
                    end_node_id = retry_node.node_id
                    await self._emit(
                        handle,
                        "model_empty_response_recovered",
                        node_id=retry_node.node_id,
                        payload={"reason": "empty_model_response_after_tool_result"},
                    )
                    continue
                handle.status = RunStatus.COMPLETED
                stop_decision = await self._run_stop_hooks(handle, end_node_id=end_node_id)
                if stop_decision.denied:
                    await self._emit(
                        handle,
                        "stop_hook_prevented",
                        node_id=end_node_id,
                        payload={"reason": stop_decision.reason, "metadata": stop_decision.metadata},
                    )
                    note_text = f"Stop hook prevented completion. Reason: {stop_decision.reason or 'unspecified'}"
                    note_node = await self.store.add_node(
                        session_id=handle.session_id,
                        parent_id=end_node_id,
                        agent_id=handle.agent_id,
                        run_id=handle.run_id,
                        role="user",
                        node_type="hook_context",
                        content=[TextBlock(text=note_text)],
                        metadata={"hook_event": "Stop", "reason": stop_decision.reason, "metadata": stop_decision.metadata},
                        make_active=False,
                    )
                    messages.append(
                        ModelMessage(
                            role=ModelRole.USER,
                            content=[TextBlock(text=note_text)],
                            node_type="hook_context",
                            metadata={"node_id": note_node.node_id},
                        )
                    )
                    end_node_id = note_node.node_id
                    handle.status = RunStatus.RUNNING
                    continue
                await self.store.update_run(
                    run_id=handle.run_id,
                    status=RunStatus.COMPLETED.value,
                    end_node_id=end_node_id,
                    end_reason="completed",
                )
                await self._emit(handle, "run_completed", node_id=end_node_id)
                await self._emit(handle, "loop_completed", node_id=end_node_id)
                await self._maybe_run_memory_extraction(handle, prompt_text=prompt_text)
                await self._maybe_start_background_compact(handle)
                return
            tool_results = await self._execute_tool_calls(handle, completed.tool_calls)
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
                session_id=handle.session_id,
                parent_id=end_node_id,
                agent_id=handle.agent_id,
                run_id=handle.run_id,
                role="tool",
                node_type="message",
                content=tool_content,
                metadata={},
                make_active=False,
            )
            end_node_id = tool_node.node_id
            messages.append(
                ModelMessage(
                    role=ModelRole.TOOL,
                    content=tool_content,
                    node_type="message",
                    metadata={"node_id": tool_node.node_id},
                )
            )
            for synthetic in _synthetic_context_nodes_from_tool_results(tool_results):
                synthetic_node = await self.store.add_node(
                    session_id=handle.session_id,
                    parent_id=end_node_id,
                    agent_id=handle.agent_id,
                    run_id=handle.run_id,
                    role="user",
                    node_type=synthetic["node_type"],
                    content=[TextBlock(text=synthetic["text"])],
                    metadata=synthetic["metadata"],
                    make_active=False,
                )
                end_node_id = synthetic_node.node_id
                messages.append(
                    ModelMessage(
                        role=ModelRole.USER,
                        content=[TextBlock(text=synthetic["text"])],
                        node_type=synthetic["node_type"],
                        metadata={"node_id": synthetic_node.node_id, **synthetic["metadata"]},
                    )
                )
            refreshed_task_board_message = _task_board_context_message(self.task_service, handle.session_id)
            if refreshed_task_board_message is not None:
                messages = [message for message in messages if message.node_type != "task_board"]
                messages.append(refreshed_task_board_message)
        raise AgentCoreError(ErrorCode.INTERNAL_ERROR, "max turns exceeded")
    except asyncio.CancelledError:
        if partial_text_parts:
            partial_node = await self._persist_partial_assistant_node(
                handle,
                parent_id=model_parent_node_id,
                text="".join(partial_text_parts),
                metadata={"partial": True, "aborted": True, "abort_reason": "cancelled"},
            )
            end_node_id = partial_node.node_id
            await self._emit(
                handle,
                "aborted_streaming",
                node_id=partial_node.node_id,
                payload={"reason": "cancelled"},
            )
        handle.status = RunStatus.CANCELLED
        await self.store.update_run(
            run_id=handle.run_id,
            status=RunStatus.CANCELLED.value,
            end_node_id=end_node_id,
            end_reason="aborted_streaming" if partial_text_parts else "aborted_tools",
            error={"code": ErrorCode.CANCELLED.value, "message": "run cancelled", "reason": "cancelled"},
        )
        await self._emit(handle, "run_cancelled")
    except Exception as exc:
        handle.status = RunStatus.FAILED
        code = getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
        message_text = getattr(exc, "message", str(exc))
        end_reason = getattr(exc, "details", {}).get("end_reason") if isinstance(getattr(exc, "details", None), dict) else None
        await self.store.update_run(
            run_id=handle.run_id,
            status=RunStatus.FAILED.value,
            end_node_id=end_node_id,
            end_reason=end_reason or "failed",
            error={"code": str(code), "message": message_text},
        )
        payload = {"code": str(code), "message": message_text}
        if end_reason:
            payload["end_reason"] = end_reason
        await self._emit(handle, "loop_failed", level="error", payload=payload)
    finally:
        self._session_active.pop(handle.session_id, None)
        await handle._stream.close()
        await self._start_next_queued(handle.session_id)


def _directive_system_blocks(handle: RunHandle) -> list[Any]:
    directives = handle.directives or {}
    mentioned = directives.get("mentioned_worker") if isinstance(directives, dict) else None
    if not isinstance(mentioned, dict) or not mentioned.get("worker_id"):
        return []
    from agent_core.providers import SystemBlock

    worker_id = str(mentioned.get("worker_id"))
    worker_agent_id = str(mentioned.get("worker_agent_id") or "")
    worker_pool_id = str(mentioned.get("worker_pool_id") or "")
    name = str(mentioned.get("name") or worker_id)
    content = (
        "The user explicitly mentioned a worker for this run.\n"
        f"- Worker id: {worker_id}\n"
        f"- Worker agent id: {worker_agent_id}\n"
        f"- Worker pool id: {worker_pool_id}\n"
        f"- Worker name: {name}\n\n"
        "If you dispatch work with agent.dispatch_worker, dispatch only to this worker. "
        "Do not dispatch to a different worker. If no worker_agent_id is provided, the tool will use this worker automatically."
    )
    return [
        SystemBlock(
            block_id="run_directive.mentioned_worker",
            source="run_directive",
            content=content,
            priority=850,
            dynamic=True,
            metadata={"mentioned_worker": mentioned},
        )
    ]
