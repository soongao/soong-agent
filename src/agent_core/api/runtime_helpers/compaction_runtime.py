from __future__ import annotations

import asyncio
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.api.runtime_helpers.context import _apply_context_budget, _estimate_message_tokens, _estimate_system_tokens
from agent_core.api.runtime_helpers.views import _task_board_context_message
from agent_core.config.models import ContextConfig
from agent_core.context import build_context_messages
from agent_core.errors import AgentCoreError
from agent_core.events import make_event
from agent_core.providers import ModelMessage, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.types import Node, ToolDefinition


async def maybe_start_background_compact(runtime: Any, handle: RunHandle) -> None:
    assert runtime.config and runtime.store
    if not runtime.config.compact.enabled or not runtime.config.compact.auto_background:
        return
    replay = await runtime.replay_session(handle.session_id)
    text = "\n".join(
        getattr(block, "text", "")
        for node in replay.nodes
        for block in node.content
        if getattr(block, "type", None) == "text"
    )
    estimated_tokens = max(len(text) // 4, 0)
    threshold = max(runtime.config.model.context_window - runtime.config.compact.reserve_tokens, runtime.config.compact.keep_recent_tokens)
    if estimated_tokens < threshold:
        return
    await runtime.store.add_event(
        make_event(
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            event_type="compact_pending",
            payload={"estimated_tokens": estimated_tokens, "threshold": threshold},
        )
    )
    asyncio.create_task(runtime.run_compact_agent(session_id=handle.session_id, reason="auto_background"))


async def try_recovery_compact(
    runtime: Any,
    handle: RunHandle,
    *,
    user_node: Node,
    messages: list[ModelMessage],
    system_blocks: list[SystemBlock],
    context_config: ContextConfig,
    tools: list[ToolDefinition],
) -> dict[str, Any] | None:
    assert runtime.config and runtime.store
    if not runtime.config.compact.enabled or not runtime.config.compact.recovery_sync:
        return None
    budget = context_config.non_system_budget
    if budget is None:
        budget = max(runtime.config.model.context_window - runtime.config.model.max_output_tokens - _estimate_system_tokens(system_blocks), 0)
    latest_tokens = _estimate_message_tokens(
        ModelMessage(role=ModelRole.USER, content=user_node.content, node_type=user_node.node_type, metadata={"node_id": user_node.node_id})
    )
    if latest_tokens > max(budget, 0):
        return None
    replay = await runtime.replay_session(handle.session_id)
    source_node_ids = [
        node.node_id
        for node in replay.nodes
        if node.node_id != user_node.node_id
        and node.node_type not in {"compaction", "compact_input"}
        and node.role in {"user", "assistant", "tool"}
    ]
    if not source_node_ids:
        return None
    try:
        await runtime._emit(
            handle,
            "compact_pending",
            payload={"reason": "recovery_sync", "source_node_ids": source_node_ids},
        )
        result = await runtime.run_compact_agent(
            session_id=handle.session_id,
            source_node_ids=source_node_ids,
            reason="recovery_sync",
            first_kept_node_id=user_node.parent_id,
        )
    except AgentCoreError as exc:
        await runtime._emit(
            handle,
            "compact_failed",
            level="warning",
            payload={"reason": "recovery_sync", "code": exc.code.value, "message": exc.message},
        )
        return None
    if result.get("stale") or not result.get("compaction_node_id"):
        return None
    active_path = await runtime.store.get_node_path(str(result["compaction_node_id"]))
    recovered_messages = build_context_messages(active_path)
    task_board_message = _task_board_context_message(runtime.task_service, handle.session_id)
    if task_board_message is not None:
        recovered_messages.append(task_board_message)
    context_bundle = _apply_context_budget(
        messages=recovered_messages,
        system_blocks=system_blocks,
        context_config=context_config,
        model_config=runtime.config.model,
    )
    return {"messages": recovered_messages, "context_bundle": context_bundle, "end_node_id": result["compaction_node_id"]}
