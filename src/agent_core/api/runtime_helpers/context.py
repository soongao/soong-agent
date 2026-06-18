from __future__ import annotations

import json
from typing import Any

from agent_core.config.models import ContextConfig, ModelConfig
from agent_core.providers import ModelMessage, SystemBlock
from agent_core.providers.base import ModelRole
from agent_core.types import ToolDefinition


def _apply_context_budget(
    *,
    messages: list[ModelMessage],
    system_blocks: list[SystemBlock],
    context_config: ContextConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    tokens_before = _estimate_model_messages_tokens(messages, system_blocks)
    message_tokens_before = sum(_estimate_message_tokens(message) for message in messages)
    budget = context_config.non_system_budget
    if budget is None:
        system_tokens = _estimate_system_tokens(system_blocks)
        budget = max(model_config.context_window - model_config.max_output_tokens - system_tokens, 0)
    if budget <= 0:
        tokens_after = _estimate_model_messages_tokens(messages, system_blocks)
        return {
            "messages": messages,
            "trimmed_node_ids": [],
            "budget": budget,
            "tokens_before_trim": tokens_before,
            "tokens_after_trim": tokens_after,
            "non_system_tokens_before_trim": message_tokens_before,
            "non_system_tokens_after_trim": message_tokens_before,
            "too_long": message_tokens_before > max(budget, 0),
        }

    protected_start = _protected_message_suffix_start(messages)
    retained_reversed: list[ModelMessage] = []
    retained_tokens = 0
    trimmed_node_ids: list[str] = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        message_tokens = _estimate_message_tokens(message)
        node_id = _message_node_id(message)
        if (
            retained_reversed
            and retained_tokens + message_tokens > budget
            and index < protected_start
            and _message_can_trim(message)
        ):
            if node_id is not None:
                trimmed_node_ids.append(node_id)
            continue
        retained_reversed.append(message)
        retained_tokens += message_tokens
    retained = list(reversed(retained_reversed))
    trimmed_node_ids.reverse()
    tokens_after = _estimate_model_messages_tokens(retained, system_blocks)
    message_tokens_after = sum(_estimate_message_tokens(message) for message in retained)
    return {
        "messages": retained,
        "trimmed_node_ids": trimmed_node_ids,
        "budget": budget,
        "tokens_before_trim": tokens_before,
        "tokens_after_trim": tokens_after,
        "non_system_tokens_before_trim": message_tokens_before,
        "non_system_tokens_after_trim": message_tokens_after,
        "too_long": message_tokens_after > budget,
    }


def _apply_system_block_budget(system_blocks: list[SystemBlock], dynamic_system_budget: int | None) -> dict[str, Any]:
    if dynamic_system_budget is None or dynamic_system_budget <= 0:
        return {"system_blocks": system_blocks, "trimmed_system_blocks": []}
    dynamic_blocks = sorted((block for block in system_blocks if block.dynamic), key=lambda block: block.priority, reverse=True)
    retained_dynamic: list[SystemBlock] = []
    trimmed: list[dict[str, Any]] = []
    used = 0
    for block in dynamic_blocks:
        tokens = block.token_count if block.token_count is not None else max(len(block.content) // 4, 0)
        if used + tokens > dynamic_system_budget:
            trimmed.append(
                {
                    "block_id": block.block_id,
                    "source": block.source,
                    "priority": block.priority,
                    "estimated_tokens": tokens,
                }
            )
            continue
        retained_dynamic.append(block)
        used += tokens
    retained_ids = {id(block) for block in retained_dynamic}
    ordered = [block for block in system_blocks if not block.dynamic or id(block) in retained_ids]
    return {"system_blocks": ordered, "trimmed_system_blocks": trimmed}


def _protected_message_suffix_start(messages: list[ModelMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == ModelRole.USER and message.node_type in {"message", "hook_context", "worker_dispatch"}:
            return index
    return max(len(messages) - 1, 0)


def _message_can_trim(message: ModelMessage) -> bool:
    if isinstance(message.metadata, dict) and message.metadata.get("synthetic") is True:
        return False
    return message.node_type not in {"task_board", "compaction"}


def _message_node_id(message: ModelMessage) -> str | None:
    if isinstance(message.metadata, dict) and message.metadata.get("node_id") is not None:
        return str(message.metadata["node_id"])
    return None


def _context_build_report(
    messages: list[ModelMessage],
    system_blocks: list[SystemBlock],
    tools: list[ToolDefinition],
    *,
    trimmed_node_ids: list[str] | None = None,
    trimmed_system_blocks: list[dict[str, Any]] | None = None,
    budget: int | None = None,
    tokens_before_trim: int | None = None,
    tokens_after_trim: int | None = None,
    non_system_tokens_before_trim: int | None = None,
    non_system_tokens_after_trim: int | None = None,
    too_long: bool = False,
) -> dict[str, Any]:
    retained_node_ids = [
        str(message.metadata["node_id"])
        for message in messages
        if isinstance(message.metadata, dict) and message.metadata.get("node_id") is not None
    ]
    synthetic_messages = [
        {
            "node_type": message.node_type,
            "source": message.metadata.get("source"),
        }
        for message in messages
        if isinstance(message.metadata, dict) and message.metadata.get("synthetic") is True
    ]
    return {
        "model": None,
        "messages": len(messages),
        "tools": len(tools),
        "tool_names": [tool.name for tool in tools],
        "system_blocks": [
            {
                "block_id": block.block_id,
                "source": block.source,
                "dynamic": block.dynamic,
                "priority": block.priority,
            }
            for block in system_blocks
        ],
        "retained_node_ids": retained_node_ids,
        "trimmed_node_ids": trimmed_node_ids or [],
        "trimmed_system_blocks": trimmed_system_blocks or [],
        "synthetic_messages": synthetic_messages,
        "estimated_input_tokens": _estimate_model_messages_tokens(messages, system_blocks),
        "tokens_before_trim": tokens_before_trim,
        "tokens_after_trim": tokens_after_trim,
        "non_system_tokens_before_trim": non_system_tokens_before_trim,
        "non_system_tokens_after_trim": non_system_tokens_after_trim,
        "non_system_budget": budget,
        "too_long": too_long,
    }


def _estimate_model_messages_tokens(messages: list[ModelMessage], system_blocks: list[SystemBlock]) -> int:
    return _estimate_system_tokens(system_blocks) + sum(_estimate_message_tokens(message) for message in messages)


def _estimate_system_tokens(system_blocks: list[SystemBlock]) -> int:
    return sum(max(len(block.content) // 4, 0) for block in system_blocks)


def _estimate_message_tokens(message: ModelMessage) -> int:
    char_count = 0
    for block in message.content:
        if getattr(block, "type", None) == "text":
            char_count += len(getattr(block, "text", ""))
        elif getattr(block, "type", None) == "json":
            char_count += len(json.dumps(getattr(block, "data", None), ensure_ascii=False))
        elif getattr(block, "type", None) == "tool_call":
            char_count += len(json.dumps(getattr(block, "arguments", {}), ensure_ascii=False))
        elif getattr(block, "type", None) == "tool_result":
            char_count += len(json.dumps(getattr(block, "metadata", {}), ensure_ascii=False))
    return max(char_count // 4, 1)
