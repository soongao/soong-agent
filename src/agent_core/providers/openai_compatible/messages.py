from __future__ import annotations

import json
from typing import Any

from agent_core.providers.base import ModelMessage, ModelRole
from agent_core.providers.message_parts import message_text, tool_result_text
from agent_core.providers.tool_mapping import to_provider_tool_name


def openai_messages(message: ModelMessage) -> list[dict[str, Any]]:
    if message.role == ModelRole.ASSISTANT:
        tool_calls = []
        for block in message.content:
            if getattr(block, "type", None) != "tool_call":
                continue
            tool_calls.append(
                {
                    "id": getattr(block, "tool_call_id"),
                    "type": "function",
                    "function": {
                        "name": to_provider_tool_name(getattr(block, "name")),
                        "arguments": json.dumps(getattr(block, "arguments", {}), ensure_ascii=False),
                    },
                }
            )
        payload: dict[str, Any] = {"role": "assistant", "content": message_text(message.content) or None}
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return [payload]
    if message.role == ModelRole.TOOL:
        return [
            {
                "role": "tool",
                "tool_call_id": getattr(block, "tool_call_id"),
                "name": to_provider_tool_name(str(getattr(block, "metadata", {}).get("tool_name") or "")) or None,
                "content": tool_result_text(block),
            }
            for block in message.content
            if getattr(block, "type", None) == "tool_result"
        ]
    return [{"role": openai_role(message.role), "content": message_text(message.content)}]


def openai_role(role: ModelRole) -> str:
    if role == ModelRole.TOOL:
        return "tool"
    return role.value
