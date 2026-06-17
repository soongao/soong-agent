from __future__ import annotations

from typing import Any

from agent_core.providers.base import ModelRequest
from agent_core.providers.openai_compatible.messages import openai_messages
from agent_core.providers.tool_mapping import to_provider_tool_name


def build_openai_chat_payload(request: ModelRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": "\n\n".join(block.content for block in request.system)})
    for message in request.messages:
        messages.extend(openai_messages(message))
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "temperature": request.temperature,
        "max_tokens": request.max_output_tokens,
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": to_provider_tool_name(tool.name),
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = openai_tool_choice(request.tool_choice)
    return payload


def openai_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice = dict(tool_choice)
    function = choice.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        choice["function"] = {**function, "name": to_provider_tool_name(function["name"])}
    return choice
