from __future__ import annotations

import json
import os
import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_core.errors.codes import ErrorCode
from agent_core.providers.base import ModelEvent, ModelRequest, ModelRole, ProviderAdapter, StopReason, Usage
from agent_core.providers.errors import classify_provider_exception, retry_delay_seconds
from agent_core.providers.tool_mapping import from_provider_tool_name, to_provider_tool_name
from agent_core.types.common import ErrorPayload
from agent_core.types.content import TextBlock
from agent_core.types.tools import ToolCall


class OpenAICompatibleProvider(ProviderAdapter):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.base_url = (getattr(config, "base_url", None) or "").rstrip("/")
        self.model = getattr(config, "name", "")
        self.api_key_env = getattr(config, "api_key_env", "") or ""
        self.retry = getattr(config, "retry", None)
        timeout_ms = getattr(config, "timeout_ms", 60000) or 60000
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_ms / 1000))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        provider_options, option_error = _openai_provider_options(request)
        if option_error is not None:
            yield _failed(option_error)
            return
        if not self.base_url:
            yield _failed("openai base_url is required")
            return
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                yield ModelEvent(
                    event_type="model_failed",
                    error=ErrorPayload(
                        code=ErrorCode.PROVIDER_AUTH_FAILED,
                        message=f"missing environment variable: {self.api_key_env}",
                    ),
                )
                return
            headers["Authorization"] = f"Bearer {api_key}"
        payload = build_openai_chat_payload(request)
        payload.update(provider_options)
        known_names = {tool.name for tool in request.tools}
        yield ModelEvent(event_type="model_started", metadata={"provider": "openai"})
        state = OpenAIToolAccumulator(known_names=known_names)
        text_parts: list[str] = []
        attempts = max(getattr(self.retry, "max_attempts", 1) or 1, 1)
        retry_count = 0
        for attempt in range(1, attempts + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            break
                        for event in openai_chunk_to_events(json.loads(data), state):
                            if event.event_type == "model_text_delta" and event.text_delta:
                                text_parts.append(event.text_delta)
                            yield event
                    break
            except Exception as exc:
                info = classify_provider_exception(exc)
                if not info.retryable or attempt >= attempts:
                    yield ModelEvent(
                        event_type="model_failed",
                        error=info.payload(),
                        metadata={"provider": "openai", "retry_count": retry_count},
                    )
                    return
                retry_count += 1
                await asyncio.sleep(retry_delay_seconds(attempt_index=attempt, retry=self.retry, retry_after_ms=info.retry_after_ms))
        yield ModelEvent(
            event_type="model_completed",
            content=[TextBlock(text="".join(text_parts))] if text_parts else [],
            tool_calls=state.tool_calls(),
            stop_reason=StopReason.TOOL_USE if state.tool_calls() else StopReason.END_TURN,
            metadata={"provider": "openai", "retry_count": retry_count},
        )

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIToolAccumulator:
    def __init__(self, *, known_names: set[str]) -> None:
        self.known_names = known_names
        self._calls: dict[int, dict[str, Any]] = {}

    def add_delta(self, raw: dict[str, Any]) -> None:
        index = int(raw.get("index", 0))
        call = self._calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
        if raw.get("id"):
            call["id"] = raw["id"]
        fn = raw.get("function") or {}
        if fn.get("name"):
            call["name"] += fn["name"]
        if fn.get("arguments"):
            call["arguments"] += fn["arguments"]

    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, raw in sorted(self._calls.items()):
            raw_name = raw.get("name") or ""
            name = from_provider_tool_name(raw_name, self.known_names)
            try:
                arguments = json.loads(raw.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    tool_call_id=raw.get("id") or f"openai_tool_{index}",
                    name=name,
                    arguments=arguments,
                    metadata={"raw_name": raw_name},
                )
            )
        return calls


def build_openai_chat_payload(request: ModelRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": "\n\n".join(block.content for block in request.system)})
    for message in request.messages:
        messages.extend(_openai_messages(message))
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
        payload["tool_choice"] = _openai_tool_choice(request.tool_choice)
    return payload


def _openai_provider_options(request: ModelRequest) -> tuple[dict[str, Any], str | None]:
    if not request.provider_options:
        return {}, None
    unknown_namespaces = sorted(key for key in request.provider_options if key != "openai")
    if unknown_namespaces:
        return {}, f"unknown provider_options namespace for openai: {', '.join(unknown_namespaces)}"
    options = request.provider_options.get("openai") or {}
    if not isinstance(options, dict):
        return {}, "provider_options.openai must be an object"
    allowed_keys = {"response_format", "seed", "parallel_tool_calls"}
    unknown_keys = sorted(key for key in options if key not in allowed_keys)
    if unknown_keys:
        return {}, f"unsupported openai provider_options: {', '.join(unknown_keys)}"
    return dict(options), None


def openai_chunk_to_events(chunk: dict[str, Any], state: OpenAIToolAccumulator) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content"):
            events.append(ModelEvent(event_type="model_text_delta", text_delta=delta["content"]))
        for raw_call in delta.get("tool_calls") or []:
            state.add_delta(raw_call)
            events.append(ModelEvent(event_type="tool_call_delta", tool_call_delta=raw_call))
    return events


def _openai_messages(message: ModelMessage) -> list[dict[str, Any]]:
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
        payload: dict[str, Any] = {"role": "assistant", "content": _message_text(message.content) or None}
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return [payload]
    if message.role == ModelRole.TOOL:
        return [
            {
                "role": "tool",
                "tool_call_id": getattr(block, "tool_call_id"),
                "name": to_provider_tool_name(str(getattr(block, "metadata", {}).get("tool_name") or "")) or None,
                "content": _tool_result_text(block),
            }
            for block in message.content
            if getattr(block, "type", None) == "tool_result"
        ]
    return [{"role": _openai_role(message.role), "content": _message_text(message.content)}]


def _message_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        elif getattr(block, "type", None) == "json":
            parts.append(json.dumps(getattr(block, "data", None), ensure_ascii=False))
        elif getattr(block, "type", None) == "artifact_ref":
            parts.append(f"[artifact:{getattr(block, 'artifact_id', '')}] {getattr(block, 'summary', '') or ''}".strip())
    return "\n".join(part for part in parts if part)


def _tool_result_text(block: Any) -> str:
    if getattr(block, "is_error", False) and getattr(block, "error", None) is not None:
        error = getattr(block, "error")
        return json.dumps({"error": error.model_dump(mode="json")}, ensure_ascii=False)
    text = _message_text(getattr(block, "content", []) or [])
    if text:
        return text
    return json.dumps(getattr(block, "metadata", {}) or {}, ensure_ascii=False)


def _openai_role(role: ModelRole) -> str:
    if role == ModelRole.TOOL:
        return "tool"
    return role.value


def _openai_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice = dict(tool_choice)
    function = choice.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        choice["function"] = {**function, "name": to_provider_tool_name(function["name"])}
    return choice


def _failed(message: str) -> ModelEvent:
    return ModelEvent(event_type="model_failed", error=ErrorPayload(code=ErrorCode.CONFIG_ERROR, message=message))
