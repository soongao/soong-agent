from __future__ import annotations

import json
import os
import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_core.errors.codes import ErrorCode
from agent_core.providers.base import ModelEvent, ModelRequest, ModelRole, ProviderAdapter, StopReason
from agent_core.providers.errors import classify_provider_exception, retry_delay_seconds
from agent_core.providers.tool_mapping import from_provider_tool_name, to_provider_tool_name
from agent_core.types.common import ErrorPayload
from agent_core.types.content import TextBlock
from agent_core.types.tools import ToolCall


class AnthropicProvider(ProviderAdapter):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.base_url = (getattr(config, "base_url", None) or "https://api.anthropic.com").rstrip("/")
        self.model = getattr(config, "name", "")
        self.api_key_env = getattr(config, "api_key_env", "") or ""
        self.retry = getattr(config, "retry", None)
        timeout_ms = getattr(config, "timeout_ms", 60000) or 60000
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_ms / 1000))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        provider_options, option_error = _anthropic_provider_options(request)
        if option_error is not None:
            yield _failed(option_error)
            return
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
        else:
            api_key = ""
        payload = build_anthropic_payload(request)
        known_names = {tool.name for tool in request.tools}
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        headers.update(provider_options.pop("headers", {}))
        payload.update(provider_options)
        yield ModelEvent(event_type="model_started", metadata={"provider": "anthropic"})
        state = AnthropicToolAccumulator(known_names=known_names)
        text_parts: list[str] = []
        attempts = max(getattr(self.retry, "max_attempts", 1) or 1, 1)
        retry_count = 0
        for attempt in range(1, attempts + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    event_name = None
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            event_name = line.removeprefix("event:").strip()
                        elif line.startswith("data:"):
                            data = json.loads(line.removeprefix("data:").strip())
                            for event in anthropic_sse_to_events(event_name or data.get("type", ""), data, state):
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
                        metadata={"provider": "anthropic", "retry_count": retry_count},
                    )
                    return
                retry_count += 1
                await asyncio.sleep(retry_delay_seconds(attempt_index=attempt, retry=self.retry, retry_after_ms=info.retry_after_ms))
        yield ModelEvent(
            event_type="model_completed",
            content=[TextBlock(text="".join(text_parts))] if text_parts else [],
            tool_calls=state.tool_calls(),
            stop_reason=StopReason.TOOL_USE if state.tool_calls() else StopReason.END_TURN,
            metadata={"provider": "anthropic", "retry_count": retry_count},
        )

    async def close(self) -> None:
        await self._client.aclose()


class AnthropicToolAccumulator:
    def __init__(self, *, known_names: set[str]) -> None:
        self.known_names = known_names
        self._blocks: dict[int, dict[str, Any]] = {}

    def start_block(self, index: int, block: dict[str, Any]) -> None:
        if block.get("type") == "tool_use":
            self._blocks[index] = {
                "id": block.get("id"),
                "name": block.get("name", ""),
                "input_json": "",
                "input": block.get("input") if isinstance(block.get("input"), dict) else None,
            }

    def add_delta(self, index: int, delta: dict[str, Any]) -> None:
        if index not in self._blocks:
            return
        if delta.get("partial_json"):
            self._blocks[index]["input_json"] += delta["partial_json"]

    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, raw in sorted(self._blocks.items()):
            arguments = raw.get("input")
            if arguments is None:
                try:
                    arguments = json.loads(raw.get("input_json") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
            raw_name = raw.get("name") or ""
            calls.append(
                ToolCall(
                    tool_call_id=raw.get("id") or f"anthropic_tool_{index}",
                    name=from_provider_tool_name(raw_name, self.known_names),
                    arguments=arguments,
                    metadata={"raw_name": raw_name},
                )
            )
        return calls


def build_anthropic_payload(request: ModelRequest) -> dict[str, Any]:
    messages = []
    for message in request.messages:
        if message.role == ModelRole.SYSTEM:
            continue
        messages.append(_anthropic_message(message))
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "system": "\n\n".join(block.content for block in request.system) if request.system else None,
        "stream": True,
        "max_tokens": request.max_output_tokens or 4096,
        "temperature": request.temperature,
    }
    if request.tools:
        payload["tools"] = [
            {
                "name": to_provider_tool_name(tool.name),
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ]
    return {key: value for key, value in payload.items() if value is not None}


def _anthropic_provider_options(request: ModelRequest) -> tuple[dict[str, Any], str | None]:
    if not request.provider_options:
        return {}, None
    unknown_namespaces = sorted(key for key in request.provider_options if key != "anthropic")
    if unknown_namespaces:
        return {}, f"unknown provider_options namespace for anthropic: {', '.join(unknown_namespaces)}"
    options = request.provider_options.get("anthropic") or {}
    if not isinstance(options, dict):
        return {}, "provider_options.anthropic must be an object"
    allowed_keys = {"metadata", "stop_sequences", "thinking", "headers"}
    unknown_keys = sorted(key for key in options if key not in allowed_keys)
    if unknown_keys:
        return {}, f"unsupported anthropic provider_options: {', '.join(unknown_keys)}"
    if "headers" in options and not isinstance(options["headers"], dict):
        return {}, "provider_options.anthropic.headers must be an object"
    return dict(options), None


def anthropic_sse_to_events(event_name: str, data: dict[str, Any], state: AnthropicToolAccumulator) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    event_type = data.get("type") or event_name
    if event_type == "content_block_start":
        index = int(data.get("index", 0))
        state.start_block(index, data.get("content_block") or {})
    elif event_type == "content_block_delta":
        index = int(data.get("index", 0))
        delta = data.get("delta") or {}
        if delta.get("type") == "text_delta" and delta.get("text"):
            events.append(ModelEvent(event_type="model_text_delta", text_delta=delta["text"]))
        elif delta.get("type") == "input_json_delta":
            state.add_delta(index, delta)
            events.append(ModelEvent(event_type="tool_call_delta", tool_call_delta=data))
    return events


def _anthropic_message(message: ModelMessage) -> dict[str, Any]:
    if message.role == ModelRole.ASSISTANT:
        content: list[dict[str, Any]] = []
        text = _message_text(message.content)
        if text:
            content.append({"type": "text", "text": text})
        for block in message.content:
            if getattr(block, "type", None) != "tool_call":
                continue
            content.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "tool_call_id"),
                    "name": to_provider_tool_name(getattr(block, "name")),
                    "input": getattr(block, "arguments", {}),
                }
            )
        return {"role": "assistant", "content": content or ""}
    if message.role == ModelRole.TOOL:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_call_id"),
                "content": _tool_result_text(block),
                "is_error": bool(getattr(block, "is_error", False)),
            }
            for block in message.content
            if getattr(block, "type", None) == "tool_result"
        ]
        return {"role": "user", "content": content}
    return {"role": "user", "content": _message_text(message.content)}


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


def _failed(message: str) -> ModelEvent:
    return ModelEvent(event_type="model_failed", error=ErrorPayload(code=ErrorCode.CONFIG_ERROR, message=message))
