from __future__ import annotations

import json
import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_core.errors.codes import ErrorCode
from agent_core.providers.base import ModelEvent, ModelRequest, ProviderAdapter, StopReason, Usage
from agent_core.providers.errors import classify_provider_exception, retry_delay_seconds
from agent_core.providers.tool_mapping import from_provider_tool_name, to_provider_tool_name
from agent_core.types.common import ErrorPayload
from agent_core.types.content import TextBlock
from agent_core.types.tools import ToolCall


class OllamaProvider(ProviderAdapter):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.base_url = (getattr(config, "base_url", None) or "http://127.0.0.1:11434").rstrip("/")
        self.model = getattr(config, "name", None)
        self.retry = getattr(config, "retry", None)
        self.timeout = httpx.Timeout((getattr(config, "timeout_ms", 60000) or 60000) / 1000)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": "\n\n".join(block.content for block in request.system)})
        for message in request.messages:
            messages.extend(_ollama_messages(message))
        tools = []
        known_names = {tool.name for tool in request.tools}
        for tool in request.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": to_provider_tool_name(tool.name),
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        yield ModelEvent(event_type="model_started", metadata={"provider": "ollama"})
        attempts = max(getattr(self.retry, "max_attempts", 1) or 1, 1)
        retry_count = 0
        for attempt in range(1, attempts + 1):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            try:
                async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        message = data.get("message") or {}
                        content = message.get("content")
                        if content:
                            text_parts.append(content)
                            yield ModelEvent(event_type="model_text_delta", text_delta=content)
                        for index, raw_call in enumerate(message.get("tool_calls") or []):
                            fn = raw_call.get("function") or {}
                            name = fn.get("name")
                            args = fn.get("arguments") or {}
                            if name:
                                tool_calls.append(
                                    ToolCall(
                                        tool_call_id=f"ollama_tool_{len(tool_calls) + index}",
                                        name=from_provider_tool_name(name, known_names),
                                        arguments=args if isinstance(args, dict) else {},
                                        metadata={"raw": raw_call, "raw_name": name},
                                    )
                                )
                        if data.get("done"):
                            usage = Usage(
                                input_tokens=data.get("prompt_eval_count"),
                                output_tokens=data.get("eval_count"),
                                total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
                                raw_usage=data,
                            )
                            stop_reason = StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
                            yield ModelEvent(
                                event_type="model_completed",
                                content=[TextBlock(text="".join(text_parts))] if text_parts else [],
                                tool_calls=tool_calls,
                                stop_reason=stop_reason,
                                usage=usage,
                                metadata={"provider": "ollama", "raw_done": data, "retry_count": retry_count},
                            )
                            return
                    return
            except Exception as exc:
                info = classify_provider_exception(exc)
                if not info.retryable or attempt >= attempts:
                    yield ModelEvent(
                        event_type="model_failed",
                        error=info.payload(),
                        metadata={"provider": "ollama", "retry_count": retry_count},
                    )
                    return
                retry_count += 1
                await asyncio.sleep(retry_delay_seconds(attempt_index=attempt, retry=self.retry, retry_after_ms=info.retry_after_ms))

    async def close(self) -> None:
        await self._client.aclose()


def _ollama_messages(message: Any) -> list[dict[str, Any]]:
    if message.role.value == "assistant":
        tool_calls = []
        for block in message.content:
            if getattr(block, "type", None) != "tool_call":
                continue
            tool_calls.append(
                {
                    "function": {
                        "name": to_provider_tool_name(getattr(block, "name")),
                        "arguments": getattr(block, "arguments", {}),
                    }
                }
            )
        payload: dict[str, Any] = {"role": "assistant", "content": _message_text(message.content)}
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return [payload]
    if message.role.value == "tool":
        return [
            {
                "role": "tool",
                "name": to_provider_tool_name(str(getattr(block, "metadata", {}).get("tool_name") or "")),
                "content": _tool_result_text(block),
            }
            for block in message.content
            if getattr(block, "type", None) == "tool_result"
        ]
    return [{"role": message.role.value, "content": _message_text(message.content)}]


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
