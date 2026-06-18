from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import os
from typing import Any

import httpx

from agent_core.errors.codes import ErrorCode
from agent_core.providers.base import ModelEvent, ModelRequest, ProviderAdapter, StopReason
from agent_core.providers.errors import classify_provider_exception, retry_delay_seconds
from agent_core.providers.openai_compatible.options import failed_config_event, openai_provider_options
from agent_core.providers.openai_compatible.payload import build_openai_chat_payload
from agent_core.providers.openai_compatible.stream import OpenAIToolAccumulator, openai_chunk_to_events
from agent_core.types.common import ErrorPayload
from agent_core.types.content import TextBlock


class OpenAICompatibleProvider(ProviderAdapter):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.base_url = (getattr(config, "base_url", None) or "").rstrip("/")
        self.model = getattr(config, "name", "")
        self.api_key = getattr(config, "api_key", None) or ""
        self.api_key_env = getattr(config, "api_key_env", "") or ""
        self.retry = getattr(config, "retry", None)
        timeout_ms = getattr(config, "timeout_ms", 60000) or 60000
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_ms / 1000))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        provider_options, option_error = openai_provider_options(request)
        if option_error is not None:
            yield failed_config_event(option_error)
            return
        if not self.base_url:
            yield failed_config_event("openai base_url is required")
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
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
