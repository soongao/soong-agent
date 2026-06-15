from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from typing import Any

import httpx
import pytest

from agent_core.config.models import ModelConfig
from agent_core.providers.ollama import OllamaProvider
from agent_core.providers.registry import ProviderRegistry
from agent_core.providers.tool_mapping import to_provider_tool_name
from agent_core.types.tools import ToolCall


Responder = Callable[[dict[str, Any], int], "ScriptedOllamaResponse"]
Blocker = Callable[[], Awaitable[None]]


@dataclass
class ScriptedOllamaResponse:
    status: int = 200
    lines: list[dict[str, Any] | str | bytes] = field(default_factory=list)
    pre_block: Blocker | None = None
    block: Blocker | None = None
    block_after_lines: int | None = None
    started: Callable[[], None] | None = None


class ScriptedOllama:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses: deque[ScriptedOllamaResponse | Responder] = deque()
        self.base_url = "http://scripted-ollama.test"

    def enqueue(self, response: ScriptedOllamaResponse | Responder) -> None:
        self._responses.append(response)

    def enqueue_text(self, text: str, *, block: Blocker | None = None, started: Callable[[], None] | None = None) -> None:
        self.enqueue(text_response(text, block=block, started=started))

    def enqueue_streaming_text(
        self,
        text: str,
        *,
        pre_block: Blocker | None = None,
        block_after_delta: Blocker | None = None,
        started: Callable[[], None] | None = None,
    ) -> None:
        self.enqueue(text_response(text, pre_block=pre_block, block=block_after_delta, block_after_lines=1, started=started))

    def enqueue_tool_calls(
        self,
        calls: list[ToolCall],
        *,
        block: Blocker | None = None,
        started: Callable[[], None] | None = None,
    ) -> None:
        self.enqueue(tool_call_response(calls, block=block, started=started))

    def enqueue_failure_after_delta(self, text: str) -> None:
        self.enqueue(ScriptedOllamaResponse(lines=[{"message": {"content": text}, "done": False}, b"{malformed json"]))

    def provider_registry(self) -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register("ollama", lambda config: _ScriptedOllamaProvider(config, self._handle_request))
        return registry

    def _next_response(self, payload: dict[str, Any]) -> ScriptedOllamaResponse:
        self.requests.append(payload)
        index = len(self.requests) - 1
        response = self._responses.popleft() if self._responses else text_response("done")
        if callable(response):
            return response(payload, index)
        return response

    async def _handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, request=request, json={"models": [{"name": "gemma4"}]})
        if request.url.path != "/api/chat":
            return httpx.Response(404, request=request)
        payload = json.loads(request.content.decode("utf-8") or "{}")
        response = self._next_response(payload)
        if response.started is not None:
            response.started()
        if response.pre_block is not None:
            await response.pre_block()
        if response.block is not None and response.block_after_lines is None:
            await response.block()
        stream = _ScriptedStream(response.lines, block=response.block, block_after_lines=response.block_after_lines)
        return httpx.Response(response.status, request=request, stream=stream)


def text_response(
    text: str,
    *,
    pre_block: Blocker | None = None,
    block: Blocker | None = None,
    block_after_lines: int | None = None,
    started: Callable[[], None] | None = None,
) -> ScriptedOllamaResponse:
    return ScriptedOllamaResponse(
        lines=[
            {"message": {"content": text}, "done": False},
            {"message": {}, "done": True, "prompt_eval_count": 1, "eval_count": 1},
        ],
        pre_block=pre_block,
        block=block,
        block_after_lines=block_after_lines,
        started=started,
    )


def tool_call_response(
    calls: list[ToolCall],
    *,
    block: Blocker | None = None,
    started: Callable[[], None] | None = None,
) -> ScriptedOllamaResponse:
    tool_calls = [
        {
            "function": {
                "name": to_provider_tool_name(call.name),
                "arguments": call.arguments,
            }
        }
        for call in calls
    ]
    return ScriptedOllamaResponse(
        lines=[
            {"message": {"tool_calls": tool_calls}, "done": False},
            {"message": {}, "done": True, "prompt_eval_count": 1, "eval_count": 1},
        ],
        block=block,
        started=started,
    )


class _ScriptedOllamaProvider(OllamaProvider):
    def __init__(self, config: ModelConfig, handler: Callable[[httpx.Request], Awaitable[httpx.Response]]) -> None:
        self.config = config
        self.base_url = (getattr(config, "base_url", None) or "http://scripted-ollama.test").rstrip("/")
        self.model = getattr(config, "name", None)
        self.retry = getattr(config, "retry", None)
        self.timeout = httpx.Timeout((getattr(config, "timeout_ms", 60000) or 60000) / 1000)
        self._client = httpx.AsyncClient(timeout=self.timeout, transport=httpx.MockTransport(handler))


class _ScriptedStream(httpx.AsyncByteStream):
    def __init__(
        self,
        lines: list[dict[str, Any] | str | bytes],
        *,
        block: Blocker | None = None,
        block_after_lines: int | None = None,
    ) -> None:
        self._lines = lines
        self._block = block
        self._block_after_lines = block_after_lines

    async def __aiter__(self):
        for index, line in enumerate(self._lines, start=1):
            yield _line_bytes(line) + b"\n"
            if self._block is not None and self._block_after_lines == index:
                await self._block()


def _line_bytes(line: dict[str, Any] | str | bytes) -> bytes:
    if isinstance(line, bytes):
        return line
    if isinstance(line, str):
        return line.encode("utf-8")
    return json.dumps(line, separators=(",", ":")).encode("utf-8")


@pytest.fixture
def scripted_ollama() -> ScriptedOllama:
    return ScriptedOllama()
