from __future__ import annotations

import json

import httpx
import pytest

from agent_core.config.models import ModelConfig, RetryConfig
from agent_core.providers.anthropic import AnthropicToolAccumulator, anthropic_sse_to_events, build_anthropic_payload
from agent_core.providers.base import ModelMessage, ModelRequest, ModelRole, SystemBlock
from agent_core.providers.errors import classify_provider_exception, retry_delay_seconds
from agent_core.providers.openai_compatible import OpenAICompatibleProvider
from agent_core.providers.openai_compatible import OpenAIToolAccumulator, build_openai_chat_payload, openai_chunk_to_events
from agent_core.providers.ollama import OllamaProvider
from agent_core.errors.codes import ErrorCode
from agent_core.types.content import TextBlock, ToolCallBlock, ToolResultBlock
from agent_core.types.tools import ToolDefinition


def json_loads(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def test_openai_payload_mangles_tool_name() -> None:
    request = ModelRequest(
        model="m",
        system=[SystemBlock(block_id="s", source="test", content="system")],
        messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text="hi")])],
        tools=[
            ToolDefinition(
                name="code.read_file",
                description="read",
                input_schema={"type": "object", "properties": {}},
                permission="readonly",
            )
        ],
    )
    payload = build_openai_chat_payload(request)
    assert payload["tools"][0]["function"]["name"] == "code__read_file"
    assert payload["messages"][0]["role"] == "system"


def test_openai_payload_preserves_tool_call_and_result_messages() -> None:
    request = ModelRequest(
        model="m",
        messages=[
            ModelMessage(
                role=ModelRole.ASSISTANT,
                content=[ToolCallBlock(tool_call_id="call1", name="code.list_dir", arguments={"path": "."})],
            ),
            ModelMessage(
                role=ModelRole.TOOL,
                content=[
                    ToolResultBlock(
                        tool_call_id="call1",
                        content=[TextBlock(text="a.txt")],
                        metadata={"tool_name": "code.list_dir"},
                    )
                ],
            ),
        ],
    )
    payload = build_openai_chat_payload(request)
    assert payload["messages"][0]["tool_calls"][0]["function"]["name"] == "code__list_dir"
    assert json.loads(payload["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"path": "."}
    assert payload["messages"][1]["role"] == "tool"
    assert payload["messages"][1]["tool_call_id"] == "call1"
    assert payload["messages"][1]["content"] == "a.txt"


def test_openai_tool_call_accumulates_arguments() -> None:
    state = OpenAIToolAccumulator(known_names={"code.read_file"})
    events = openai_chunk_to_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "function": {"name": "code__read_file", "arguments": '{"path":"a'},
                            }
                        ]
                    }
                }
            ]
        },
        state,
    )
    assert events[0].event_type == "tool_call_delta"
    openai_chunk_to_events(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '.txt"}'}}]}}]},
        state,
    )
    call = state.tool_calls()[0]
    assert call.name == "code.read_file"
    assert call.arguments == {"path": "a.txt"}


def test_anthropic_payload_mangles_tool_name() -> None:
    request = ModelRequest(
        model="m",
        system=[SystemBlock(block_id="s", source="test", content="system")],
        messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text="hi")])],
        tools=[
            ToolDefinition(
                name="code.search",
                description="search",
                input_schema={"type": "object", "properties": {}},
                permission="readonly",
            )
        ],
    )
    payload = build_anthropic_payload(request)
    assert payload["system"] == "system"
    assert payload["tools"][0]["name"] == "code__search"


def test_anthropic_payload_preserves_tool_call_and_result_messages() -> None:
    request = ModelRequest(
        model="m",
        messages=[
            ModelMessage(
                role=ModelRole.ASSISTANT,
                content=[ToolCallBlock(tool_call_id="toolu_1", name="code.list_dir", arguments={"path": "."})],
            ),
            ModelMessage(
                role=ModelRole.TOOL,
                content=[
                    ToolResultBlock(
                        tool_call_id="toolu_1",
                        content=[TextBlock(text="a.txt")],
                        metadata={"tool_name": "code.list_dir"},
                    )
                ],
            ),
        ],
    )
    payload = build_anthropic_payload(request)
    assert payload["messages"][0]["content"][0]["type"] == "tool_use"
    assert payload["messages"][0]["content"][0]["name"] == "code__list_dir"
    assert payload["messages"][0]["content"][0]["input"] == {"path": "."}
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"][0]["type"] == "tool_result"
    assert payload["messages"][1]["content"][0]["content"] == "a.txt"


def test_anthropic_tool_call_accumulates_partial_json() -> None:
    state = AnthropicToolAccumulator(known_names={"code.search"})
    anthropic_sse_to_events(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "code__search"}},
        state,
    )
    events = anthropic_sse_to_events(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"x"}'}},
        state,
    )
    assert events[0].event_type == "tool_call_delta"
    call = state.tool_calls()[0]
    assert call.name == "code.search"
    assert call.arguments == {"query": "x"}


@pytest.mark.asyncio
async def test_ollama_payload_preserves_tool_call_and_result_messages() -> None:
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json_loads(request.content)
        return httpx.Response(200, request=request, content=b'{"message":{},"done":true}\n')

    provider = OllamaProvider(ModelConfig(provider="ollama", name="gemma4"))
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="gemma4",
                messages=[
                    ModelMessage(
                        role=ModelRole.ASSISTANT,
                        content=[ToolCallBlock(tool_call_id="call1", name="code.list_dir", arguments={"path": "."})],
                    ),
                    ModelMessage(
                        role=ModelRole.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_call_id="call1",
                                content=[TextBlock(text="a.txt")],
                                metadata={"tool_name": "code.list_dir"},
                            )
                        ],
                    ),
                ],
            )
        )
    ]
    await provider.close()

    assert captured_payload["messages"][0]["role"] == "assistant"
    assert captured_payload["messages"][0]["tool_calls"][0]["function"]["name"] == "code__list_dir"
    assert captured_payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"path": "."}
    assert captured_payload["messages"][1]["role"] == "tool"
    assert captured_payload["messages"][1]["content"] == "a.txt"


@pytest.mark.asyncio
async def test_ollama_provider_mangles_and_unmangles_tool_name() -> None:
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json_loads(request.content)
        return httpx.Response(
            200,
            request=request,
            content=(
                b'{"message":{"tool_calls":[{"function":{"name":"code__list_dir","arguments":{"path":"."}}}]},"done":false}\n'
                b'{"message":{},"done":true,"prompt_eval_count":1,"eval_count":1}\n'
            ),
        )

    provider = OllamaProvider(ModelConfig(provider="ollama", name="gemma4"))
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="gemma4",
                messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text="list")])],
                tools=[
                    ToolDefinition(
                        name="code.list_dir",
                        description="list",
                        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                        permission="readonly",
                    )
                ],
            )
        )
    ]
    await provider.close()

    assert captured_payload["tools"][0]["function"]["name"] == "code__list_dir"
    completed = events[-1]
    assert completed.tool_calls[0].name == "code.list_dir"
    assert completed.tool_calls[0].metadata["raw_name"] == "code__list_dir"


def test_provider_error_classification() -> None:
    request = httpx.Request("POST", "https://provider.test")
    unauthorized = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, request=request),
    )
    rate_limited = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=httpx.Response(429, request=request, headers={"retry-after": "2"}),
    )
    server_error = httpx.HTTPStatusError(
        "server error",
        request=request,
        response=httpx.Response(503, request=request),
    )

    assert classify_provider_exception(unauthorized).code == ErrorCode.PROVIDER_AUTH_FAILED
    assert not classify_provider_exception(unauthorized).retryable
    assert classify_provider_exception(rate_limited).code == ErrorCode.PROVIDER_RATE_LIMITED
    assert classify_provider_exception(rate_limited).retry_after_ms == 2000
    assert classify_provider_exception(server_error).retryable
    assert retry_delay_seconds(attempt_index=2, retry=RetryConfig(initial_backoff_ms=10, max_backoff_ms=15)) == 0.015


@pytest.mark.asyncio
async def test_openai_provider_retries_retryable_http_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai-compatible",
            base_url="https://provider.test",
            name="model",
            retry=RetryConfig(max_attempts=2, initial_backoff_ms=0, max_backoff_ms=0),
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [
        event
        async for event in provider.stream(
            ModelRequest(model="model", messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text="hi")])])
        )
    ]
    await provider.close()

    assert calls == 2
    assert events[-1].event_type == "model_completed"
    assert events[-1].metadata["retry_count"] == 1
    assert events[-1].content[0].text == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_openai_provider_does_not_retry_auth_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, request=request)

    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai-compatible",
            base_url="https://provider.test",
            name="model",
            retry=RetryConfig(max_attempts=3, initial_backoff_ms=0, max_backoff_ms=0),
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [
        event
        async for event in provider.stream(
            ModelRequest(model="model", messages=[ModelMessage(role=ModelRole.USER, content=[TextBlock(text="hi")])])
        )
    ]
    await provider.close()

    assert calls == 1
    assert events[-1].event_type == "model_failed"
    assert events[-1].error and events[-1].error.code == ErrorCode.PROVIDER_AUTH_FAILED
    assert events[-1].error.details["retryable"] is False
