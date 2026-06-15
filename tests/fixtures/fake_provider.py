from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio

from agent_core.providers.base import ModelEvent, ModelRequest, ProviderAdapter, StopReason
from agent_core.types.content import TextBlock
from agent_core.types.tools import ToolCall


class FakeProvider(ProviderAdapter):
    def __init__(self, config=None, *, tool_call: ToolCall | None = None, final_text: str = "done", block_event: asyncio.Event | None = None) -> None:
        self.config = config
        self.tool_call = tool_call
        self.final_text = final_text
        self.block_event = block_event
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelEvent(event_type="model_started")
        if self.block_event is not None and len(self.requests) == 1:
            await self.block_event.wait()
        if self.tool_call and len(self.requests) == 1:
            yield ModelEvent(
                event_type="model_completed",
                tool_calls=[self.tool_call],
                stop_reason=StopReason.TOOL_USE,
            )
            return
        yield ModelEvent(event_type="model_text_delta", text_delta=self.final_text)
        yield ModelEvent(
            event_type="model_completed",
            content=[TextBlock(text=self.final_text)],
            stop_reason=StopReason.END_TURN,
        )

    async def close(self) -> None:
        return None
