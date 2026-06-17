from __future__ import annotations

import json
from typing import Any

from agent_core.providers.base import ModelEvent
from agent_core.providers.tool_mapping import from_provider_tool_name
from agent_core.types.tools import ToolCall


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
