from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_core.types.runtime import RuntimeEvent


@dataclass(frozen=True)
class RenderLine:
    text: str
    level: str = "info"


def event_to_lines(event: RuntimeEvent) -> list[RenderLine]:
    event_type = event.event_type
    payload: dict[str, Any] = event.payload
    if event_type == "tool_started":
        return [RenderLine(f"[tool] {payload.get('name', 'unknown')} started")]
    if event_type == "tool_completed":
        name = payload.get("name", "unknown")
        if payload.get("is_error"):
            error = payload.get("error") or {}
            return [RenderLine(f"[tool] {name} failed: {error.get('message', 'error')}", "error")]
        return [RenderLine(f"[tool] {name} completed")]
    if event_type == "tool_failed":
        return [RenderLine(f"[tool] {payload.get('name', 'unknown')} failed", "error")]
    if event_type == "loop_failed":
        return [RenderLine(f"[error] {payload.get('message', 'loop failed')}", "error")]
    if event_type == "run_cancelled":
        return [RenderLine("[run] cancelled", "warning")]
    if event_type == "run_completed":
        return [RenderLine("[run] completed")]
    if event_type.startswith("memory_extraction_"):
        reason = payload.get("reason")
        suffix = f" ({reason})" if reason else ""
        return [RenderLine(f"[memory] {event_type.removeprefix('memory_extraction_')}{suffix}")]
    if event_type.startswith("compact_"):
        reason = payload.get("reason")
        suffix = f" ({reason})" if reason else ""
        return [RenderLine(f"[compact] {event_type.removeprefix('compact_')}{suffix}")]
    return []
