from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_core.events import EventStream
from agent_core.types.runtime import CancelResult, InspectResult, RunMode, RunStatus, RuntimeEvent

if TYPE_CHECKING:
    from agent_core.api.runtime import AgentRuntime


@dataclass
class RunHandle:
    run_id: str
    session_id: str
    agent_id: str
    status: RunStatus
    mode: RunMode
    _runtime: "AgentRuntime"
    _stream: EventStream
    _task: asyncio.Task | None = None
    _queued: bool = False
    _message: object | None = None

    def events(self, debug: bool = False) -> AsyncIterator[RuntimeEvent]:
        return _filtered_events(self._stream, debug=debug)

    async def cancel(self) -> CancelResult:
        return await self._runtime._cancel_run(self)

    async def inspect_child(self, child_run_id: str, include_sensitive: bool = False) -> InspectResult:
        return await self._runtime._inspect_run(child_run_id, include_sensitive=include_sensitive)

    def child_events(self, child_run_id: str, debug: bool = False) -> AsyncIterator[RuntimeEvent]:
        return self._runtime._child_events(child_run_id, debug=debug)


async def _filtered_events(stream: EventStream, *, debug: bool) -> AsyncIterator[RuntimeEvent]:
    async for event in stream.iter():
        if debug or event.level != "debug":
            yield event
