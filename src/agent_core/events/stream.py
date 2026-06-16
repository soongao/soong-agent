from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent_core.types.runtime import RuntimeEvent


class EventStream:
    def __init__(self, *, maxsize: int = 256) -> None:
        self._queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._consumed = False

    @property
    def has_consumer(self) -> bool:
        return self._consumed

    async def put(self, event: RuntimeEvent) -> None:
        if not self._closed:
            await self._queue.put(event)

    def put_nowait(self, event: RuntimeEvent) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            return False
        return True

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._consumed:
                await self._queue.put(None)
                return
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._queue.put_nowait(None)

    async def iter(self) -> AsyncIterator[RuntimeEvent]:
        if self._consumed:
            raise RuntimeError("run.events() is a single-consumer stream")
        self._consumed = True
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
