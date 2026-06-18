from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator

from agent_core.storage import new_id
from agent_core.types.common import utc_iso
from agent_hub.backend.models import HubEvent


class HubEventHub:
    def __init__(self, *, queue_size: int = 100) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[str | None, set[asyncio.Queue[HubEvent]]] = defaultdict(set)

    async def publish(self, event_type: str, *, conversation_id: str | None = None, payload: dict | None = None) -> HubEvent:
        event = HubEvent(
            id=new_id("hub_evt"),
            type=event_type,
            conversation_id=conversation_id,
            payload=payload or {},
            created_at=utc_iso(),
        )
        targets = set(self._subscribers.get(None, set()))
        targets.update(self._subscribers.get(conversation_id, set()))
        for queue in targets:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
        return event

    async def subscribe(self, conversation_id: str | None = None) -> AsyncIterator[HubEvent]:
        queue: asyncio.Queue[HubEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[conversation_id].add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[conversation_id].discard(queue)


def sse_encode(event: HubEvent) -> str:
    return f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False)}\n\n"

