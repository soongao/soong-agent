from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable

class RunScheduler:
    def __init__(self) -> None:
        self._queues: dict[Hashable, deque] = defaultdict(deque)

    def enqueue(self, key: Hashable, item) -> None:
        self._queues[key].append(item)

    def dequeue(self, key: Hashable):
        queue = self._queues.get(key)
        if not queue:
            return None
        return queue.popleft()

    def has_pending(self, key: Hashable) -> bool:
        return bool(self._queues.get(key))
