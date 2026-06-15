from __future__ import annotations

from collections import deque

class RunQueue:
    def __init__(self) -> None:
        self._items = deque()

    def put(self, item) -> None:
        self._items.append(item)

    def get(self):
        return self._items.popleft() if self._items else None

    def remove(self, item) -> bool:
        try:
            self._items.remove(item)
            return True
        except ValueError:
            return False

    def __len__(self) -> int:
        return len(self._items)
