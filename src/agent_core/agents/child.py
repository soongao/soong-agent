from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChildAgentManager:
    max_children_per_run: int
    active_children: int = 0

    def can_start(self) -> bool:
        return self.active_children < self.max_children_per_run

    def started(self) -> None:
        self.active_children += 1

    def finished(self) -> None:
        self.active_children = max(0, self.active_children - 1)
