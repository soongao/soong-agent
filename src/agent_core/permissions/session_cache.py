from __future__ import annotations


class PermissionSessionCache:
    def __init__(self) -> None:
        self._allowed: set[tuple[str, str]] = set()

    def allow(self, *, tool_name: str, target_scope: str | None) -> None:
        self._allowed.add((tool_name, target_scope or ""))

    def is_allowed(self, *, tool_name: str, target_scope: str | None) -> bool:
        return (tool_name, target_scope or "") in self._allowed

    def clear(self) -> None:
        self._allowed.clear()

