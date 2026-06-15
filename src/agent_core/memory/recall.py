from __future__ import annotations

from pathlib import Path


class MemoryRecallService:
    def __init__(self, *, memory_dir: Path) -> None:
        self.memory_dir = memory_dir

    def recall(self, query: str, *, top_k: int = 5) -> list[dict[str, str]]:
        query_lower = query.lower()
        matches: list[dict[str, str]] = []
        if not self.memory_dir.exists():
            return matches
        for path in sorted(self.memory_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if query_lower in text.lower() or query_lower in path.name.lower():
                matches.append({"path": str(path), "content": text})
            if len(matches) >= top_k:
                break
        return matches
