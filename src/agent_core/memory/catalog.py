from __future__ import annotations

from pathlib import Path


def list_memory_files(memory_dir: Path) -> list[Path]:
    if not memory_dir.exists():
        return []
    return sorted(memory_dir.rglob("*.md"))

