from __future__ import annotations

from pathlib import Path

from agent_core.types.runtime import CleanupResult


async def cleanup_artifact_paths(paths: list[Path], *, dry_run: bool = True) -> CleanupResult:
    candidates = [{"path": str(path), "reason": "artifact_cleanup"} for path in paths if path.exists()]
    deleted: list[str] = []
    if not dry_run:
        for path in paths:
            if path.exists() and path.is_file():
                path.unlink()
                deleted.append(str(path))
    return CleanupResult(dry_run=dry_run, candidates=candidates, deleted=deleted)
