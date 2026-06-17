from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from agent_core.api.runtime_helpers.artifacts import (
    _artifact_cleanup_reason,
    _artifact_selected_for_cleanup,
    _delete_artifact_path,
    _path_mtime_iso,
    _path_older_than,
)
from agent_core.errors.codes import ErrorCode
from agent_core.types import CleanupResult, ErrorPayload


async def cleanup_project_tasks(
    runtime: Any,
    project: str | Path,
    *,
    dry_run: bool = True,
    include_failed: bool = False,
    include_cancelled: bool = False,
    older_than: datetime | None = None,
) -> CleanupResult:
    await runtime._ensure_started()
    assert runtime.paths
    root = Path(project).expanduser().resolve()
    task_root = root / ".soong-agent" / "tasks"
    candidates: list[dict[str, Any]] = []
    if task_root.exists():
        for wal in sorted(task_root.rglob("*.wal.jsonl")):
            if older_than is not None and not _path_older_than(wal, older_than):
                continue
            text = wal.read_text(encoding="utf-8", errors="replace")
            if "task_completed" in text or (include_failed and "task_failed" in text) or (
                include_cancelled and "task_cancelled" in text
            ):
                candidates.append({"path": str(wal), "reason": "terminal_task_wal", "modified_at": _path_mtime_iso(wal)})
    deleted: list[str] = []
    errors: list[ErrorPayload] = []
    if not dry_run:
        for candidate in candidates:
            try:
                Path(candidate["path"]).unlink(missing_ok=True)
            except OSError as exc:
                errors.append(
                    ErrorPayload(
                        code=ErrorCode.STORAGE_ERROR,
                        message=f"failed to delete task WAL: {candidate['path']}",
                        details={"path": candidate["path"], "error": str(exc)},
                    )
                )
                continue
            deleted.append(candidate["path"])
    return CleanupResult(dry_run=dry_run, candidates=candidates, deleted=deleted, errors=errors)


async def delete_artifact(runtime: Any, artifact_id: str) -> CleanupResult:
    await runtime._ensure_started()
    assert runtime.store
    artifact = await runtime.store.get_artifact(artifact_id)
    if artifact is None:
        return CleanupResult(
            dry_run=False,
            errors=[ErrorPayload(code=ErrorCode.VALIDATION_ERROR, message=f"artifact not found: {artifact_id}")],
        )
    try:
        _delete_artifact_path(Path(artifact["path"]))
    except OSError as exc:
        return CleanupResult(
            dry_run=False,
            candidates=[{"artifact_id": artifact_id, "path": artifact["path"], "reason": "delete_artifact"}],
            errors=[
                ErrorPayload(
                    code=ErrorCode.STORAGE_ERROR,
                    message=f"failed to delete artifact: {artifact_id}",
                    details={"artifact_id": artifact_id, "path": artifact["path"], "error": str(exc)},
                )
            ],
        )
    await runtime.store.delete_artifact(artifact_id)
    return CleanupResult(
        dry_run=False,
        candidates=[{"artifact_id": artifact_id, "path": artifact["path"], "reason": "delete_artifact"}],
        deleted=[artifact_id],
    )


async def cleanup_artifacts(
    runtime: Any,
    *,
    session_id: str | None = None,
    dry_run: bool = True,
    include_all: bool = False,
    older_than: datetime | None = None,
    max_bytes: int | None = None,
) -> CleanupResult:
    await runtime._ensure_started()
    assert runtime.store
    artifacts = await runtime.store.list_artifacts(session_id=session_id)
    candidates = []
    for artifact in artifacts:
        try:
            metadata = json.loads(artifact.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if not _artifact_selected_for_cleanup(
            artifact=artifact,
            metadata=metadata,
            include_all=include_all,
            older_than=older_than,
            max_bytes=max_bytes,
        ):
            continue
        candidates.append(
            {
                "artifact_id": artifact["artifact_id"],
                "path": artifact["path"],
                "reason": _artifact_cleanup_reason(metadata, include_all=include_all, max_bytes=max_bytes),
                "size_bytes": artifact.get("size_bytes"),
                "created_at": artifact.get("created_at"),
            }
        )
    deleted: list[str] = []
    errors: list[ErrorPayload] = []
    if not dry_run:
        for candidate in candidates:
            try:
                _delete_artifact_path(Path(candidate["path"]))
            except OSError as exc:
                errors.append(
                    ErrorPayload(
                        code=ErrorCode.STORAGE_ERROR,
                        message=f"failed to delete artifact: {candidate['artifact_id']}",
                        details={
                            "artifact_id": candidate["artifact_id"],
                            "path": candidate["path"],
                            "error": str(exc),
                        },
                    )
                )
                continue
            await runtime.store.delete_artifact(candidate["artifact_id"])
            deleted.append(candidate["artifact_id"])
    return CleanupResult(dry_run=dry_run, candidates=candidates, deleted=deleted, errors=errors)
