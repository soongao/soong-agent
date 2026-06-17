from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.types import ArtifactRefBlock, JsonBlock, ToolCall


RAW_DEBUG_METADATA_KEY = "raw_debug"


@dataclass(frozen=True)
class ProviderDebugArtifact:
    artifact_id: str
    provider: str
    summary: str | None


async def persist_result_artifacts(runtime: Any, handle: RunHandle, call: ToolCall, result: Any) -> None:
    assert runtime.store and runtime.paths
    artifact_refs: list[tuple[str, str]] = []
    metadata = getattr(result, "metadata", None) or {}
    for key in ("stdout_artifact_id", "stderr_artifact_id", "output_artifact_id"):
        artifact_id = metadata.get(key)
        if not artifact_id:
            continue
        artifact_refs.append((str(artifact_id), key))
    for artifact_id in metadata.get("artifact_ids") or []:
        artifact_refs.append((str(artifact_id), "tool_output"))
    for block in getattr(result, "content", None) or []:
        if isinstance(block, ArtifactRefBlock):
            artifact_refs.append((block.artifact_id, block.summary or "tool_output"))
        elif isinstance(block, JsonBlock) and block.artifact_id:
            artifact_refs.append((block.artifact_id, block.summary or "tool_output"))
    seen: set[str] = set()
    for artifact_id, summary in artifact_refs:
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        artifact_dir = runtime.paths.home_dir / "sessions" / handle.session_id / "artifacts" / artifact_id
        files = list(artifact_dir.iterdir()) if artifact_dir.exists() else []
        path = files[0] if files else artifact_dir
        await runtime.store.add_artifact(
            artifact_id=artifact_id,
            session_id=handle.session_id,
            agent_id=handle.agent_id,
            run_id=handle.run_id,
            tool_call_id=call.tool_call_id,
            path=str(path),
            filename=path.name,
            mime_type=_guess_artifact_mime_type(path),
            size_bytes=path.stat().st_size if path.exists() and path.is_file() else None,
            summary=summary,
            metadata={"debug": False, "raw": False},
        )


async def persist_provider_debug_artifact(
    runtime: Any,
    handle: RunHandle,
    model_event: Any,
    *,
    node_id: str | None,
) -> None:
    artifact = await persist_run_debug_artifact(
        runtime,
        session_id=handle.session_id,
        agent_id=handle.agent_id,
        run_id=handle.run_id,
        model_event=model_event,
        node_id=node_id,
    )
    if artifact is None:
        return
    await runtime._emit(
        handle,
        "provider_debug_artifact_created",
        level="debug",
        node_id=node_id,
        payload={
            "artifact_id": artifact.artifact_id,
            "provider": artifact.provider,
            "summary": artifact.summary,
        },
    )


async def persist_run_debug_artifact(
    runtime: Any,
    *,
    session_id: str,
    agent_id: str | None,
    run_id: str | None,
    model_event: Any,
    node_id: str | None,
) -> ProviderDebugArtifact | None:
    if not runtime.debug or runtime.artifacts is None or runtime.store is None:
        return None
    metadata = getattr(model_event, "metadata", {}) or {}
    raw_debug = metadata.get(RAW_DEBUG_METADATA_KEY)
    if raw_debug is None:
        return None
    provider = str(metadata.get("provider") or "provider")
    artifact = runtime.artifacts.write_text(
        session_id=session_id,
        text=json.dumps(raw_debug, ensure_ascii=False, indent=2, default=str),
        filename=f"{provider}_raw_model.json",
        mime_type="application/json",
        summary="Raw provider request/response",
    )
    await runtime.store.add_artifact(
        artifact_id=artifact.artifact_id,
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        node_id=node_id,
        path=str(artifact.path),
        filename=artifact.filename,
        mime_type=artifact.mime_type,
        size_bytes=artifact.size_bytes,
        summary=artifact.summary,
        metadata={"debug": True, "raw": True, "provider": provider},
    )
    return ProviderDebugArtifact(artifact_id=artifact.artifact_id, provider=provider, summary=artifact.summary)


def _artifact_selected_for_cleanup(
    *,
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    include_all: bool,
    older_than: datetime | None,
    max_bytes: int | None,
) -> bool:
    if not (include_all or metadata.get("debug") is True or metadata.get("raw") is True):
        return False
    if older_than is not None:
        created_at = _parse_datetime(artifact.get("created_at"))
        if created_at is None or created_at >= older_than:
            return False
    if max_bytes is not None:
        size = artifact.get("size_bytes")
        if size is None or int(size) <= max_bytes:
            return False
    return True


def _artifact_cleanup_reason(metadata: dict[str, Any], *, include_all: bool, max_bytes: int | None) -> str:
    if metadata.get("debug") is True:
        return "debug_artifact_cleanup"
    if metadata.get("raw") is True:
        return "raw_artifact_cleanup"
    if max_bytes is not None:
        return "artifact_size_limit"
    if include_all:
        return "include_all"
    return "artifact_cleanup"


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _path_older_than(path: Path, older_than: datetime) -> bool:
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return False
    if older_than.tzinfo is None:
        older_than = older_than.replace(tzinfo=UTC)
    return modified_at < older_than


def _path_mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _delete_artifact_path(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.name.startswith("art_") and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return
    if path.exists() and path.is_dir():
        for child in path.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        if not any(path.iterdir()):
            path.rmdir()


def _guess_artifact_mime_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    return "text/plain"
