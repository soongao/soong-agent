from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_core.artifacts.redaction import redact_text
from agent_core.storage.ids import new_id


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    path: Path
    filename: str
    mime_type: str | None
    size_bytes: int
    summary: str | None = None


class ArtifactManager:
    def __init__(self, *, home_dir: Path) -> None:
        self.home_dir = home_dir

    def write_bytes(
        self,
        *,
        session_id: str,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
        summary: str | None = None,
    ) -> ArtifactRecord:
        artifact_id = new_id("art")
        directory = self.home_dir / "sessions" / session_id / "artifacts" / artifact_id
        directory.mkdir(parents=True, exist_ok=True)
        safe_filename = _safe_artifact_filename(filename)
        stored = _redact_textual_bytes(data, mime_type=mime_type, filename=safe_filename)
        path = directory / safe_filename
        path.write_bytes(stored)
        return ArtifactRecord(
            artifact_id=artifact_id,
            path=path,
            filename=safe_filename,
            mime_type=mime_type,
            size_bytes=len(stored),
            summary=summary,
        )

    def write_text(
        self,
        *,
        session_id: str,
        text: str,
        filename: str,
        mime_type: str | None = "text/plain",
        summary: str | None = None,
    ) -> ArtifactRecord:
        redacted = redact_text(text)
        return self.write_bytes(
            session_id=session_id,
            data=redacted.encode("utf-8"),
            filename=filename,
            mime_type=mime_type,
            summary=summary,
        )


def _safe_artifact_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename:
        raise ValueError("artifact filename must be a single path segment")
    return filename


def _redact_textual_bytes(data: bytes, *, mime_type: str | None, filename: str) -> bytes:
    if not _looks_textual(mime_type=mime_type, filename=filename):
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return redact_text(text).encode("utf-8")


def _looks_textual(*, mime_type: str | None, filename: str) -> bool:
    if mime_type is not None:
        lowered = mime_type.lower()
        return lowered.startswith("text/") or lowered in {"application/json", "application/xml", "application/javascript"}
    return Path(filename).suffix.lower() in {".txt", ".md", ".json", ".xml", ".csv", ".log", ".yaml", ".yml"}
