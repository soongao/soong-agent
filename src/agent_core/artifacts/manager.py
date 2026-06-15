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
        path = directory / filename
        path.write_bytes(data)
        return ArtifactRecord(
            artifact_id=artifact_id,
            path=path,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(data),
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
