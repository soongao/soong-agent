from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.memory.cursor import MemoryScanCursor
from agent_core.memory.writer import ensure_memory_write_allowed
from agent_core.types.common import utc_iso


@dataclass
class MemoryCandidate:
    category: str
    filename: str
    content: str
    source_node_ids: list[str]
    memory_id: str | None = None
    summary: str | None = None
    tags: list[str] | None = None


@dataclass
class MemoryExtractionResult:
    created: list[str]
    updated: list[str]
    ignored: list[str]
    duplicate: list[str]
    files_changed: list[str]
    scan_cursor: MemoryScanCursor


class MemoryExtractionJob:
    def __init__(self, *, home_dir: Path, cursor: MemoryScanCursor | None = None) -> None:
        self.home_dir = home_dir
        self.cursor = cursor or MemoryScanCursor()

    def apply(self, candidates: list[MemoryCandidate], *, source_node_seq: int) -> MemoryExtractionResult:
        operations = [_prepare_candidate(self.home_dir, candidate) for candidate in candidates]
        memory_root = self.home_dir / "memory"
        catalog_path = memory_root / "MEMORY.md"
        old_catalog = catalog_path.read_text(encoding="utf-8") if catalog_path.exists() else None
        created: list[str] = []
        updated: list[str] = []
        ignored: list[str] = []
        duplicate: list[str] = []
        changed: list[str] = []
        writes: list[tuple[Path, str, str | None]] = []
        for operation in operations:
            if operation.get("ignored"):
                ignored.append(operation["filename"])
                continue
            path: Path = operation["path"]
            content = operation["content"]
            old = path.read_text(encoding="utf-8") if path.exists() else None
            if old == content or (old is not None and _memory_equivalent(old, content)):
                duplicate.append(str(path))
                continue
            writes.append((path, content, old))
            if old is None:
                created.append(str(path))
            else:
                updated.append(str(path))
            changed.append(str(path))
        catalog_content = _build_catalog(memory_root=memory_root, pending=[op for op in operations if not op.get("ignored")])
        if catalog_content != old_catalog:
            writes.append((catalog_path, catalog_content, old_catalog))
            changed.append(str(catalog_path))
        written: list[tuple[Path, str | None]] = []
        try:
            for path, content, old in writes:
                ensure_memory_write_allowed(path, home_dir=self.home_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                written.append((path, old))
        except Exception as exc:
            for path, old in reversed(written):
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(old, encoding="utf-8")
            if isinstance(exc, AgentCoreError):
                raise
            raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, str(exc)) from exc
        self.cursor.node_seq = source_node_seq
        return MemoryExtractionResult(
            created=created,
            updated=updated,
            ignored=ignored,
            duplicate=duplicate,
            files_changed=changed,
            scan_cursor=self.cursor,
        )


def parse_memory_candidates(text: str) -> list[MemoryCandidate]:
    payload = _json_payload(text)
    raw_items = payload.get("memories", payload.get("candidates", [])) if isinstance(payload, dict) else payload
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, "memory extraction output must contain a memories array")
    candidates: list[MemoryCandidate] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, f"memory candidate at index {index} must be an object")
        decision = str(raw.get("decision") or raw.get("action") or "new").lower()
        if decision in {"ignore", "ignored", "duplicate"}:
            continue
        candidates.append(
            MemoryCandidate(
                category=str(raw.get("category") or ""),
                filename=str(raw.get("filename") or raw.get("path") or ""),
                content=str(raw.get("content") or raw.get("body") or ""),
                source_node_ids=[str(item) for item in (raw.get("source_node_ids") or [])],
                memory_id=str(raw["memory_id"]) if raw.get("memory_id") else None,
                summary=str(raw["summary"]) if raw.get("summary") else None,
                tags=[str(item) for item in (raw.get("tags") or [])],
            )
        )
    return candidates


def _prepare_candidate(home_dir: Path, candidate: MemoryCandidate) -> dict[str, Any]:
    if not candidate.source_node_ids:
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, "memory candidate missing source_node_ids")
    if candidate.category not in {"user", "feedback", "reference"}:
        return {"ignored": True, "filename": candidate.filename}
    filename = Path(candidate.filename).name
    if filename != candidate.filename or not filename.endswith(".md"):
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, "memory filename must be a local markdown filename")
    memory_id = candidate.memory_id or filename.removesuffix(".md")
    if not memory_id.startswith("mem_"):
        memory_id = "mem_" + memory_id
    body = _strip_frontmatter(candidate.content).lstrip()
    summary = candidate.summary or _first_content_line(body) or filename
    content = _frontmatter(
        {
            "id": memory_id,
            "category": candidate.category,
            "summary": summary,
            "tags": candidate.tags or [],
            "created_at": utc_iso(),
            "updated_at": utc_iso(),
            "source_node_ids": candidate.source_node_ids,
        }
    ) + body
    path = home_dir / "memory" / candidate.category / filename
    ensure_memory_write_allowed(path, home_dir=home_dir)
    return {
        "path": path,
        "content": content,
        "id": memory_id,
        "category": candidate.category,
        "summary": summary,
        "filename": filename,
        "source_node_ids": candidate.source_node_ids,
    }


def _build_catalog(*, memory_root: Path, pending: list[dict[str, Any]]) -> str:
    entries: list[dict[str, Any]] = []
    if memory_root.exists():
        for path in sorted(memory_root.glob("*/*.md")):
            if path.parent.name not in {"user", "feedback", "reference"}:
                continue
            metadata = _parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            if metadata:
                entries.append(
                    {
                        "id": metadata.get("id") or path.stem,
                        "category": metadata.get("category") or path.parent.name,
                        "summary": metadata.get("summary") or path.stem,
                        "path": str(path.relative_to(memory_root)),
                    }
                )
    pending_by_path = {operation["path"].resolve(): operation for operation in pending}
    entries = [entry for entry in entries if (memory_root / entry["path"]).resolve() not in pending_by_path]
    for operation in pending:
        entries.append(
            {
                "id": operation["id"],
                "category": operation["category"],
                "summary": operation["summary"],
                "path": str(operation["path"].relative_to(memory_root)),
            }
        )
    entries.sort(key=lambda entry: entry["id"], reverse=True)
    lines = ["# Memory Catalog", ""]
    for entry in entries:
        lines.append(f"- `{entry['id']}` [{entry['category']}] {entry['summary']} ({entry['path']})")
    return "\n".join(lines).rstrip() + "\n"


def _frontmatter(data: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    rest = text.find("\n", end + 4)
    return text[rest + 1 :] if rest != -1 else ""


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _memory_equivalent(left: str, right: str) -> bool:
    left_meta = _parse_frontmatter(left)
    right_meta = _parse_frontmatter(right)
    for volatile in ("created_at", "updated_at"):
        left_meta.pop(volatile, None)
        right_meta.pop(volatile, None)
    return left_meta == right_meta and _strip_frontmatter(left).strip() == _strip_frontmatter(right).strip()


def _first_content_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return None


def _json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, "memory extraction output is not valid JSON") from exc
