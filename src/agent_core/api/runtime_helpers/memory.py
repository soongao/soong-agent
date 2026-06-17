from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_core.types import Node


def _compact_input(nodes: list[Node]) -> str:
    lines = ["Summarize the following active context for compaction.", ""]
    for node in nodes:
        text_parts = [getattr(block, "text", "") for block in node.content if getattr(block, "type", None) == "text"]
        if not text_parts:
            continue
        lines.append(f"[{node.node_id}] {node.role}/{node.node_type}:")
        lines.append("\n".join(text_parts))
        lines.append("")
    return "\n".join(lines).strip()


def _memory_extraction_source_text(nodes: list[Node]) -> str:
    lines: list[str] = []
    for node in nodes:
        text_parts = [getattr(block, "text", "") for block in node.content if getattr(block, "type", None) == "text"]
        if not text_parts:
            continue
        lines.append(f"<node id=\"{node.node_id}\" role=\"{node.role}\" type=\"{node.node_type}\">")
        lines.append("\n".join(text_parts)[:8000])
        lines.append("</node>")
        lines.append("")
    return "\n".join(lines).strip()


def _memory_frontmatter_candidates(memory_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not memory_root.exists():
        return candidates
    for path in sorted(memory_root.glob("*/*.md")):
        if path.parent.name not in {"user", "feedback", "reference"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = _simple_frontmatter(text)
        candidates.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(memory_root)),
                "id": metadata.get("id") or path.stem,
                "category": metadata.get("category") or path.parent.name,
                "summary": metadata.get("summary") or path.stem,
                "tags": _frontmatter_list(metadata.get("tags")),
                "excerpt": _memory_candidate_excerpt(text),
            }
        )
    return candidates


def _memory_candidate_selector_line(item: dict[str, Any]) -> str:
    tags = item.get("tags") or []
    tag_text = f" tags={', '.join(str(tag) for tag in tags)}" if tags else ""
    excerpt = str(item.get("excerpt") or "").replace("\n", " ").strip()
    excerpt_text = f" excerpt={excerpt[:800]}" if excerpt else ""
    return f"- {item['relative_path']} [{item.get('category')}] id={item.get('id')} summary={item.get('summary')}{tag_text}{excerpt_text}"


def _memory_candidate_excerpt(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            rest = text.find("\n", end + 4)
            text = text[rest + 1 :] if rest != -1 else ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:1200]


def _frontmatter_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            return [item.strip().strip('"').strip("'") for item in text[1:-1].split(",") if item.strip()]
        return [text]
    return []


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _estimate_memory_source_tokens(nodes: list[Node]) -> int:
    char_count = 0
    for node in nodes:
        for block in node.content:
            if getattr(block, "type", None) == "text":
                char_count += len(getattr(block, "text", ""))
            elif getattr(block, "type", None) == "json":
                char_count += len(json.dumps(getattr(block, "data", None), ensure_ascii=False))
    return max(char_count // 4, 0)


def _simple_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip('"').strip("'")
            current = metadata.setdefault(current_key, [])
            if isinstance(current, list):
                current.append(value)
            continue
        current_key = None
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            metadata[key] = value.strip('"').strip("'")
        else:
            metadata[key] = []
            current_key = key
    return metadata
