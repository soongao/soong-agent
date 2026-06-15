from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build", "target", ".next", ".cache", "__pycache__"}


@dataclass(frozen=True)
class InstructionEntry:
    path: Path
    metadata: dict[str, Any]


def build_instruction_catalog(*, home_dir: Path, project_dir: Path, limit: int = 200) -> tuple[list[InstructionEntry], bool]:
    candidates: list[Path] = []
    home_claude = home_dir / "CLAUDE.md"
    home_agents = home_dir / "AGENTS.md"
    if home_claude.exists():
        candidates.append(home_claude)
    elif home_agents.exists():
        candidates.append(home_agents)
    rules_dir = home_dir / "rules"
    if rules_dir.exists():
        candidates.extend(sorted(rules_dir.rglob("*.md")))
    project_candidates: dict[Path, Path] = {}
    for path in sorted(project_dir.rglob("CLAUDE.md")) + sorted(project_dir.rglob("AGENTS.md")):
        if _skip_path(path):
            continue
        if ".soong-agent" in path.parts and "rules" in path.parts:
            continue
        key = path.parent
        if path.name == "CLAUDE.md" or key not in project_candidates:
            project_candidates[key] = path
    candidates.extend(project_candidates.values())
    entries = [InstructionEntry(path=path.resolve(), metadata=_frontmatter_metadata(path)) for path in sorted(set(candidates))]
    truncated = len(entries) > limit
    return entries[:limit], truncated


def instruction_catalog_text(entries: list[InstructionEntry], *, truncated: bool) -> str:
    lines = ["# Instruction Catalog", "", "Read specific files with code.read_file when relevant.", ""]
    for entry in entries:
        lines.append(f"- path: {entry.path}")
        for key, value in entry.metadata.items():
            lines.append(f"  {key}: {value}")
    if truncated:
        lines.append("")
        lines.append("catalog_truncated: true")
    return "\n".join(lines)


def _skip_path(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _frontmatter_metadata(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"name": path.name}
    if not text.startswith("---\n"):
        return {"name": path.name}
    end = text.find("\n---", 4)
    if end == -1:
        return {"name": path.name}
    metadata: dict[str, Any] = {}
    for raw in text[4:end].splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata or {"name": path.name}
