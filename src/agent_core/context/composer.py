from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.assets.loader import get_asset, read_asset
from agent_core.context.instructions import build_auto_instruction_entries, build_instruction_catalog, instruction_catalog_text
from agent_core.context.skills import build_skill_catalog, skill_catalog_text
from agent_core.memory.writer import resolve_memory_dir
from agent_core.providers.base import SystemBlock


SYSTEM_PROMPT_ASSETS: tuple[tuple[str, str, int], ...] = (
    ("system.core", "core.md", 1000),
    ("system.tool_protocol", "tool_protocol.md", 950),
    ("system.todo", "todo.md", 900),
    ("system.permissions", "permissions.md", 850),
    ("system.multi_agent", "multi_agent.md", 830),
    ("system.memory", "memory.md", 820),
    ("system.compact", "compact.md", 810),
)


def build_static_system_blocks(*, home_dir: Path, project_dir: Path) -> list[SystemBlock]:
    entries, truncated = build_instruction_catalog(home_dir=home_dir, project_dir=project_dir)
    auto_entries = build_auto_instruction_entries(home_dir=home_dir, project_dir=project_dir)
    skills = build_skill_catalog(home_dir)
    blocks = [
        SystemBlock(
            block_id=block_id,
            source="package_asset",
            content=read_asset(block_id),
            priority=priority,
            dynamic=False,
            metadata={"asset_id": block_id, "asset_path": get_asset(block_id).resource_path},
        )
        for block_id, filename, priority in SYSTEM_PROMPT_ASSETS
    ]
    blocks.append(
        SystemBlock(
            block_id="system.instruction_catalog",
            source="instruction_catalog",
            content=instruction_catalog_text(entries, truncated=truncated),
            priority=800,
            dynamic=False,
            metadata={"truncated": truncated},
        ),
    )
    for index, entry in enumerate(auto_entries):
        content = _read_instruction(entry.path)
        if content is None:
            continue
        blocks.append(
            SystemBlock(
                block_id=f"system.auto_instruction.{index}",
                source="auto_instruction",
                content=content,
                priority=805,
                dynamic=False,
                metadata={"path": str(entry.path), "metadata": entry.metadata},
            )
        )
    blocks.append(
        SystemBlock(
            block_id="system.skill_catalog",
            source="skill_catalog",
            content=skill_catalog_text(skills),
            priority=790,
            dynamic=False,
            metadata={"count": len(skills)},
        ),
    )
    return blocks


def _read_instruction(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def build_dynamic_system_blocks(
    context_state: Any | None,
    *,
    home_dir: Path | None = None,
    memory_enabled: bool = True,
    memory_dir: Path | None = None,
) -> list[SystemBlock]:
    blocks: list[SystemBlock] = []
    if home_dir is not None and memory_enabled:
        catalog_path = (memory_dir or (home_dir / "memory")) / "MEMORY.md"
        if catalog_path.exists():
            blocks.append(
                SystemBlock(
                    block_id="memory.catalog",
                    source="memory_catalog",
                    content=catalog_path.read_text(encoding="utf-8", errors="replace"),
                    priority=770,
                    dynamic=True,
                    metadata={"path": str(catalog_path)},
                )
            )
    if context_state is None:
        return blocks
    for index, item in enumerate(getattr(context_state, "instruction_contexts", []) or []):
        body = item.get("body")
        if not body:
            continue
        blocks.append(
            SystemBlock(
                block_id=f"instruction.loaded.{index}",
                source="instruction_context",
                content=str(body),
                priority=780,
                dynamic=True,
                metadata={"path": item.get("path"), "hash": item.get("hash")},
            )
        )
    return blocks


def build_system_blocks(
    *,
    home_dir: Path,
    project_dir: Path,
    context_state: Any | None = None,
    memory_enabled: bool = True,
    memory_dir_template: str = "${SOONG_AGENT_HOME}/memory",
) -> list[SystemBlock]:
    memory_dir = resolve_memory_dir(memory_dir_template, home_dir=home_dir, project_dir=project_dir) if memory_enabled else None
    return build_static_system_blocks(home_dir=home_dir, project_dir=project_dir) + build_dynamic_system_blocks(
        context_state,
        home_dir=home_dir,
        memory_enabled=memory_enabled,
        memory_dir=memory_dir,
    )


def _memory_context_text(items: list[dict[str, Any]]) -> str:
    lines = ["# Recalled Memory", ""]
    for item in items:
        matches = item.get("matches") or []
        ids = ",".join(str(match.get("id") or "") for match in matches if match.get("id"))
        categories = ",".join(str(match.get("category") or "") for match in matches if match.get("category"))
        lines.append(f'<memory ids="{ids}" categories="{categories}" query="{item.get("query") or ""}">')
        for match in matches:
            content = match.get("content") or match.get("snippet") or match.get("text") or ""
            lines.append(f"## {match.get('id') or match.get('path')}")
            lines.append(str(content))
            lines.append("")
        lines.append("</memory>")
    return "\n".join(lines)
