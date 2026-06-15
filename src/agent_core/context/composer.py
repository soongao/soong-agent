from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from agent_core.context.instructions import build_instruction_catalog, instruction_catalog_text
from agent_core.context.skills import build_skill_catalog, skill_catalog_text
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
    skills = build_skill_catalog(home_dir)
    blocks = [
        SystemBlock(
            block_id=block_id,
            source="package_asset",
            content=_read_system_asset(filename),
            priority=priority,
            dynamic=False,
            metadata={"asset_path": f"prompts/system/{filename}"},
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


def build_dynamic_system_blocks(context_state: Any | None, *, home_dir: Path | None = None, memory_enabled: bool = True) -> list[SystemBlock]:
    blocks: list[SystemBlock] = []
    if home_dir is not None and memory_enabled:
        catalog_path = home_dir / "memory" / "MEMORY.md"
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
) -> list[SystemBlock]:
    return build_static_system_blocks(home_dir=home_dir, project_dir=project_dir) + build_dynamic_system_blocks(
        context_state,
        home_dir=home_dir,
        memory_enabled=memory_enabled,
    )


def _read_system_asset(filename: str) -> str:
    return resources.files("agent_core.assets.prompts.system").joinpath(filename).read_text(encoding="utf-8")


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
