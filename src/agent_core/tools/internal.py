from __future__ import annotations

import hashlib
from pathlib import Path

from agent_core.context.skills import find_skill_by_name
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolDefinition


def register_internal_tools(registry: ToolRegistry) -> None:
    registry.register_tool(
        ToolDefinition(
            name="internal.load_skill",
            description="Load a user-level skill body by skill name.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            permission="readonly",
            tags={"internal", "readonly"},
        ),
        load_skill,
    )
    registry.register_tool(
        ToolDefinition(
            name="internal.recall_memory",
            description="Recall user-level memory snippets.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
            permission="readonly",
            tags={"internal", "readonly"},
        ),
        recall_memory,
    )


async def load_skill(context: ToolExecutionContext, args: dict) -> dict:
    name = str(args["name"])
    entry = find_skill_by_name(context.home_dir, name)
    if entry is None:
        raise AgentCoreError(ErrorCode.SKILL_NOT_FOUND, f"skill not found: {name}")
    if entry.get("error") == "duplicate":
        raise AgentCoreError(ErrorCode.SKILL_LOAD_FAILED, f"duplicate skill name: {name}")
    path = Path(entry["path"]).resolve()
    body = _body_without_frontmatter(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    already_loaded = False
    if context.services and "context_state" in context.services:
        already_loaded = context.services["context_state"].mark_skill(name, path, body)
    return {
        "node_type": "skill_context",
        "name": name,
        "path": str(path),
        "hash": digest,
        "body": body,
        "content": _skill_context_text(name=name, body=body),
        "already_loaded": already_loaded,
    }


async def recall_memory(context: ToolExecutionContext, args: dict) -> dict:
    query = str(args["query"])
    top_k = int(args.get("top_k") or context.config.memory.recall_top_k)
    memory_dir = Path(context.config.memory.memory_dir.replace("${SOONG_AGENT_HOME}", str(context.home_dir))).expanduser()
    selected_by_model = False
    selected_paths: list[str] = []
    runtime = context.services.get("runtime") if context.services else None
    if runtime is not None and hasattr(runtime, "select_memory"):
        selection = await runtime.select_memory(session_id=context.session_id, query=query, top_k=top_k)
        selected_by_model = bool(selection.get("selected_by_model"))
        selected_paths = [str(item) for item in selection.get("selected_paths") or []]
    if not selected_paths:
        selected_paths = _fallback_selected_paths(memory_dir, query, top_k=top_k)
    matches: list[dict] = []
    budget = int(context.config.memory.memory_context_token_budget) * 4
    used = 0
    for relative_path in selected_paths[:top_k]:
        path = (memory_dir / relative_path).resolve()
        if not _inside(path, memory_dir.resolve()) or path.parent.name not in {"user", "feedback", "reference"}:
            continue
        if not path.exists() or path.name == "MEMORY.md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        remaining = max(budget - used, 0)
        if remaining <= 0:
            break
        content = text[:remaining]
        used += len(content)
        metadata = _frontmatter(text)
        matches.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(memory_dir)),
                "id": metadata.get("id") or path.stem,
                "category": metadata.get("category") or path.parent.name,
                "content": content,
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    if context.services and "context_state" in context.services:
        state = context.services["context_state"]
        existing = {(item.get("id"), item.get("hash")) for loaded in state.memory_contexts for item in loaded.get("matches", [])}
        new_matches = [match for match in matches if (match.get("id"), match.get("hash")) not in existing]
        already_recalled = bool(matches) and not new_matches
        if new_matches:
            state.memory_contexts.append(
                {
                    "query": query,
                    "matches": new_matches,
                    "selected_by_model": selected_by_model,
                    "recalled_memory_ids": [match.get("id") for match in new_matches],
                    "categories": sorted({match.get("category") for match in new_matches if match.get("category")}),
                    "source_paths": [match.get("path") for match in new_matches],
                }
            )
        else:
            already_recalled = bool(matches)
    else:
        already_recalled = False
    return {
        "node_type": "memory_context",
        "query": query,
        "matches": matches,
        "selected_by_model": selected_by_model,
        "already_recalled": already_recalled,
        "content": _memory_context_text(query=query, matches=matches),
        "metadata": {
            "query": query,
            "selected_by_model": selected_by_model,
            "recalled_memory_ids": [match.get("id") for match in matches if match.get("id")],
            "categories": sorted({match.get("category") for match in matches if match.get("category")}),
            "source_paths": [match.get("path") for match in matches if match.get("path")],
        },
    }


def _body_without_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def _skill_context_text(*, name: str, body: str) -> str:
    return f'<skill name="{name}">\n{body}\n</skill>'


def _memory_context_text(*, query: str, matches: list[dict]) -> str:
    ids = ",".join(str(match.get("id") or "") for match in matches if match.get("id"))
    categories = ",".join(str(match.get("category") or "") for match in matches if match.get("category"))
    lines = [f'<memory ids="{ids}" categories="{categories}" query="{query}">']
    for match in matches:
        content = match.get("content") or match.get("snippet") or match.get("text") or ""
        lines.append(f"## {match.get('id') or match.get('path')}")
        lines.append(str(content))
        lines.append("")
    lines.append("</memory>")
    return "\n".join(lines)


def _fallback_selected_paths(memory_dir: Path, query: str, *, top_k: int) -> list[str]:
    query_lower = query.lower()
    selected: list[str] = []
    if not memory_dir.exists():
        return selected
    for path in sorted(memory_dir.rglob("*.md")):
        if path.name == "MEMORY.md" or path.parent.name not in {"user", "feedback", "reference"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if query_lower in text.lower() or query_lower in path.name.lower():
            selected.append(str(path.relative_to(memory_dir)))
        if len(selected) >= top_k:
            break
    return selected


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
