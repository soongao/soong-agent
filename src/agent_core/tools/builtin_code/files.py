from __future__ import annotations

import asyncio
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.builtin_code.artifacts import artifact_json_if_large
from agent_core.tools.builtin_code.patching import apply_simple_unified_diff
from agent_core.tools.builtin_code.pathing import ensure_read_allowed, ensure_write_allowed, looks_binary, resolve_user_path
from agent_core.tools.execution import ToolExecutionContext

READ_LINE_BYTE_LIMIT = 4096


async def read_file(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = resolve_user_path(args["path"], context=context)
    ensure_read_allowed(path, context=context)
    start_line = int(args.get("start_line") or 1)
    max_lines = int(args.get("max_lines") or 200)
    if start_line < 1 or max_lines < 1 or max_lines > 1000:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "start_line/max_lines out of range")
    if not path.exists():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"file not found: {path}")
    if path.is_dir():
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"path is a directory: {path}")
    if looks_binary(path):
        return {
            "path": str(path),
            "content": "",
            "truncated": False,
            "next_start_line": None,
            "truncated_lines": [],
            "binary": True,
            "already_loaded": None,
        }
    lines: list[str] = []
    truncated_lines: list[int] = []
    next_start_line: int | None = None
    with path.open("rb") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if line_number < start_line:
                continue
            if len(lines) >= max_lines:
                next_start_line = line_number
                break
            if len(raw_line) > READ_LINE_BYTE_LIMIT:
                raw_line = raw_line[:READ_LINE_BYTE_LIMIT]
                truncated_lines.append(line_number)
            lines.append(raw_line.decode("utf-8", errors="replace"))
    already_loaded = mark_instruction_if_needed(context, path)
    return {
        "path": str(path),
        "content": "".join(lines),
        "truncated": next_start_line is not None or bool(truncated_lines),
        "next_start_line": next_start_line,
        "truncated_lines": truncated_lines,
        "binary": False,
        "already_loaded": already_loaded,
    }


async def list_dir(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = resolve_user_path(args["path"], context=context)
    ensure_read_allowed(path, context=context)
    recursive = bool(args.get("recursive", False))
    limit = args.get("limit")
    limit = int(limit) if limit is not None else None
    if not path.exists():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"directory not found: {path}")
    if not path.is_dir():
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"path is not a directory: {path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    entries: list[dict[str, Any]] = []
    truncated = False
    for entry in sorted(iterator):
        if limit is not None and len(entries) >= limit:
            truncated = True
            break
        entries.append(
            {
                "path": str(entry),
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else None,
            }
        )
    result = {"entries": entries, "truncated": truncated}
    return artifact_json_if_large(
        context,
        tool_name="code.list_dir",
        data=result,
        filename="list_dir.json",
        summary="large directory listing",
    )


async def search(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"])
    path = resolve_user_path(args.get("path") or str(context.project_dir), context=context)
    ensure_read_allowed(path, context=context)
    glob = args.get("glob")
    limit = args.get("limit")
    limit = int(limit) if limit is not None else None
    argv = ["rg", "--line-number", "--column", "--no-heading", "--", query, str(path)]
    if glob:
        argv[1:1] = ["--glob", str(glob)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        proc = await asyncio.create_subprocess_exec(
            "grep",
            "-R",
            "-n",
            "--",
            query,
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    stdout, stderr = await proc.communicate()
    if proc.returncode not in (0, 1):
        raise AgentCoreError(ErrorCode.INTERNAL_ERROR, stderr.decode("utf-8", errors="replace"))
    matches: list[dict[str, Any]] = []
    truncated = False
    for raw in stdout.decode("utf-8", errors="replace").splitlines():
        if limit is not None and len(matches) >= limit:
            truncated = True
            break
        parts = raw.split(":", 3)
        if len(parts) >= 4 and parts[1].isdigit():
            matches.append({"path": parts[0], "line": int(parts[1]), "column": int(parts[2]), "text": parts[3]})
        elif len(parts) >= 3 and parts[1].isdigit():
            matches.append({"path": parts[0], "line": int(parts[1]), "column": None, "text": ":".join(parts[2:])})
        else:
            matches.append({"path": None, "line": None, "column": None, "text": raw})
    result = {"matches": matches, "truncated": truncated}
    return artifact_json_if_large(
        context,
        tool_name="code.search",
        data=result,
        filename="search_results.json",
        summary="large search results",
    )


async def write_file(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = resolve_user_path(args["path"], context=context)
    ensure_write_allowed(path, context=context)
    content = str(args["content"])
    create_dirs = bool(args.get("create_dirs", True))
    overwrite = bool(args.get("overwrite", False))
    if path.exists() and not overwrite:
        raise AgentCoreError(ErrorCode.PATH_CONFLICT, f"path exists and overwrite=false: {path}")
    if not create_dirs and not path.parent.exists():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"parent directory not found: {path.parent}")
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "bytes_written": len(content.encode("utf-8")),
        "created": created,
        "overwritten": not created,
    }


async def edit_file(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = resolve_user_path(args["path"], context=context)
    ensure_write_allowed(path, context=context)
    edits = args.get("edits")
    unified_diff = args.get("unified_diff")
    if (edits is None) == (unified_diff is None):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "exactly one of edits or unified_diff is required")
    create_if_missing = bool(args.get("create_if_missing", False))
    existed = path.exists()
    if not existed:
        if not create_if_missing:
            raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"file not found: {path}")
        original = ""
    else:
        original = path.read_text(encoding="utf-8")
    if edits is not None:
        updated = original
        applied = 0
        for edit in edits:
            old = str(edit["old"])
            new = str(edit["new"])
            replace_all = bool(edit.get("replace_all", False))
            count = updated.count(old)
            if count == 0:
                raise AgentCoreError(ErrorCode.TEXT_NOT_FOUND, "old text not found")
            if count > 1 and not replace_all:
                raise AgentCoreError(ErrorCode.AMBIGUOUS_EDIT, "old text matched multiple locations")
            updated = updated.replace(old, new, -1 if replace_all else 1)
            applied += count if replace_all else 1
    else:
        updated, applied = apply_simple_unified_diff(path=path, original=original, diff=str(unified_diff))
    if not existed:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "edits_applied": applied, "bytes_written": len(updated.encode("utf-8"))}


def mark_instruction_if_needed(context: ToolExecutionContext, path) -> bool | None:
    if not context.services or "context_state" not in context.services:
        return None
    from agent_core.context.instructions import build_instruction_catalog

    try:
        active_instruction_paths = {
            entry.path.resolve()
            for entry in build_instruction_catalog(home_dir=context.home_dir, project_dir=context.project_dir)[0]
        }
    except OSError:
        return None
    if path.resolve() not in active_instruction_paths:
        return None
    state = context.services["context_state"]
    return state.mark_instruction(path)
