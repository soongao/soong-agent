from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from agent_core.config.validation import is_relative_to
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.types.common import ErrorPayload
from agent_core.types.content import ArtifactRefBlock, JsonBlock, TextBlock
from agent_core.types.tools import ToolDefinition, ToolResult
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.output_limits import truncate_bytes
from agent_core.tools.registry import ToolRegistry

READ_LINE_BYTE_LIMIT = 4096


def register_builtin_code_tools(registry: ToolRegistry) -> None:
    registry.register_tool(
        ToolDefinition(
            name="code.read_file",
            description="Read a text file by line range.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 200},
                },
                "required": ["path"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        read_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.list_dir",
            description="List directory entries.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["path"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        list_dir,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.search",
            description="Search text in files using ripgrep.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": ["string", "null"]},
                    "glob": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["query"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        search,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.write_file",
            description="Write a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "create_dirs": {"type": "boolean", "default": True},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
            permission="write",
            tags={"code", "filesystem", "write"},
        ),
        write_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.edit_file",
            description="Edit a file by exact replacement or a single-file unified diff.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {"type": ["array", "null"]},
                    "unified_diff": {"type": ["string", "null"]},
                    "create_if_missing": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
            permission="write",
            tags={"code", "filesystem", "write"},
        ),
        edit_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.run_command",
            description="Run a command using argv list, without a shell string.",
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": ["string", "null"]},
                    "timeout_ms": {"type": ["integer", "null"]},
                },
                "required": ["argv"],
            },
            permission="write",
            tags={"code", "dangerous"},
        ),
        run_command,
    )


async def read_file(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_user_path(args["path"], context=context)
    _ensure_read_allowed(path, context=context)
    start_line = int(args.get("start_line") or 1)
    max_lines = int(args.get("max_lines") or 200)
    if start_line < 1 or max_lines < 1 or max_lines > 1000:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "start_line/max_lines out of range")
    if not path.exists():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"file not found: {path}")
    if path.is_dir():
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"path is a directory: {path}")
    if _looks_binary(path):
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
    already_loaded = _mark_instruction_if_needed(context, path)
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
    path = _resolve_user_path(args["path"], context=context)
    _ensure_read_allowed(path, context=context)
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
    return _artifact_json_if_large(
        context,
        tool_name="code.list_dir",
        data=result,
        filename="list_dir.json",
        summary="large directory listing",
    )


async def search(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"])
    path = _resolve_user_path(args.get("path") or str(context.project_dir), context=context)
    _ensure_read_allowed(path, context=context)
    glob = args.get("glob")
    limit = args.get("limit")
    limit = int(limit) if limit is not None else None
    argv = ["rg", "--line-number", "--column", "--no-heading", query, str(path)]
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
    return _artifact_json_if_large(
        context,
        tool_name="code.search",
        data=result,
        filename="search_results.json",
        summary="large search results",
    )


async def write_file(context: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_user_path(args["path"], context=context)
    _ensure_write_allowed(path, context=context)
    content = str(args["content"])
    create_dirs = bool(args.get("create_dirs", True))
    overwrite = bool(args.get("overwrite", False))
    if path.exists() and not overwrite:
        raise AgentCoreError(ErrorCode.PATH_CONFLICT, f"path exists and overwrite=false: {path}")
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
    path = _resolve_user_path(args["path"], context=context)
    _ensure_write_allowed(path, context=context)
    edits = args.get("edits")
    unified_diff = args.get("unified_diff")
    if (edits is None) == (unified_diff is None):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "exactly one of edits or unified_diff is required")
    create_if_missing = bool(args.get("create_if_missing", False))
    if not path.exists():
        if not create_if_missing:
            raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"file not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
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
        updated, applied = _apply_simple_unified_diff(path=path, original=original, diff=str(unified_diff))
    path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "edits_applied": applied, "bytes_written": len(updated.encode("utf-8"))}


async def run_command(context: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    argv = args.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "argv must be a non-empty list of strings")
    cwd = _resolve_user_path(args.get("cwd") or str(context.project_dir), context=context)
    _ensure_cwd_allowed(cwd, context=context)
    timeout_ms = int(args.get("timeout_ms") or context.config.tools.default_timeout_ms)
    timeout_ms = min(timeout_ms, context.config.tools.max_timeout_ms)
    env = {key: value for key, value in os.environ.items() if key in context.config.tools.env_allowlist}
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError as exc:
        try:
            proc.kill()
        except Exception:
            pass
        raise AgentCoreError(ErrorCode.TIMEOUT, "command timed out") from exc
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    stdout_text, stdout_truncated = truncate_bytes(stdout, context.config.tools.stdout_limit_bytes)
    stderr_text, stderr_truncated = truncate_bytes(stderr, context.config.tools.stderr_limit_bytes)
    content = [
        JsonBlock(
            data={
                "exit_code": proc.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_artifact_id": None,
                "stderr_artifact_id": None,
                "truncated": stdout_truncated or stderr_truncated,
            }
        )
    ]
    metadata: dict[str, Any] = {}
    if stdout_truncated:
        artifact = context.artifact_manager.write_text(
            session_id=context.session_id,
            text=stdout,
            filename="stdout.txt",
            summary="truncated command stdout",
        )
        content.append(ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=artifact.summary, mime_type="text/plain"))
        content[0].data["stdout_artifact_id"] = artifact.artifact_id  # type: ignore[index]
        metadata["stdout_artifact_id"] = artifact.artifact_id
    if stderr_truncated:
        artifact = context.artifact_manager.write_text(
            session_id=context.session_id,
            text=stderr,
            filename="stderr.txt",
            summary="truncated command stderr",
        )
        content.append(ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=artifact.summary, mime_type="text/plain"))
        content[0].data["stderr_artifact_id"] = artifact.artifact_id  # type: ignore[index]
        metadata["stderr_artifact_id"] = artifact.artifact_id
    error = None
    is_error = proc.returncode != 0
    if is_error:
        error = ErrorPayload(code=ErrorCode.INTERNAL_ERROR, message=f"command exited with {proc.returncode}")
    return ToolResult(
        tool_call_id="",
        tool_name="code.run_command",
        content=content,
        is_error=is_error,
        error=error,
        metadata=metadata,
    )


def _resolve_user_path(value: str, *, context: ToolExecutionContext) -> Path:
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = context.effective_cwd / raw
    return raw.resolve()


def _ensure_write_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    allowed = [context.project_dir.resolve()]
    allowed.extend(Path(root).expanduser().resolve() for root in context.config.tools.allowed_write_roots)
    if context.config.tools.allow_tmp_write:
        allowed.append(Path("/tmp").resolve())
    if not any(is_relative_to(path, root) for root in allowed):
        raise AgentCoreError(ErrorCode.WRITE_OUTSIDE_ALLOWED_ROOTS, f"write outside allowed roots: {path}")


def _ensure_read_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    return None


def _ensure_cwd_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    if not path.is_dir():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"cwd not found: {path}")
    _ensure_write_allowed(path, context=context)


def _looks_binary(path: Path) -> bool:
    sample = path.read_bytes()[:2048]
    return b"\x00" in sample


def _mark_instruction_if_needed(context: ToolExecutionContext, path: Path) -> bool | None:
    if path.name not in {"CLAUDE.md", "AGENTS.md"} and "rules" not in path.parts:
        return None
    if not context.services or "context_state" not in context.services:
        return None
    state = context.services["context_state"]
    return state.mark_instruction(path)


def _apply_simple_unified_diff(*, path: Path, original: str, diff: str) -> tuple[str, int]:
    lines = diff.splitlines(keepends=True)
    _validate_single_file_diff_path(path=path, lines=lines)
    old_lines = original.splitlines(keepends=True)
    result: list[str] = []
    old_index = 0
    applied = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("@@"):
            i += 1
            continue
        try:
            old_spec = line.split(" ")[1]
            start = int(old_spec.split(",")[0].lstrip("-"))
        except Exception as exc:
            raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "invalid hunk header") from exc
        target_index = max(start - 1, 0)
        result.extend(old_lines[old_index:target_index])
        old_index = target_index
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            hunk_line = lines[i]
            marker = hunk_line[:1]
            body = hunk_line[1:]
            if marker == " ":
                if old_index >= len(old_lines) or old_lines[old_index] != body:
                    raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "hunk context mismatch")
                result.append(old_lines[old_index])
                old_index += 1
            elif marker == "-":
                if old_index >= len(old_lines) or old_lines[old_index] != body:
                    raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "hunk removal mismatch")
                old_index += 1
                applied += 1
            elif marker == "+":
                result.append(body)
                applied += 1
            elif hunk_line.startswith("\\ No newline"):
                pass
            i += 1
    if applied == 0:
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "no patch hunks applied")
    result.extend(old_lines[old_index:])
    return "".join(result), applied


def _validate_single_file_diff_path(*, path: Path, lines: list[str]) -> None:
    old_headers = [line for line in lines if line.startswith("--- ")]
    new_headers = [line for line in lines if line.startswith("+++ ")]
    if len(old_headers) > 1 or len(new_headers) > 1:
        raise AgentCoreError(ErrorCode.PATCH_PATH_MISMATCH, "unified diff must modify exactly one file")
    if any(line.startswith(("rename from ", "rename to ", "deleted file mode ", "new file mode ", "Binary files ")) for line in lines):
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "rename/delete/binary patches are not supported")
    if not old_headers and not new_headers:
        return
    if len(old_headers) != 1 or len(new_headers) != 1:
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "unified diff must contain matching --- and +++ headers")
    old_path = _diff_header_path(old_headers[0])
    new_path = _diff_header_path(new_headers[0])
    if old_path == "/dev/null" or new_path == "/dev/null":
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "create/delete patches are not supported")
    if not (_diff_path_matches_target(path, old_path) and _diff_path_matches_target(path, new_path)):
        raise AgentCoreError(ErrorCode.PATCH_PATH_MISMATCH, "unified diff path does not match target path")


def _diff_header_path(header: str) -> str:
    value = header[4:].strip()
    if "\t" in value:
        value = value.split("\t", 1)[0]
    if " " in value:
        value = value.split(" ", 1)[0]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def _diff_path_matches_target(path: Path, diff_path: str) -> bool:
    if not diff_path:
        return False
    candidate = Path(diff_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve() == path.resolve()
        except OSError:
            return candidate == path
    target_parts = path.parts
    diff_parts = candidate.parts
    return len(diff_parts) <= len(target_parts) and tuple(target_parts[-len(diff_parts) :]) == diff_parts


def _artifact_json_if_large(
    context: ToolExecutionContext,
    *,
    tool_name: str,
    data: dict[str, Any],
    filename: str,
    summary: str,
) -> dict[str, Any] | ToolResult:
    import json

    text = json.dumps(data, ensure_ascii=False, indent=2)
    limit = context.config.tools.stdout_limit_bytes
    if len(text.encode("utf-8")) <= limit:
        return data
    artifact = context.artifact_manager.write_text(
        session_id=context.session_id,
        text=text,
        filename=filename,
        mime_type="application/json",
        summary=summary,
    )
    return ToolResult(
        tool_call_id="",
        tool_name=tool_name,
        content=[
            JsonBlock(
                data={
                    "truncated": True,
                    "artifact_id": artifact.artifact_id,
                    "summary": summary,
                }
            ),
            ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=summary, mime_type="application/json"),
        ],
        metadata={"artifact_ids": [artifact.artifact_id], "output_artifact_id": artifact.artifact_id},
    )
