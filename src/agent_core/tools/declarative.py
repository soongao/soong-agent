from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any

from agent_core.config.validation import is_relative_to
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.output_limits import truncate_bytes
from agent_core.tools.registry import ToolRegistry
from agent_core.types.content import ArtifactRefBlock, JsonBlock
from agent_core.types.common import ErrorPayload
from agent_core.types.tools import ToolDefinition, ToolResult


_ALLOWED_DECLARATIVE_FIELDS = {
    "name",
    "description",
    "input_schema",
    "permission",
    "tags",
    "command_type",
    "command",
    "args",
    "working_dir",
    "env_allowlist",
    "env",
    "timeout_ms",
    "stdout_limit_bytes",
    "stderr_limit_bytes",
}
_TEMPLATE_RE = re.compile(r"{{\s*([^{}]+?)\s*}}")
_ARG_FIELD_RE = re.compile(r"^args\.([A-Za-z_][A-Za-z0-9_]*)$")


def load_declarative_tools(registry: ToolRegistry, home_dir: Path) -> None:
    tools_dir = home_dir / "tools"
    if not tools_dir.exists():
        return
    for path in sorted(tools_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        _validate_declarative_tool(data, path=path)
        definition = ToolDefinition(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("input_schema") or {"type": "object", "properties": {}},
            permission=data.get("permission", "write"),
            tags=set(data.get("tags") or ["declarative"]),
            metadata={"path": str(path), "declarative": data},
        )
        registry.register_tool(definition, _make_handler(data))


def _make_handler(data: dict[str, Any]):
    async def handler(context: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
        command_type = data.get("command_type", "exec")
        cwd = _declarative_cwd(context, data)
        timeout_ms = min(int(data.get("timeout_ms") or context.config.tools.default_timeout_ms), context.config.tools.max_timeout_ms)
        env_allowlist = set(context.config.tools.env_allowlist)
        env_allowlist.update(str(item) for item in data.get("env_allowlist") or [])
        env = {key: value for key, value in os.environ.items() if key in env_allowlist}
        env.update({str(key): str(value) for key, value in (data.get("env") or {}).items() if key in env_allowlist})
        stdin_payload = json.dumps(
            {
                "tool_name": data["name"],
                "arguments": args,
                "session_id": context.session_id,
                "run_id": context.run_id,
                "agent_id": context.agent_id,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        if command_type == "shell":
            command = _shell_command(data, args)
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            argv = _exec_argv(data, args)
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(stdin_payload), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise AgentCoreError(ErrorCode.TIMEOUT, "declarative tool timed out") from exc
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        parsed = _tool_result_from_stdout(data["name"], stdout)
        if parsed is not None:
            metadata = dict(parsed.metadata)
            metadata["exit_code"] = proc.returncode
            if stderr:
                metadata["stderr"] = stderr
            if proc.returncode != 0:
                error = parsed.error or ErrorPayload(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"declarative tool exited with {proc.returncode}",
                    details={"stderr": stderr} if stderr else {},
                )
                return parsed.model_copy(update={"is_error": True, "error": error, "metadata": metadata})
            return parsed.model_copy(update={"metadata": metadata})
        stdout_limit = int(data.get("stdout_limit_bytes") or context.config.tools.stdout_limit_bytes)
        stderr_limit = int(data.get("stderr_limit_bytes") or context.config.tools.stderr_limit_bytes)
        stdout_text, stdout_truncated = truncate_bytes(stdout, stdout_limit)
        stderr_text, stderr_truncated = truncate_bytes(stderr, stderr_limit)
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
                filename=f"{data['name'].replace('.', '_')}_stdout.txt",
                summary="truncated declarative stdout",
            )
            content[0].data["stdout_artifact_id"] = artifact.artifact_id  # type: ignore[index]
            content.append(ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=artifact.summary, mime_type="text/plain"))
            metadata["stdout_artifact_id"] = artifact.artifact_id
        if stderr_truncated:
            artifact = context.artifact_manager.write_text(
                session_id=context.session_id,
                text=stderr,
                filename=f"{data['name'].replace('.', '_')}_stderr.txt",
                summary="truncated declarative stderr",
            )
            content[0].data["stderr_artifact_id"] = artifact.artifact_id  # type: ignore[index]
            content.append(ArtifactRefBlock(artifact_id=artifact.artifact_id, summary=artifact.summary, mime_type="text/plain"))
            metadata["stderr_artifact_id"] = artifact.artifact_id
        error = None
        if proc.returncode != 0:
            error = ErrorPayload(code=ErrorCode.INTERNAL_ERROR, message=f"declarative tool exited with {proc.returncode}")
        return ToolResult(
            tool_call_id="",
            tool_name=data["name"],
            content=content,
            is_error=proc.returncode != 0,
            error=error,
            metadata=metadata,
        )

    return handler


def _validate_declarative_tool(data: dict[str, Any], *, path: Path) -> None:
    unknown = sorted(set(data) - _ALLOWED_DECLARATIVE_FIELDS)
    if unknown:
        raise AgentCoreError(
            ErrorCode.VALIDATION_ERROR,
            f"declarative tool {path.name} contains unknown field: {unknown[0]}",
        )
    if not data.get("name"):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"declarative tool {path.name} missing name")
    command_type = data.get("command_type", "exec")
    if command_type not in {"exec", "shell"}:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported declarative command_type: {command_type}")
    tags = {str(tag) for tag in data.get("tags") or []}
    permission = data.get("permission", "write")
    if command_type == "shell" and permission != "write" and "dangerous" not in tags:
        raise AgentCoreError(
            ErrorCode.VALIDATION_ERROR,
            "declarative shell tools must use write permission or dangerous tag",
        )


def _exec_argv(data: dict[str, Any], args: dict[str, Any]) -> list[str]:
    raw = data.get("command") or data.get("args") or []
    if not isinstance(raw, list):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "declarative exec command must be an argv list")
    argv = [_render_template(str(item), args, shell=False) for item in raw]
    if not argv:
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "declarative tool command is empty")
    return argv


def _shell_command(data: dict[str, Any], args: dict[str, Any]) -> str:
    command = data.get("command")
    if not isinstance(command, str) or not command.strip():
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "declarative shell command must be a non-empty string")
    return _render_template(command, args, shell=True)


def _render_template(template: str, args: dict[str, Any], *, shell: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        field_match = _ARG_FIELD_RE.match(expression)
        if field_match is None:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"unsupported declarative template expression: {expression}")
        key = field_match.group(1)
        if key not in args:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, f"missing declarative template argument: {key}")
        value = str(args[key])
        return shlex.quote(value) if shell else value

    return _TEMPLATE_RE.sub(replace, template)


def _tool_result_from_stdout(tool_name: str, stdout: str) -> ToolResult | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "content" not in data:
        return None
    payload = {
        "tool_call_id": data.get("tool_call_id") or "",
        "tool_name": data.get("tool_name") or tool_name,
        "content": data.get("content") or [],
        "is_error": bool(data.get("is_error", False)),
        "error": data.get("error"),
        "metadata": data.get("metadata") or {},
    }
    try:
        return ToolResult.model_validate(payload)
    except Exception as exc:
        raise AgentCoreError(ErrorCode.SCHEMA_ERROR, "declarative stdout ToolResult is invalid") from exc


def _declarative_cwd(context: ToolExecutionContext, data: dict[str, Any]) -> Path:
    raw = data.get("working_dir") or str(context.project_dir)
    path = Path(str(raw).replace("<project>", str(context.project_dir))).expanduser()
    if not path.is_absolute():
        path = context.project_dir / path
    resolved = path.resolve()
    allowed = [context.project_dir.resolve()]
    allowed.extend(Path(root).expanduser().resolve() for root in context.config.tools.allowed_write_roots)
    if context.config.tools.allow_tmp_write:
        allowed.append(Path("/tmp").resolve())
    if not any(is_relative_to(resolved, root) for root in allowed):
        raise AgentCoreError(ErrorCode.WRITE_OUTSIDE_ALLOWED_ROOTS, f"declarative working_dir outside allowed roots: {resolved}")
    if not resolved.is_dir():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"declarative working_dir not found: {resolved}")
    return resolved
