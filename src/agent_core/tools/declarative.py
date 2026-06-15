from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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


def load_declarative_tools(registry: ToolRegistry, home_dir: Path) -> None:
    tools_dir = home_dir / "tools"
    if not tools_dir.exists():
        return
    for path in sorted(tools_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
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
        if command_type != "exec":
            raise AgentCoreError(ErrorCode.UNSUPPORTED_CAPABILITY, "only declarative exec tools are implemented")
        argv = [_render_arg(str(item), args) for item in data.get("command", [])]
        if not argv:
            argv = [_render_arg(str(item), args) for item in data.get("args", [])]
        if not argv:
            raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "declarative tool command is empty")
        cwd = _declarative_cwd(context, data)
        timeout_ms = min(int(data.get("timeout_ms") or context.config.tools.default_timeout_ms), context.config.tools.max_timeout_ms)
        env_allowlist = set(context.config.tools.env_allowlist)
        env_allowlist.update(str(item) for item in data.get("env_allowlist") or [])
        env = {key: value for key, value in os.environ.items() if key in env_allowlist}
        env.update({str(key): str(value) for key, value in (data.get("env") or {}).items() if key in env_allowlist})
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise AgentCoreError(ErrorCode.TIMEOUT, "declarative tool timed out") from exc
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
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


def _render_arg(template: str, args: dict[str, Any]) -> str:
    rendered = template
    for key, value in args.items():
        rendered = rendered.replace("{{args." + key + "}}", str(value))
    return rendered


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
