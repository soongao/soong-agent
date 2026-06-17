from __future__ import annotations

import asyncio
import os
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.builtin_code.pathing import ensure_cwd_allowed, resolve_user_path
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.output_limits import truncate_bytes
from agent_core.types.common import ErrorPayload
from agent_core.types.content import ArtifactRefBlock, JsonBlock
from agent_core.types.tools import ToolResult


async def run_command(context: ToolExecutionContext, args: dict[str, Any]) -> ToolResult:
    argv = args.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise AgentCoreError(ErrorCode.VALIDATION_ERROR, "argv must be a non-empty list of strings")
    cwd = resolve_user_path(args.get("cwd") or str(context.project_dir), context=context)
    ensure_cwd_allowed(cwd, context=context)
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
        await kill_and_reap(proc)
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


async def kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    try:
        if proc.returncode is None:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        await proc.communicate()
    except Exception:
        try:
            await proc.wait()
        except Exception:
            pass
