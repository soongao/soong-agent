from __future__ import annotations

from pathlib import Path

from agent_core.config.validation import is_relative_to
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.tools.execution import ToolExecutionContext


def resolve_user_path(value: str, *, context: ToolExecutionContext) -> Path:
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = context.effective_cwd / raw
    return raw.resolve()


def ensure_write_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    allowed = [context.project_dir.resolve()]
    allowed.extend(Path(root).expanduser().resolve() for root in context.config.tools.allowed_write_roots)
    if context.config.tools.allow_tmp_write:
        allowed.append(Path("/tmp").resolve())
    if not any(is_relative_to(path, root) for root in allowed):
        raise AgentCoreError(ErrorCode.WRITE_OUTSIDE_ALLOWED_ROOTS, f"write outside allowed roots: {path}")


def ensure_read_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    return None


def ensure_cwd_allowed(path: Path, *, context: ToolExecutionContext) -> None:
    if not path.is_dir():
        raise AgentCoreError(ErrorCode.FILE_NOT_FOUND, f"cwd not found: {path}")
    ensure_write_allowed(path, context=context)


def looks_binary(path: Path) -> bool:
    sample = path.read_bytes()[:2048]
    return b"\x00" in sample
