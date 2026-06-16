from __future__ import annotations

from pathlib import Path

from agent_core.config.validation import is_relative_to
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode


def resolve_memory_dir(template: str, *, home_dir: Path, project_dir: Path | None = None) -> Path:
    expanded = template.replace("${SOONG_AGENT_HOME}", str(home_dir))
    if project_dir is not None:
        expanded = expanded.replace("<project>", str(project_dir))
    memory_dir = Path(expanded).expanduser().resolve()
    home_root = home_dir.resolve()
    if memory_dir != home_root and not is_relative_to(memory_dir, home_root):
        raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"memory_dir must be inside SOONG_AGENT_HOME: {memory_dir}")
    return memory_dir


def ensure_memory_write_allowed(path: Path, *, home_dir: Path, memory_dir: Path | None = None) -> None:
    memory_dir = (memory_dir or (home_dir / "memory")).resolve()
    target = path.expanduser().resolve()
    home_root = home_dir.resolve()
    if memory_dir != home_root and not is_relative_to(memory_dir, home_root):
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, f"memory root outside SOONG_AGENT_HOME: {memory_dir}")
    allowed_dirs = [memory_dir / "user", memory_dir / "feedback", memory_dir / "reference"]
    if target != memory_dir / "MEMORY.md" and not any(is_relative_to(target, root) for root in allowed_dirs):
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, f"memory write outside allowed paths: {target}")
