from __future__ import annotations

from pathlib import Path

from agent_core.config.validation import is_relative_to
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode


def ensure_memory_write_allowed(path: Path, *, home_dir: Path) -> None:
    memory_dir = (home_dir / "memory").resolve()
    target = path.expanduser().resolve()
    allowed_dirs = [
        memory_dir,
        memory_dir / "user",
        memory_dir / "feedback",
        memory_dir / "reference",
    ]
    if target != memory_dir / "MEMORY.md" and not any(is_relative_to(target, root) for root in allowed_dirs[1:]):
        raise AgentCoreError(ErrorCode.MEMORY_WRITE_FAILED, f"memory write outside allowed paths: {target}")

