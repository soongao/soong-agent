from __future__ import annotations

import fnmatch
from collections.abc import Awaitable, Callable
from pathlib import Path

from agent_core.config.models import AgentCoreConfig
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest


PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]


def is_sensitive_path(path: Path, *, patterns: list[str]) -> bool:
    resolved = str(path.expanduser())
    name = path.name
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser()) if pattern.startswith("~") else pattern
        if pattern.startswith("~"):
            try:
                if Path(resolved).resolve().is_relative_to(Path(expanded).resolve()):
                    return True
            except OSError:
                if resolved.startswith(expanded):
                    return True
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(resolved, pattern):
            return True
    return False


def needs_permission(*, permission: str, tags: set[str], target: Path | None, config: AgentCoreConfig) -> bool:
    if permission == "write":
        return True
    if "dangerous" in tags or "network" in tags:
        return True
    if target is not None and is_sensitive_path(target, patterns=config.tools.sensitive_paths):
        return True
    return config.permissions.readonly_default != "allow"


def deny_decision(reason: str = "permission denied") -> PermissionDecision:
    return PermissionDecision(decision=PermissionDecisionKind.DENY, reason=reason)

