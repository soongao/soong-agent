from __future__ import annotations

import sys

from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind, PermissionRequest


async def stdin_permission_callback(request: PermissionRequest) -> PermissionDecision:
    print(
        f"Permission required for {request.tool_name} ({request.permission})\n"
        f"Target: {request.target_scope or request.cwd}\n"
        "1) allow once\n"
        "2) allow for session\n"
        "3) deny\n"
        "> ",
        end="",
        flush=True,
    )
    try:
        value = sys.stdin.readline()
    except Exception:
        value = ""
    normalized = value.strip().lower()
    if normalized in {"1", "allow once", "allow_once"}:
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)
    if normalized in {"2", "allow for session", "allow_for_session"}:
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_FOR_SESSION)
    return PermissionDecision(decision=PermissionDecisionKind.DENY)

