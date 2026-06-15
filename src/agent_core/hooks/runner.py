from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.errors.codes import ErrorCode
from agent_core.hooks.matcher import hook_matches
from agent_core.types.common import ErrorPayload


@dataclass(frozen=True)
class HookDecision:
    decision: str = "allow"
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    hook: dict[str, Any] | None = None
    error: ErrorPayload | None = None

    @property
    def denied(self) -> bool:
        return self.decision == "deny"


class HookRunner:
    def __init__(self, hooks: list[dict]) -> None:
        self.hooks = hooks

    def deny_reason(self, *, event_type: str, tool_name: str | None = None, tags: set[str] | None = None) -> str | None:
        for hook in self.hooks:
            enriched = dict(hook)
            enriched["_tool_tags"] = sorted(tags or set())
            if hook_matches(hook=enriched, event_type=event_type, tool_name=tool_name):
                action = hook.get("action") or hook.get("decision")
                if action == "deny":
                    return hook.get("reason") or "hook denied"
        return None

    async def run(
        self,
        *,
        event_type: str,
        tool_name: str | None = None,
        tags: set[str] | None = None,
        target_path: str | None = None,
        payload: dict[str, Any] | None = None,
        cwd: Path,
        timeout_ms: int,
        env_allowlist: list[str] | None = None,
    ) -> HookDecision:
        last_decision: HookDecision | None = None
        for hook in self.hooks:
            enriched = dict(hook)
            enriched["_tool_tags"] = sorted(tags or set())
            if not hook_matches(hook=enriched, event_type=event_type, tool_name=tool_name, target_path=target_path):
                continue
            action = hook.get("action") or hook.get("decision")
            if action == "deny":
                return HookDecision(decision="deny", reason=hook.get("reason") or "hook denied", hook=hook)
            if hook.get("type") == "command" or hook.get("command"):
                decision = await _run_command_hook(
                    hook,
                    payload=payload or {},
                    cwd=cwd,
                    timeout_ms=int(hook.get("timeout_ms") or timeout_ms),
                    env_allowlist=env_allowlist,
                )
                if decision.denied:
                    return decision
                last_decision = decision
            else:
                last_decision = HookDecision(hook=hook)
        return last_decision or HookDecision()


async def _run_command_hook(
    hook: dict[str, Any],
    *,
    payload: dict[str, Any],
    cwd: Path,
    timeout_ms: int,
    env_allowlist: list[str] | None = None,
) -> HookDecision:
    command = hook.get("command")
    if not command:
        return HookDecision(error=ErrorPayload(code=ErrorCode.CONFIG_ERROR, message="hook command is empty"), hook=hook)
    env = None
    if env_allowlist is not None:
        env = {key: value for key, value in os.environ.items() if key in set(env_allowlist)}
    if isinstance(command, str):
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    elif isinstance(command, list) and command and all(isinstance(item, str) for item in command):
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        return HookDecision(error=ErrorPayload(code=ErrorCode.CONFIG_ERROR, message="hook command must be a string or argv list"), hook=hook)
    stdin = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return HookDecision(error=ErrorPayload(code=ErrorCode.TIMEOUT, message="hook timed out"), hook=hook)
    stderr = stderr_b.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        return HookDecision(
            error=ErrorPayload(code=ErrorCode.INTERNAL_ERROR, message=f"hook exited with {proc.returncode}", details={"stderr": stderr}),
            logs=[stderr] if stderr else [],
            hook=hook,
        )
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    if not stdout:
        return HookDecision(logs=[stderr] if stderr else [], hook=hook)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return HookDecision(error=ErrorPayload(code=ErrorCode.SCHEMA_ERROR, message="hook stdout is not JSON"), logs=[stderr] if stderr else [], hook=hook)
    return HookDecision(
        decision=data.get("decision") or "allow",
        reason=data.get("reason"),
        metadata=data.get("metadata") or {},
        logs=list(data.get("logs") or ([] if not stderr else [stderr])),
        hook=hook,
    )
