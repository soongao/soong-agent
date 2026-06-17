from __future__ import annotations

from typing import Any

from agent_core.hooks.runner import HookRunner
from agent_core.tools.execution import ToolExecutionContext
from agent_core.types.tools import ToolCall, ToolDefinition, ToolResult


async def run_post_tool_hooks(
    *,
    context: ToolExecutionContext,
    call: ToolCall,
    definition: ToolDefinition,
    target_path: Any,
    result: ToolResult,
) -> ToolResult:
    post_decision = await HookRunner(context.hooks or []).run(
        event_type="tool_completed",
        tool_name=definition.name,
        tags=set(definition.tags),
        target_path=str(target_path) if target_path is not None else None,
        payload={
            "event_type": "PostToolUse",
            "tool_name": definition.name,
            "arguments": call.arguments,
            "result": result.model_dump(mode="json"),
            "session_id": context.session_id,
            "run_id": context.run_id,
            "agent_id": context.agent_id,
        },
        cwd=context.effective_cwd,
        timeout_ms=context.config.hooks.default_timeout_ms,
        env_allowlist=context.config.tools.env_allowlist,
    )
    if post_decision.hook or post_decision.error:
        metadata = dict(result.metadata)
        metadata["post_hook_summary"] = {
            "decision": post_decision.decision,
            "reason": post_decision.reason,
            "metadata": post_decision.metadata,
            "error": post_decision.error.model_dump(mode="json") if post_decision.error else None,
        }
        return result.model_copy(update={"metadata": metadata})
    return result


def hook_summary(decision: Any) -> dict[str, Any] | None:
    if not decision.hook and not decision.error and not decision.metadata and not decision.logs and not decision.denied:
        return None
    return {
        "action": "deny" if decision.denied else "allow",
        "reason": decision.reason,
        "metadata": decision.metadata,
        "logs": decision.logs,
        "error": decision.error.model_dump(mode="json") if decision.error else None,
    }
