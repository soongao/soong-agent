from __future__ import annotations

from typing import Any

from agent_core.api.runtime_helpers.agents.tools import worker_tool_context


def fail_unclosed_worker_step(
    runtime: Any,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    parent_agent_id: str,
    parent_run_id: str,
    worker_scope: dict[str, Any],
    task_id: str,
    reason: str,
) -> dict[str, Any] | None:
    context = worker_tool_context(
        runtime,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        parent_run_id=parent_run_id,
        worker_scope=worker_scope,
    )
    return runtime.task_service.fail_unclosed_worker_step(
        context,
        task_id=task_id,
        worker_run_id=run_id,
        reason=reason,
    )
