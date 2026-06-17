from __future__ import annotations

from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.events import make_event


async def run_stop_hooks(runtime: Any, handle: RunHandle, *, end_node_id: str | None):
    assert runtime.paths and runtime.config
    from agent_core.hooks.runner import HookRunner

    return await HookRunner(runtime._hooks or []).run(
        event_type="stop",
        payload={
            "event_type": "Stop",
            "session_id": handle.session_id,
            "run_id": handle.run_id,
            "agent_id": handle.agent_id,
            "end_node_id": end_node_id,
        },
        cwd=runtime.paths.project_dir,
        timeout_ms=runtime.config.hooks.default_timeout_ms,
        env_allowlist=runtime.config.tools.env_allowlist,
    )


async def run_observe_hook(
    runtime: Any,
    *,
    event_type: str,
    session_id: str,
    agent_id: str | None,
    run_id: str | None,
    payload: dict[str, Any],
) -> None:
    assert runtime.paths and runtime.config and runtime.store
    if not runtime._hooks:
        return
    from agent_core.hooks.runner import HookRunner

    decision = await HookRunner(runtime._hooks).run(
        event_type=event_type,
        payload=payload,
        cwd=runtime.paths.project_dir,
        timeout_ms=runtime.config.hooks.default_timeout_ms,
        env_allowlist=runtime.config.tools.env_allowlist,
    )
    if decision.hook or decision.error or decision.logs or decision.denied:
        await runtime.store.add_event(
            make_event(
                session_id=session_id,
                agent_id=agent_id,
                run_id=run_id,
                event_type=f"{event_type}_hook_observed",
                level="warning" if decision.error else "info",
                payload={
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "metadata": decision.metadata,
                    "logs": decision.logs,
                    "error": decision.error.model_dump(mode="json") if decision.error else None,
                },
            )
        )
