from __future__ import annotations

from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.events import make_event
from agent_core.mcp.tools import register_mcp_tools


async def ensure_mcp_tools(
    runtime: Any,
    *,
    handle: RunHandle | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> None:
    assert runtime.store
    if runtime._mcp_discovered or runtime._mcp_manager is None:
        return
    runtime._mcp_discovered = True
    result = await runtime._mcp_manager.discover()
    register_mcp_tools(runtime.tool_registry, result.tools)
    for failure in result.failures:
        payload = {"server_id": failure.get("server_id"), "message": failure.get("message")}
        if handle is not None:
            await runtime._emit(handle, "mcp_server_failed", level="warning", payload=payload)
        elif session_id is not None:
            await runtime.store.add_event(
                make_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    event_type="mcp_server_failed",
                    level="warning",
                    payload=payload,
                )
            )
