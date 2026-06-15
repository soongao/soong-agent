from __future__ import annotations

from agent_core.events.stream import EventStream
from agent_core.storage import new_id
from agent_core.types.common import utc_now
from agent_core.types.runtime import RuntimeEvent


def make_event(
    *,
    session_id: str,
    event_type: str,
    agent_id: str | None = None,
    run_id: str | None = None,
    level: str = "info",
    node_id: str | None = None,
    tool_call_id: str | None = None,
    payload: dict | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=new_id("evt"),
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        level=level,  # type: ignore[arg-type]
        event_type=event_type,
        node_id=node_id,
        tool_call_id=tool_call_id,
        payload=payload or {},
        created_at=utc_now(),
    )


__all__ = ["EventStream", "make_event"]

