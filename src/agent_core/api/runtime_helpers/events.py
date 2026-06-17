from __future__ import annotations

from typing import Any

from agent_core.api.handles import RunHandle
from agent_core.errors.codes import ErrorCode
from agent_core.events import EventStream, make_event
from agent_core.types import ErrorPayload, RuntimeEvent


async def emit(
    runtime: Any,
    handle: RunHandle,
    event_type: str,
    *,
    level: str = "info",
    node_id: str | None = None,
    tool_call_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RuntimeEvent:
    assert runtime.store
    event = make_event(
        session_id=handle.session_id,
        agent_id=handle.agent_id,
        run_id=handle.run_id,
        event_type=event_type,
        level=level,
        node_id=node_id,
        tool_call_id=tool_call_id,
        payload=payload,
    )
    stored = await runtime.store.add_event(event)
    await handle._stream.put(stored)
    return stored


def emit_realtime(
    handle: RunHandle,
    event_type: str,
    *,
    level: str = "info",
    node_id: str | None = None,
    tool_call_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RuntimeEvent | None:
    event = make_event(
        session_id=handle.session_id,
        agent_id=handle.agent_id,
        run_id=handle.run_id,
        event_type=event_type,
        level=level,
        node_id=node_id,
        tool_call_id=tool_call_id,
        payload=payload,
    )
    return event if handle._stream.put_nowait(event) else None


def open_child_run_stream(runtime: Any, run_id: str) -> EventStream:
    stream = EventStream()
    runtime._child_run_streams[run_id] = stream
    return stream


async def close_child_run_stream(runtime: Any, run_id: str) -> None:
    stream = runtime._child_run_streams.pop(run_id, None)
    if stream is not None:
        await stream.close()


async def emit_child_run_event(
    runtime: Any,
    *,
    stream: EventStream | None,
    mirror_handle: RunHandle | None = None,
    session_id: str,
    agent_id: str,
    run_id: str,
    event_type: str,
    level: str = "info",
    node_id: str | None = None,
    tool_call_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RuntimeEvent:
    assert runtime.store
    stored = await runtime.store.add_event(
        make_event(
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type=event_type,
            level=level,
            node_id=node_id,
            tool_call_id=tool_call_id,
            payload=payload,
        )
    )
    if stream is not None and stream.has_consumer:
        await stream.put(stored)
    if mirror_handle is not None:
        await mirror_handle._stream.put(stored)
    return stored


async def emit_child_model_event(
    runtime: Any,
    *,
    stream: EventStream,
    session_id: str,
    agent_id: str,
    run_id: str,
    event: Any,
) -> None:
    if event.event_type == "model_started":
        await emit_child_run_event(
            runtime,
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="model_started",
            payload=event.metadata,
        )
    elif event.event_type == "model_text_delta":
        if not stream.has_consumer:
            return
        transient = make_event(
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="model_text_delta",
            payload={"text": event.text_delta or ""},
        )
        await stream.put(transient)
    elif event.event_type == "model_completed":
        await emit_child_run_event(
            runtime,
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="model_completed",
            payload={
                "stop_reason": event.stop_reason.value if event.stop_reason else None,
                "tool_calls": [call.model_dump(mode="json") for call in event.tool_calls],
                "usage": event.usage.model_dump(mode="json") if event.usage else None,
            },
        )
    elif event.event_type == "model_failed":
        error = event.error or ErrorPayload(code=ErrorCode.PROVIDER_ERROR, message="provider failed")
        await emit_child_run_event(
            runtime,
            stream=stream,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            event_type="model_failed",
            level="error",
            payload=error.model_dump(mode="json"),
        )


async def child_events(runtime: Any, child_run_id: str, debug: bool = False):
    stream = runtime._child_run_streams.get(child_run_id)
    if stream is not None and not stream.has_consumer:
        async for event in stream.iter():
            if debug or event.level != "debug":
                yield event
        return
    replay = await runtime.replay_run(child_run_id)
    for event in replay.events:
        if debug or event.level != "debug":
            yield event
