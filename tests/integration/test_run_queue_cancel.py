from __future__ import annotations

import asyncio
import json

import pytest

from agent_core import AgentRuntime
from agent_core.errors.codes import ErrorCode
from agent_core.types.runtime import RunStatus
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


@pytest.mark.asyncio
async def test_queued_run_cancel_does_not_hit_provider_or_permission(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    gate = asyncio.Event()
    scripted_ollama.enqueue_text("done", block=lambda: gate.wait())
    permission_calls = []

    async def permission_callback(request):
        permission_calls.append(request)
        raise AssertionError("queued cancel must not request permissions")

    async with _runtime(project, scripted_ollama, permission_callback=permission_callback) as runtime:
        first = await runtime.start("first", session_id="sess_queue")
        first_events = []

        async def consume_first():
            async for event in first.events():
                first_events.append(event.event_type)

        first_consumer = asyncio.create_task(consume_first())
        await asyncio.sleep(0)
        second = await runtime.start("second", session_id="sess_queue")
        queued_events = []

        async def consume_second():
            async for event in second.events():
                queued_events.append(event.event_type)

        second_consumer = asyncio.create_task(consume_second())
        await asyncio.sleep(0)
        assert second.status == RunStatus.QUEUED
        cancel = await second.cancel()
        delete_while_first_active = await runtime.delete_session("sess_queue")
        gate.set()
        await first_consumer
        await second_consumer
        delete_after_complete = await runtime.delete_session("sess_queue")
    assert cancel.cancelled is True
    assert delete_while_first_active.deleted is False
    assert delete_while_first_active.error is not None
    assert delete_while_first_active.error.code == ErrorCode.SESSION_ACTIVE
    assert delete_after_complete.deleted is True
    assert "run_queued" in queued_events
    assert "run_dequeued" in queued_events
    assert "run_cancelled" in queued_events
    assert queued_events.index("run_dequeued") < queued_events.index("run_cancelled")
    assert len(scripted_ollama.requests) == 1
    assert permission_calls == []


@pytest.mark.asyncio
async def test_run_queue_drains_in_fifo_order(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    scripted_ollama.enqueue_text("first done", block=lambda: first_gate.wait())
    scripted_ollama.enqueue_text("second done", block=lambda: second_gate.wait())
    scripted_ollama.enqueue_text("third done")

    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first", session_id="sess_fifo")
        first_events: list[str] = []
        second_events: list[str] = []
        third_events: list[str] = []

        async def collect(handle, target: list[str]) -> None:
            async for event in handle.events():
                target.append(event.event_type)

        first_consumer = asyncio.create_task(collect(first, first_events))
        while len(scripted_ollama.requests) < 1:
            await asyncio.sleep(0)

        second = await runtime.start("second", session_id="sess_fifo")
        third = await runtime.start("third", session_id="sess_fifo")
        second_consumer = asyncio.create_task(collect(second, second_events))
        third_consumer = asyncio.create_task(collect(third, third_events))
        await asyncio.sleep(0)

        assert second.status == RunStatus.QUEUED
        assert third.status == RunStatus.QUEUED

        first_gate.set()
        await first_consumer
        while len(scripted_ollama.requests) < 2:
            await asyncio.sleep(0)

        assert second.status == RunStatus.RUNNING
        assert third.status == RunStatus.QUEUED
        assert "second" in json.dumps(scripted_ollama.requests[1])

        second_gate.set()
        await second_consumer
        while len(scripted_ollama.requests) < 3:
            await asyncio.sleep(0)
        await third_consumer

    request_texts = [json.dumps(request) for request in scripted_ollama.requests]
    assert ["first" in request_texts[0], "second" in request_texts[1], "third" in request_texts[2]] == [True, True, True]
    assert "run_queued" in second_events
    assert second_events.index("run_dequeued") < second_events.index("loop_started")
    assert "run_queued" in third_events
    assert third_events.index("run_dequeued") < third_events.index("loop_started")


@pytest.mark.asyncio
async def test_active_run_cancel_updates_status_and_frees_session(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    gate = asyncio.Event()
    scripted_ollama.enqueue_text("done", block=lambda: gate.wait())
    scripted_ollama.enqueue_text("done")

    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first", session_id="sess_active_cancel")
        events = []

        async def consume_first():
            async for event in first.events():
                events.append(event.event_type)

        consumer = asyncio.create_task(consume_first())
        while not scripted_ollama.requests:
            await asyncio.sleep(0)

        cancel = await first.cancel()
        await consumer
        replay = await runtime.replay_session("sess_active_cancel")
        cancelled_replay = await runtime.replay_run(first.run_id)
        second = await runtime.start("second", session_id="sess_active_cancel")
        second_events = [event.event_type async for event in second.events()]

    assert cancel.cancelled is True
    assert first.status == RunStatus.CANCELLED
    assert "run_cancelled" in events
    assert "loop_completed" in second_events
    assert any(event.event_type == "run_cancelled" for event in cancelled_replay.events)
    assert any(event.event_type == "run_cancelled" for event in replay.events)


@pytest.mark.asyncio
async def test_active_run_cancel_after_streaming_delta_persists_aborted_partial(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    gate = asyncio.Event()
    scripted_ollama.enqueue_streaming_text("partial text", block_after_delta=lambda: gate.wait())

    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("first", session_id="sess_stream_cancel")
        events = []
        delta_seen = asyncio.Event()

        async def consume_events():
            async for event in handle.events(debug=True):
                events.append(event)
                if event.event_type == "model_text_delta":
                    delta_seen.set()

        consumer = asyncio.create_task(consume_events())
        await asyncio.wait_for(delta_seen.wait(), timeout=2)
        cancel = await handle.cancel()
        await consumer
        replay = await runtime.replay_run(handle.run_id)
        active_node_id = await runtime.store.active_node_id("sess_stream_cancel")  # type: ignore[union-attr]

    assert cancel.cancelled is True
    assert handle.status == RunStatus.CANCELLED
    assert any(event.event_type == "model_text_delta" for event in events)
    assert any(event.event_type == "aborted_streaming" for event in replay.events)
    assert any(event.event_type == "run_cancelled" for event in replay.events)
    partial = [node for node in replay.nodes if node.role == "assistant" and node.metadata.get("partial")]
    assert partial
    assert partial[-1].metadata["aborted"] is True
    assert partial[-1].metadata["abort_reason"] == "cancelled"
    assert any(getattr(block, "text", "") == "partial text" for block in partial[-1].content)
    active_nodes = [node for node in replay.nodes if node.node_id == active_node_id]
    assert active_nodes and active_nodes[0].role == "user"
