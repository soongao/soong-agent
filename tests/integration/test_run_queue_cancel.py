from __future__ import annotations

import asyncio

import pytest

from agent_core import AgentRuntime
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
    assert delete_after_complete.deleted is True
    assert "run_queued" in queued_events
    assert "run_cancelled" in queued_events
    assert len(scripted_ollama.requests) == 1
    assert permission_calls == []


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
