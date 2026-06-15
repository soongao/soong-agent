from __future__ import annotations

import asyncio

import pytest

from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from agent_core.types.runtime import RunStatus
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider


@pytest.mark.asyncio
async def test_queued_run_cancel_does_not_hit_provider_or_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    gate = asyncio.Event()
    fake = FakeProvider(final_text="done", block_event=gate)
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)
    permission_calls = []

    async def permission_callback(request):
        permission_calls.append(request)
        raise AssertionError("queued cancel must not request permissions")

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=permission_callback) as runtime:
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
    assert len(fake.requests) == 1
    assert permission_calls == []
