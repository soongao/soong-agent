from __future__ import annotations

import pytest

from agent_core import AgentRuntime
from agent_core.errors import ConfigError
from agent_core.providers import ProviderRegistry
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider


@pytest.mark.asyncio
async def test_orchestrator_without_worker_pool_fails(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=False)
    registry = ProviderRegistry()
    registry.register("fake", lambda config: FakeProvider())
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        with pytest.raises(ConfigError):
            await runtime.start("x", mode="orchestrator")


@pytest.mark.asyncio
async def test_orchestrator_with_worker_pool_starts(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home, worker_pool=True)
    registry = ProviderRegistry()
    registry.register("fake", lambda config: FakeProvider())
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("x", mode="orchestrator")
        events = [event.event_type async for event in handle.events()]
    assert "loop_completed" in events

