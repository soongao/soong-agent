from __future__ import annotations

import pytest

from agent_core import AgentRuntime
from agent_core.errors import ConfigError
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


@pytest.mark.asyncio
async def test_orchestrator_without_worker_pool_fails(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama, worker_pool=False)
    async with _runtime(project, scripted_ollama) as runtime:
        with pytest.raises(ConfigError):
            await runtime.start("x", mode="orchestrator")


@pytest.mark.asyncio
async def test_orchestrator_with_worker_pool_starts(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama, worker_pool=True)
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("x", mode="orchestrator")
        events = [event.event_type async for event in handle.events()]
    assert "loop_completed" in events
