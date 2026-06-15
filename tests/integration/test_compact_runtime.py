from __future__ import annotations

import pytest

from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider


@pytest.mark.asyncio
async def test_runtime_compact_agent_writes_compaction_node(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = write_config(home)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 64\n\n",
        encoding="utf-8",
    )
    provider = FakeProvider(final_text="compact summary")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("remember this context", session_id="sess_compact")
        _events = [event async for event in handle.events()]
        result = await runtime.run_compact_agent(session_id="sess_compact", reason="test")
        replay = await runtime.replay_session("sess_compact")

    assert result["stale"] is False
    assert result["compaction_node_id"] is not None
    assert any(node.node_type == "compact_input" for node in replay.nodes)
    compaction_nodes = [node for node in replay.nodes if node.node_type == "compaction"]
    assert compaction_nodes
    assert compaction_nodes[-1].content[0].text == "compact summary"  # type: ignore[union-attr]
    compact_events = [event for event in replay.events if event.event_type == "compact_completed"]
    assert compact_events and compact_events[-1].payload["stale"] is False
    compact_requests = [request for request in provider.requests if request.metadata.get("purpose") == "compact"]
    assert compact_requests
    assert compact_requests[-1].model == "compact-model"
    assert compact_requests[-1].tools == []


@pytest.mark.asyncio
async def test_runtime_auto_background_compact_after_completed_run(isolated_dirs) -> None:
    home, project = isolated_dirs
    config_path = write_config(home)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("context_window = 8192", "context_window = 80")
    text = text.replace("max_output_tokens = 1024", "max_output_tokens = 16")
    text = text.replace(
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"',
        'session_db_path = "${SOONG_AGENT_HOME}/sessions.sqlite"\nnon_system_budget = 120',
    )
    text += "\n[compact]\nenabled = true\nreserve_tokens = 0\nkeep_recent_tokens = 5\nauto_background = true\nrecovery_sync = true\nmodel_profile = \"compact\"\nmax_summary_tokens = 64\n\n"
    text += "[model_overrides.compact]\nname = \"compact-model\"\nmax_output_tokens = 64\n\n"
    config_path.write_text(text, encoding="utf-8")
    provider = FakeProvider(final_text="background compact summary")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: provider)
    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("x" * 400, session_id="sess_auto_compact")
        _events = [event async for event in handle.events()]
        for _ in range(20):
            replay = await runtime.replay_session("sess_auto_compact")
            if any(node.node_type == "compaction" for node in replay.nodes):
                break
            import asyncio

            await asyncio.sleep(0.01)
        replay = await runtime.replay_session("sess_auto_compact")

    assert "compact_pending" in [event.event_type for event in replay.events]
    assert any(node.node_type == "compaction" for node in replay.nodes)
