from __future__ import annotations

import pytest

from agent_core.agents.registry import AgentDefinitionRegistry
from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.errors import AgentCoreError
from agent_core.permissions import PermissionSessionCache
from agent_core.tools.agent_tools import register_agent_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolCall
from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from tests.fixtures.fake_provider import FakeProvider
from tests.conftest import write_config


@pytest.mark.asyncio
async def test_list_builtin_agent_definitions(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    definitions = AgentDefinitionRegistry()
    registry = ToolRegistry()
    register_agent_tools(registry, definitions)
    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_cache=PermissionSessionCache(),
    )
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="agent.list_agent_definitions", arguments={}),
        context,
    )
    ids = {item["agent_definition_id"] for item in result.content[0].data["agent_definitions"]}  # type: ignore[union-attr]
    assert {"default_sub_agent", "default_fork_agent", "default_worker_agent"} <= ids
    assert "default_compact_agent" not in ids


def test_builtin_definitions_loaded_from_assets_and_compact_internal_hidden() -> None:
    definitions = AgentDefinitionRegistry()
    compact = definitions.get("default_compact_agent")
    assert compact is not None
    assert compact.metadata["asset_path"] == "agents/default_compact_agent.md"
    assert "internal-only" in compact.body
    assert "default_compact_agent" not in {definition.agent_definition_id for definition in definitions.list()}


def test_user_cannot_override_default_compact_agent(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    (agents / "compact.md").write_text(
        "---\n"
        "id: default_compact_agent\n"
        "name: Bad\n"
        "description: Bad override\n"
        "---\n"
        "bad\n",
        encoding="utf-8",
    )
    definitions = AgentDefinitionRegistry()
    with pytest.raises(AgentCoreError) as exc:
        definitions.load_user_dir(agents)
    assert exc.value.code.value == "invalid_agent_override"


@pytest.mark.asyncio
async def test_user_agent_definition_invalid_suggested_tool_fails_startup(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    agents = home / "agents"
    agents.mkdir()
    (agents / "bad.md").write_text(
        "---\n"
        "id: bad_agent\n"
        "name: Bad\n"
        "description: Bad suggested tool\n"
        "suggested_tools: [\"missing.tool\"]\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: FakeProvider(final_text="ok"))
    runtime = AgentRuntime(project_dir=project, provider_registry=registry)
    with pytest.raises(AgentCoreError) as exc:
        await runtime._ensure_started()
    assert exc.value.code.value == "invalid_agent_definition"
