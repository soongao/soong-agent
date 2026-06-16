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
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_agent(path, *, agent_id: str, name: str = "Agent", body: str = "body", overrides: str | None = None) -> None:
    override_text = f"overrides: {overrides}\n" if overrides else ""
    path.write_text(
        "---\n"
        f"id: {agent_id}\n"
        f"name: {name}\n"
        "description: Test agent\n"
        f"{override_text}"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


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
    default_worker = next(
        item
        for item in result.content[0].data["agent_definitions"]  # type: ignore[union-attr]
        if item["agent_definition_id"] == "default_worker_agent"
    )
    assert set(default_worker) == {"agent_definition_id", "name", "description", "source", "suggested_tools", "tags"}
    assert "body" not in default_worker
    assert default_worker["source"] == "builtin"
    assert all({"name", "available"} <= set(tool) for tool in default_worker["suggested_tools"])


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


def test_user_agent_definition_loads_and_explicitly_overrides_builtin(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    _write_agent(agents / "reviewer.md", agent_id="reviewer", name="Reviewer", body="review body")
    _write_agent(
        agents / "sub.md",
        agent_id="default_sub_agent",
        name="Custom Sub",
        body="custom sub body",
        overrides="builtin:default_sub_agent",
    )

    definitions = AgentDefinitionRegistry()
    definitions.load_user_dir(agents)

    reviewer = definitions.get("reviewer")
    assert reviewer is not None
    assert reviewer.source == "user"
    assert reviewer.body == "review body\n"

    sub = definitions.get("default_sub_agent")
    assert sub is not None
    assert sub.source == "user"
    assert sub.name == "Custom Sub"
    assert sub.body == "custom sub body\n"
    assert sub.metadata["overrides"] == {"agent_definition_id": "default_sub_agent", "source": "builtin"}


def test_user_agent_definition_implicit_builtin_override_fails(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    _write_agent(agents / "sub.md", agent_id="default_sub_agent", name="Custom Sub", body="custom sub body")

    definitions = AgentDefinitionRegistry()
    with pytest.raises(AgentCoreError) as exc:
        definitions.load_user_dir(agents)
    assert exc.value.code.value == "duplicate_agent_definition"


def test_duplicate_user_agent_definition_fails(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    _write_agent(agents / "a.md", agent_id="dupe")
    _write_agent(agents / "b.md", agent_id="dupe")

    definitions = AgentDefinitionRegistry()
    with pytest.raises(AgentCoreError) as exc:
        definitions.load_user_dir(agents)
    assert exc.value.code.value == "duplicate_agent_definition"


def test_code_agent_definition_override_prevents_later_user_override(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    _write_agent(agents / "code.md", agent_id="code_agent")

    definitions = AgentDefinitionRegistry()
    definitions.register(
        definitions.get("default_sub_agent").model_copy(update={"agent_definition_id": "code_agent", "source": "code"}),  # type: ignore[union-attr]
        source="code",
    )
    with pytest.raises(AgentCoreError) as exc:
        definitions.load_user_dir(agents)
    assert exc.value.code.value == "duplicate_agent_definition"


def test_code_agent_definition_can_explicitly_override_user_definition(isolated_dirs) -> None:
    home, _project = isolated_dirs
    agents = home / "agents"
    agents.mkdir()
    _write_agent(agents / "reviewer.md", agent_id="reviewer", name="Reviewer", body="review body")

    definitions = AgentDefinitionRegistry()
    definitions.load_user_dir(agents)
    base = definitions.get("reviewer")
    assert base is not None
    definitions.register(
        base.model_copy(update={"name": "Code Reviewer", "source": "code", "overrides": "user:reviewer", "body": "code body"}),
        source="code",
    )

    reviewer = definitions.get("reviewer")
    assert reviewer is not None
    assert reviewer.source == "code"
    assert reviewer.body == "code body"
    assert reviewer.metadata["overrides"] == {"agent_definition_id": "reviewer", "source": "user"}


def test_override_target_must_exist_and_match(isolated_dirs) -> None:
    _home, _project = isolated_dirs
    definitions = AgentDefinitionRegistry()
    with pytest.raises(AgentCoreError) as exc:
        definitions.register(
            definitions.get("default_sub_agent").model_copy(  # type: ignore[union-attr]
                update={"agent_definition_id": "new_agent", "source": "code", "overrides": "builtin:missing"}
            ),
            source="code",
        )
    assert exc.value.code.value == "invalid_agent_override"


@pytest.mark.asyncio
async def test_user_agent_definition_invalid_suggested_tool_fails_startup(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
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
    runtime = AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with pytest.raises(AgentCoreError) as exc:
        await runtime._ensure_started()
    assert exc.value.code.value == "invalid_agent_definition"
