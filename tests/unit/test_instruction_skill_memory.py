from __future__ import annotations

import pytest

from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.context.composer import build_static_system_blocks, build_system_blocks
from agent_core.context.instructions import build_instruction_catalog, instruction_catalog_text
from agent_core.context.state import RuntimeContextState
from agent_core.permissions import PermissionSessionCache
from agent_core.tools.builtin_code import register_builtin_code_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.internal import register_internal_tools
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolCall
from tests.conftest import write_config


async def make_context(home, project, state=None) -> ToolExecutionContext:
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    return ToolExecutionContext(
        session_id="sess_ctx",
        run_id="run_ctx",
        agent_id="agent_main",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_cache=PermissionSessionCache(),
        services={"context_state": state or RuntimeContextState()},
    )


def test_instruction_catalog_frontmatter_only(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "CLAUDE.md").write_text("---\ntitle: Home\n---\nsecret body\n", encoding="utf-8")
    (project / "CLAUDE.md").write_text("project body\n", encoding="utf-8")
    entries, truncated = build_instruction_catalog(home_dir=home, project_dir=project)
    text = instruction_catalog_text(entries, truncated=truncated)
    assert "title: Home" in text
    assert "secret body" not in text
    assert "project body" not in text


def test_instruction_catalog_prefers_claude_over_agents_in_same_directory(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "CLAUDE.md").write_text("---\ntitle: Home Claude\n---\nclaude body\n", encoding="utf-8")
    (home / "AGENTS.md").write_text("---\ntitle: Home Agents\n---\nagents body\n", encoding="utf-8")
    nested = project / "pkg"
    nested.mkdir()
    (nested / "CLAUDE.md").write_text("---\ntitle: Project Claude\n---\n", encoding="utf-8")
    (nested / "AGENTS.md").write_text("---\ntitle: Project Agents\n---\n", encoding="utf-8")

    entries, truncated = build_instruction_catalog(home_dir=home, project_dir=project)
    text = instruction_catalog_text(entries, truncated=truncated)

    assert str((home / "CLAUDE.md").resolve()) in text
    assert str((home / "AGENTS.md").resolve()) not in text
    assert str((nested / "CLAUDE.md").resolve()) in text
    assert str((nested / "AGENTS.md").resolve()) not in text


def test_instruction_catalog_skips_project_rules_and_common_generated_dirs(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "rules").mkdir()
    (home / "rules" / "home-rule.md").write_text("---\ntitle: Home Rule\n---\nbody\n", encoding="utf-8")
    project_rules = project / ".soong-agent" / "rules"
    project_rules.mkdir(parents=True)
    (project_rules / "ignored.md").write_text("---\ntitle: Project Rule\n---\nbody\n", encoding="utf-8")
    node_modules = project / "node_modules" / "pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "CLAUDE.md").write_text("---\ntitle: Dependency\n---\nbody\n", encoding="utf-8")
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "AGENTS.md").write_text("---\ntitle: Git\n---\nbody\n", encoding="utf-8")

    entries, truncated = build_instruction_catalog(home_dir=home, project_dir=project)
    text = instruction_catalog_text(entries, truncated=truncated)

    assert str((home / "rules" / "home-rule.md").resolve()) in text
    assert "Project Rule" not in text
    assert "Dependency" not in text
    assert "Git" not in text


def test_static_system_blocks_load_package_assets(isolated_dirs) -> None:
    home, project = isolated_dirs
    blocks = build_static_system_blocks(home_dir=home, project_dir=project)
    by_id = {block.block_id: block for block in blocks}
    assert "system.core" in by_id
    assert "system.tool_protocol" in by_id
    assert by_id["system.core"].source == "package_asset"
    assert by_id["system.core"].content.strip()
    assert by_id["system.instruction_catalog"].source == "instruction_catalog"


def test_skill_catalog_frontmatter_only(isolated_dirs) -> None:
    home, project = isolated_dirs
    skills = home / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("---\nname: review\ndescription: Review code\n---\nsecret body\n", encoding="utf-8")
    blocks = build_static_system_blocks(home_dir=home, project_dir=project)
    catalog = next(block for block in blocks if block.source == "skill_catalog")
    assert "review" in catalog.content
    assert "Review code" in catalog.content
    assert "secret body" not in catalog.content


def test_memory_catalog_is_dynamic_system_block(isolated_dirs) -> None:
    home, project = isolated_dirs
    memory_dir = home / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# Memory Catalog\n\n- `mem_1` [user] likes pytest (user/a.md)\n", encoding="utf-8")
    blocks = build_system_blocks(home_dir=home, project_dir=project)
    catalog = [block for block in blocks if block.source == "memory_catalog"]
    assert len(catalog) == 1
    assert catalog[0].dynamic is True
    assert "likes pytest" in catalog[0].content


def test_skill_and_recalled_memory_body_are_not_system_blocks(isolated_dirs) -> None:
    home, project = isolated_dirs
    state = RuntimeContextState()
    state.skill_contexts.append({"name": "review", "path": str(home / "skills" / "review.md"), "hash": "h1", "body": "Skill body"})
    state.memory_contexts.append({"query": "pytest", "matches": [{"id": "mem_1", "content": "likes pytest"}]})
    blocks = build_system_blocks(home_dir=home, project_dir=project, context_state=state)
    assert not [block for block in blocks if block.source == "skill_context"]
    assert not [block for block in blocks if block.source == "memory_context"]


@pytest.mark.asyncio
async def test_read_instruction_marks_already_loaded(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = project / "CLAUDE.md"
    path.write_text("rules\n", encoding="utf-8")
    state = RuntimeContextState()
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project, state)
    first = await registry.execute(ToolCall(tool_call_id="c1", name="code.read_file", arguments={"path": str(path)}), context)
    second = await registry.execute(ToolCall(tool_call_id="c2", name="code.read_file", arguments={"path": str(path)}), context)
    assert first.content[0].data["already_loaded"] is False  # type: ignore[union-attr]
    assert second.content[0].data["already_loaded"] is True  # type: ignore[union-attr]
    blocks = build_system_blocks(home_dir=home, project_dir=project, context_state=state)
    instruction_blocks = [block for block in blocks if block.source == "instruction_context"]
    assert len(instruction_blocks) == 1
    assert instruction_blocks[0].content == "rules\n"


@pytest.mark.asyncio
async def test_read_file_only_marks_active_catalog_instructions(isolated_dirs) -> None:
    home, project = isolated_dirs
    claude = project / "pkg" / "CLAUDE.md"
    agents = project / "pkg" / "AGENTS.md"
    project_rules = project / ".soong-agent" / "rules" / "ignored.md"
    claude.parent.mkdir()
    project_rules.parent.mkdir(parents=True)
    claude.write_text("active rules\n", encoding="utf-8")
    agents.write_text("shadowed agents\n", encoding="utf-8")
    project_rules.write_text("project local rules\n", encoding="utf-8")
    state = RuntimeContextState()
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project, state)

    shadowed = await registry.execute(ToolCall(tool_call_id="c1", name="code.read_file", arguments={"path": str(agents)}), context)
    ignored = await registry.execute(ToolCall(tool_call_id="c2", name="code.read_file", arguments={"path": str(project_rules)}), context)
    active = await registry.execute(ToolCall(tool_call_id="c3", name="code.read_file", arguments={"path": str(claude)}), context)

    assert shadowed.content[0].data["already_loaded"] is None  # type: ignore[union-attr]
    assert ignored.content[0].data["already_loaded"] is None  # type: ignore[union-attr]
    assert active.content[0].data["already_loaded"] is False  # type: ignore[union-attr]
    blocks = build_system_blocks(home_dir=home, project_dir=project, context_state=state)
    instruction_blocks = [block for block in blocks if block.source == "instruction_context"]
    assert [block.content for block in instruction_blocks] == ["active rules\n"]


@pytest.mark.asyncio
async def test_read_instruction_reload_when_hash_changes(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = project / "CLAUDE.md"
    path.write_text("rules v1\n", encoding="utf-8")
    state = RuntimeContextState()
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project, state)

    first = await registry.execute(ToolCall(tool_call_id="c1", name="code.read_file", arguments={"path": str(path)}), context)
    path.write_text("rules v2\n", encoding="utf-8")
    second = await registry.execute(ToolCall(tool_call_id="c2", name="code.read_file", arguments={"path": str(path)}), context)

    assert first.content[0].data["already_loaded"] is False  # type: ignore[union-attr]
    assert second.content[0].data["already_loaded"] is False  # type: ignore[union-attr]
    blocks = build_system_blocks(home_dir=home, project_dir=project, context_state=state)
    instruction_blocks = [block for block in blocks if block.source == "instruction_context"]
    assert len(instruction_blocks) == 1
    assert instruction_blocks[0].content == "rules v2\n"


@pytest.mark.asyncio
async def test_load_skill_marks_already_loaded(isolated_dirs) -> None:
    home, project = isolated_dirs
    skills = home / "skills"
    skills.mkdir()
    (skills / "review.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    state = RuntimeContextState()
    registry = ToolRegistry()
    register_internal_tools(registry)
    context = await make_context(home, project, state)
    first = await registry.execute(ToolCall(tool_call_id="s1", name="internal.load_skill", arguments={"name": "review"}), context)
    second = await registry.execute(ToolCall(tool_call_id="s2", name="internal.load_skill", arguments={"name": "review"}), context)
    assert first.content[0].data["already_loaded"] is False  # type: ignore[union-attr]
    assert first.content[0].data["node_type"] == "skill_context"  # type: ignore[union-attr]
    assert first.content[0].data["content"].startswith('<skill name="review">')  # type: ignore[union-attr]
    assert second.content[0].data["already_loaded"] is True  # type: ignore[union-attr]
    assert len(state.skill_contexts) == 1


@pytest.mark.asyncio
async def test_load_skill_uses_frontmatter_name_not_filename(isolated_dirs) -> None:
    home, project = isolated_dirs
    skills = home / "skills"
    skills.mkdir()
    (skills / "code-review.md").write_text("---\nname: review\ndescription: Review code\n---\nSkill body\n", encoding="utf-8")
    registry = ToolRegistry()
    register_internal_tools(registry)
    context = await make_context(home, project, RuntimeContextState())
    result = await registry.execute(ToolCall(tool_call_id="s1", name="internal.load_skill", arguments={"name": "review"}), context)
    assert not result.is_error
    assert result.content[0].data["path"].endswith("code-review.md")  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_recall_memory_user_level_only(isolated_dirs) -> None:
    home, project = isolated_dirs
    memory = home / "memory" / "user"
    memory.mkdir(parents=True)
    (memory / "prefs.md").write_text("likes pytest", encoding="utf-8")
    registry = ToolRegistry()
    register_internal_tools(registry)
    context = await make_context(home, project)
    result = await registry.execute(
        ToolCall(tool_call_id="m1", name="internal.recall_memory", arguments={"query": "pytest"}),
        context,
    )
    matches = result.content[0].data["matches"]  # type: ignore[union-attr]
    assert len(matches) == 1
    assert matches[0]["path"].endswith("prefs.md")
    assert result.content[0].data["node_type"] == "memory_context"  # type: ignore[union-attr]
    assert "<memory" in result.content[0].data["content"]  # type: ignore[union-attr]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_role", ["sub", "fork", "worker", "compact"])
async def test_recall_memory_denied_for_non_main_agent_roles(isolated_dirs, agent_role: str) -> None:
    home, project = isolated_dirs
    memory = home / "memory" / "user"
    memory.mkdir(parents=True)
    (memory / "prefs.md").write_text("likes pytest", encoding="utf-8")
    registry = ToolRegistry()
    register_internal_tools(registry)
    context = await make_context(home, project)
    context.agent_role = agent_role

    result = await registry.execute(
        ToolCall(tool_call_id="m1", name="internal.recall_memory", arguments={"query": "pytest"}),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code.value == "tool_not_available"
