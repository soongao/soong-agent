from __future__ import annotations

from importlib import resources

import pytest

from agent_core.cli import async_main, build_parser
from agent_core.config import load_config
from tests.conftest import write_config


def test_required_assets_exist() -> None:
    for package, names in {
        "agent_core.assets.templates": ["config_default.toml", "plan_template.md", "task_template.md"],
        "agent_core.assets.prompts.system": [
            "core.md",
            "tool_protocol.md",
            "todo.md",
            "permissions.md",
            "multi_agent.md",
            "memory.md",
            "compact.md",
        ],
        "agent_core.assets.agents": [
            "default_sub_agent.md",
            "default_fork_agent.md",
            "default_worker_agent.md",
            "default_compact_agent.md",
        ],
    }.items():
        files = resources.files(package)
        for name in names:
            text = files.joinpath(name).read_text(encoding="utf-8")
            assert text.strip()


def test_default_config_template_matches_contract(tmp_path) -> None:
    text = resources.files("agent_core.assets.templates").joinpath("config_default.toml").read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    config = load_config(path)

    assert config.model.provider == "ollama"
    assert config.model.name == "gemma4"
    assert config.model_overrides["compact"]["max_output_tokens"] == 2048
    assert config.permissions.network_policy.default == "confirm"
    assert config.agents.worker_pools[0].pool_id == "default"
    assert config.agents.worker_pools[0].workers[0].worker_id == "worker_general_1"
    assert config.agents.worker_pools[0].workers[0].agent_definition_id == "default_worker_agent"


def test_cli_help_has_run_but_no_init(capsys) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    assert "init" not in help_text


@pytest.mark.asyncio
async def test_cli_missing_config_exits_nonzero(isolated_dirs, capsys) -> None:
    _home, project = isolated_dirs
    code = await async_main(["run", "--path", str(project), "hi"])
    captured = capsys.readouterr()
    assert code == 1
    assert "config" in captured.err.lower()


@pytest.mark.asyncio
async def test_cli_path_file_uses_parent_dir(isolated_dirs, monkeypatch, capsys) -> None:
    home, project = isolated_dirs
    write_config(home)
    source = project / "src" / "a.py"
    source.parent.mkdir()
    source.write_text("print('x')\n", encoding="utf-8")

    from agent_core import cli
    from tests.fixtures.fake_provider import FakeProvider
    from agent_core.providers import ProviderRegistry

    fake = FakeProvider(final_text="ok")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)

    original_runtime = cli.AgentRuntime

    def runtime_factory(*args, **kwargs):
        kwargs["provider_registry"] = registry
        return original_runtime(*args, **kwargs)

    monkeypatch.setattr(cli, "AgentRuntime", runtime_factory)
    code = await async_main(["run", "--path", str(source), "hi"])
    captured = capsys.readouterr()
    assert code == 0
    assert "ok" in captured.out
    assert fake.requests[0].metadata["session_id"]
