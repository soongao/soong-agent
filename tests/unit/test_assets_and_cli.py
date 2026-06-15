from __future__ import annotations

from importlib import resources

import pytest

from agent_core.cli import async_main, build_parser
from agent_core.config import load_config
from agent_core.permissions import stdin_permission_callback
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


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
    assert "--json" not in help_text

    run_parser = parser._subparsers._group_actions[0].choices["run"]  # type: ignore[attr-defined]
    run_help = run_parser.format_help()
    assert "--path" in run_help
    assert "--orchestrator" in run_help
    assert "--session-id" in run_help
    assert "--json" not in run_help


@pytest.mark.asyncio
async def test_cli_missing_config_exits_nonzero(isolated_dirs, capsys) -> None:
    _home, project = isolated_dirs
    code = await async_main(["run", "--path", str(project), "hi"])
    captured = capsys.readouterr()
    assert code == 1
    assert "config" in captured.err.lower()


@pytest.mark.asyncio
async def test_cli_run_uses_ollama_provider(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("cli ok")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    code = await async_main(["run", "--path", str(project), "hello from cli"])
    captured = capsys.readouterr()

    assert code == 0
    assert "cli ok" in captured.out
    assert captured.err == ""
    assert len(scripted_ollama.requests) == 1
    assert scripted_ollama.requests[0]["model"] == "gemma4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdin_text", "expected"),
    [
        ("1\n", "allow_once"),
        ("2\n", "allow_for_session"),
        ("3\n", "deny"),
        ("invalid\n", "deny"),
        ("", "deny"),
    ],
)
async def test_stdin_permission_choices(monkeypatch, stdin_text: str, expected: str) -> None:
    from io import StringIO

    from agent_core.types.permissions import PermissionRequest

    monkeypatch.setattr("sys.stdin", StringIO(stdin_text))
    request = PermissionRequest(
        request_id="perm_test",
        session_id="sess_test",
        agent_id="agent_main",
        run_id="run_test",
        parent_agent_id=None,
        parent_run_id=None,
        agent_role="main",
        tool_name="code.write_file",
        permission="write",
        tags=["code", "write"],
        args_summary="{}",
        target_scope="/tmp/a.txt",
        cwd="/tmp",
        env_summary={},
        network_host=None,
        dangerous=False,
        hook_summary=None,
        suggested_decision="deny",
    )

    decision = await stdin_permission_callback(request)

    assert decision.decision.value == expected
