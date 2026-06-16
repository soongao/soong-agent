from __future__ import annotations

from importlib import resources
from pathlib import Path
import tomllib

import pytest

from agent_core.assets.loader import get_asset, list_required_assets, read_asset
from agent_core.cli import async_main, build_parser
from agent_core.config import load_config
from agent_core.permissions import stdin_permission_callback
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def test_required_assets_exist() -> None:
    for package, names in {
        "agent_core.assets.templates": [
            "config_default.toml",
            "plan_default.md",
            "task_dag_default.md",
            "plan_template.md",
            "task_template.md",
        ],
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


def test_asset_loader_reads_contract_asset_ids() -> None:
    asset_ids = {asset.asset_id for asset in list_required_assets()}

    assert {
        "system.core",
        "template.config.default",
        "template.plan.default",
        "template.task_dag.default",
        "agent.default_worker_agent",
        "agent.default_compact_agent",
    } <= asset_ids

    for asset_id in asset_ids:
        text = read_asset(asset_id)
        assert text.strip()
        assert get_asset(asset_id).resource_path


def test_wheel_build_config_includes_typed_marker_and_assets() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert wheel["packages"] == ["src/agent_core"]
    assert "src/agent_core/py.typed" in wheel["include"]
    assert "src/agent_core/assets/**/*.md" in wheel["include"]
    assert "src/agent_core/assets/**/*.toml" in wheel["include"]


def test_default_config_template_matches_contract(tmp_path) -> None:
    text = read_asset("template.config.default")
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
async def test_cli_path_file_uses_parent_dir(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    source = project / "src" / "a.py"
    source.parent.mkdir()
    source.write_text("print('x')\n", encoding="utf-8")
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("file path ok")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    code = await async_main(["run", "--path", str(source), "hello from file path"])
    captured = capsys.readouterr()

    assert code == 0
    assert "file path ok" in captured.out
    assert captured.err == ""
    db_path = home / "sessions.sqlite"
    assert db_path.exists()
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT cwd FROM sessions").fetchone()
    assert row is not None
    assert row[0] == str(source.parent.resolve())


@pytest.mark.asyncio
async def test_cli_permission_allow_once_writes_file(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="write", name="code.write_file", arguments={"path": "allowed.txt", "content": "ok"})]
    )
    scripted_ollama.enqueue_text("write done")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("1\n"))
    code = await async_main(["run", "--path", str(project), "write a file"])
    captured = capsys.readouterr()

    assert code == 0
    assert "allow once" in captured.out
    assert "write done" in captured.out
    assert captured.err == ""
    assert (project / "allowed.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_cli_permission_allow_for_session_reuses_scope(
    isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="write_first", name="code.write_file", arguments={"path": "session.txt", "content": "one"})]
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="write_second",
                name="code.write_file",
                arguments={"path": "session.txt", "content": "two", "overwrite": True},
            )
        ]
    )
    scripted_ollama.enqueue_text("session write done")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("2\n"))
    code = await async_main(["run", "--path", str(project), "write the same file twice"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.count("Permission required for code.write_file") == 1
    assert "session write done" in captured.out
    assert captured.err == ""
    assert (project / "session.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_cli_permission_deny_blocks_write(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="write", name="code.write_file", arguments={"path": "denied.txt", "content": "no"})]
    )
    scripted_ollama.enqueue_text("write denied")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("3\n"))
    code = await async_main(["run", "--path", str(project), "try writing a file"])
    captured = capsys.readouterr()

    assert code == 0
    assert "deny" in captured.out
    assert "write denied" in captured.out
    assert captured.err == ""
    assert not (project / "denied.txt").exists()


@pytest.mark.asyncio
async def test_cli_permission_deny_stops_following_write_tool_calls(
    isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(tool_call_id="denied", name="code.write_file", arguments={"path": "denied.txt", "content": "no"}),
            ToolCall(tool_call_id="skipped", name="code.write_file", arguments={"path": "skipped.txt", "content": "skip"}),
        ]
    )
    scripted_ollama.enqueue_text("writes stopped")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("3\n"))
    code = await async_main(["run", "--path", str(project), "try writing two files"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.count("Permission required for code.write_file") == 1
    assert "deny" in captured.out
    assert "writes stopped" in captured.out
    assert captured.err == ""
    assert not (project / "denied.txt").exists()
    assert not (project / "skipped.txt").exists()


@pytest.mark.asyncio
async def test_cli_orchestrator_dispatches_worker_with_ollama(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url, worker_pool=True)
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="create",
                name="agent.task_create",
                arguments={
                    "task_id": "cli_task",
                    "wal_name": "cli_task.wal.jsonl",
                    "title": "CLI task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step 1"}],
                },
            )
        ]
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="dispatch",
                name="agent.dispatch_worker",
                arguments={"task_id": "cli_task", "instruction": "finish it", "allowed_step_ids": ["s1"]},
            )
        ]
    )
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="claim", name="agent.task_claim_step", arguments={"task_id": "cli_task", "step_id": "s1"})]
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="complete",
                name="agent.task_update_step",
                arguments={"task_id": "cli_task", "step_id": "s1", "status": "completed", "result_summary": "done"},
            )
        ]
    )
    scripted_ollama.enqueue_text("worker done")
    scripted_ollama.enqueue_text("orchestrator done")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())
    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("1\n1\n1\n"))

    code = await async_main(["run", "--path", str(project), "--orchestrator", "orchestrate cli task"])
    captured = capsys.readouterr()

    assert code == 0
    assert "orchestrator done" in captured.out
    assert captured.err == ""
    assert len(scripted_ollama.requests) == 6
    wal_files = list((project / ".soong-agent" / "tasks").glob("*/cli_task.wal.jsonl"))
    assert len(wal_files) == 1
    wal_text = wal_files[0].read_text(encoding="utf-8")
    assert "task_step_completed" in wal_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdin_text", "expected"),
    [
        ("1\n", "allow_once"),
        ("allow once\n", "allow_once"),
        ("allow_once\n", "allow_once"),
        ("2\n", "allow_for_session"),
        ("allow for session\n", "allow_for_session"),
        ("allow_for_session\n", "allow_for_session"),
        ("3\n", "deny"),
        ("deny\n", "deny"),
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
