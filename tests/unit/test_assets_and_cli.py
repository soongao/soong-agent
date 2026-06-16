from __future__ import annotations

import asyncio
from io import StringIO
from importlib import resources
from pathlib import Path
import sqlite3
import sys
import tomllib

import pytest

from agent_core.assets.loader import get_asset, list_required_assets, read_asset
from agent_cli.cli import async_main, build_parser
from agent_cli.render import event_to_lines
from agent_core.config import load_config
from agent_core.permissions import stdin_permission_callback
from agent_core.events import make_event
from agent_core.types.tools import ToolCall
from agent_cli.config_bootstrap import ensure_default_config
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _chat_stdin(text: str) -> StringIO:
    return StringIO(text)


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


def test_prompt_assets_replace_placeholder_contract_text() -> None:
    prompt_expectations = {
        "system.core": ["soong-agent", "Build context from the repository", "Preserve unrelated user"],
        "system.tool_protocol": ["code.read_file", "code.edit_file", "internal.load_skill", "agent.dispatch_worker"],
        "system.permissions": ["allow_once", "allow_for_session", "Hook errors and timeouts"],
        "system.multi_agent": ["agent.create_sub_agent", "agent.fork_agent", "Task DAG", "no_step_claimed"],
        "system.compact": ["compaction node", "first_kept_node_id", "Task/worker status"],
        "system.todo": ["not a tool", "Task DAG", "private scratchpad"],
        "system.memory": ["internal.recall_memory", "memory_context", "Project memory is not supported"],
    }
    old_placeholders = [
        "Follow the Agent Core runtime contract",
        "Use only tools exposed in the current effective tool set.",
        "Todo is internal scratchpad state and is not persisted as a Task DAG.",
        "Write, dangerous, network, and sensitive read operations require permission.",
        "Sub agents and workers must stay within their effective tool set and assigned scope.",
        "Memory is recalled progressively and only through approved internal mechanisms.",
        "Compaction summarizes context without changing source conversation nodes.",
    ]

    for asset_id, expected_fragments in prompt_expectations.items():
        text = read_asset(asset_id)
        for fragment in expected_fragments:
            assert fragment in text
        for placeholder in old_placeholders:
            assert placeholder not in text


def test_agent_and_template_assets_are_expanded_for_runtime_semantics() -> None:
    assert "bounded sub agent" in read_asset("agent.default_sub_agent")
    assert "code.search" in read_asset("agent.default_fork_agent")
    assert "agent.task_claim_step" in read_asset("agent.default_worker_agent")
    assert "internal-only compaction agent" in read_asset("agent.default_compact_agent")
    assert "decision-complete Markdown plan" in read_asset("template.plan.default")
    assert "depends_on_step_ids" in read_asset("template.task_dag.default")


def test_wheel_build_config_includes_typed_marker_and_assets() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert wheel["packages"] == ["src/agent_core", "src/agent_cli"]
    assert "src/agent_core/py.typed" in wheel["include"]
    assert "src/agent_cli/py.typed" in wheel["include"]
    assert "src/agent_core/assets/**/*.md" in wheel["include"]
    assert "src/agent_core/assets/**/*.toml" in wheel["include"]


def test_default_config_template_matches_contract(tmp_path) -> None:
    text = read_asset("template.config.default")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    config = load_config(path)

    assert config.runtime.max_turns == 128
    assert config.model.provider == "ollama"
    assert config.model.name == "gemma4"
    assert config.model_overrides["compact"]["max_output_tokens"] == 2048
    assert config.permissions.network_policy.default == "confirm"
    assert config.agents.worker_pools[0].pool_id == "default"
    assert config.agents.worker_pools[0].workers[0].worker_id == "worker_general_1"
    assert config.agents.worker_pools[0].workers[0].agent_definition_id == "default_worker_agent"


def test_cli_help_has_chat_but_no_run_or_init(capsys) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "chat" in help_text
    assert "run" not in help_text
    assert "init" not in help_text
    assert "--json" not in help_text

    chat_parser = parser._subparsers._group_actions[0].choices["chat"]  # type: ignore[attr-defined]
    chat_help = chat_parser.format_help()
    assert "--path" in chat_help
    assert "--orchestrator" in chat_help
    assert "--session-id" in chat_help
    assert "--plain" in chat_help
    assert "--debug-events" in chat_help
    assert "--json" not in chat_help


def test_cli_render_maps_core_events() -> None:
    tool_lines = event_to_lines(
        make_event(session_id="sess", event_type="tool_started", payload={"name": "code.read_file"})
    )
    failed_lines = event_to_lines(
        make_event(session_id="sess", event_type="loop_failed", payload={"message": "bad"})
    )
    memory_lines = event_to_lines(
        make_event(session_id="sess", event_type="memory_extraction_completed", payload={"reason": "idle"})
    )

    assert tool_lines[0].text == "[tool] code.read_file started"
    assert failed_lines[0].text == "[error] bad"
    assert memory_lines[0].text == "[memory] completed (idle)"


@pytest.mark.asyncio
async def test_tui_prompt_newline_history_and_markdown(isolated_dirs) -> None:
    pytest.importorskip("textual")
    from agent_cli.tui import PromptTextArea, SoongAgentTui

    _home, project = isolated_dirs
    args = type(
        "Args",
        (),
        {"session_id": "sess_tui", "orchestrator": False, "path": str(project), "debug_events": False},
    )()
    app = SoongAgentTui(args)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptTextArea)

        prompt.load_text("hello")
        prompt.move_cursor((0, 5))
        await pilot.press("ctrl+j")
        await pilot.pause()
        assert prompt.text == "hello\n"

        app._record_history("first")
        app._record_history("second")
        prompt.load_text("")
        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "second"
        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "first"
        await pilot.press("down")
        await pilot.pause()
        assert prompt.text == "second"

        await app._append_assistant_delta("**done**")
        await app._finalize_assistant_stream()
        await pilot.pause()
        assert list(app.query(".assistant-markdown"))


@pytest.mark.asyncio
async def test_tui_permission_prompt_is_inline_and_waits(isolated_dirs) -> None:
    pytest.importorskip("textual")
    from agent_cli.tui import InlinePermissionPrompt, SoongAgentTui
    from agent_core.types.permissions import PermissionRequest

    _home, project = isolated_dirs
    args = type(
        "Args",
        (),
        {"session_id": "sess_tui_perm", "orchestrator": False, "path": str(project), "debug_events": False},
    )()
    request = PermissionRequest(
        request_id="perm_test",
        session_id="sess_tui_perm",
        agent_id="agent_main",
        run_id="run_main",
        agent_role="main",
        tool_name="code.write_file",
        permission="write",
        tags=["write"],
        args_summary="{'path': 'x.txt'}",
        target_scope=str(project / "x.txt"),
        cwd=str(project),
    )
    app = SoongAgentTui(args)
    async with app.run_test() as pilot:
        task = asyncio.create_task(app.permission_callback(request))
        await pilot.pause()
        assert not task.done()
        prompts = list(app.query(InlinePermissionPrompt))
        assert len(prompts) == 1
        await pilot.click("#allow_once")
        decision = await asyncio.wait_for(task, timeout=1)
        assert decision.decision.value == "allow_once"
        assert not list(app.query(InlinePermissionPrompt))


def test_cli_bootstrap_creates_default_config(isolated_dirs) -> None:
    home, _project = isolated_dirs

    created = ensure_default_config()

    assert created == home / "config.toml"
    assert created.exists()
    config = load_config(created)
    assert config.model.provider == "ollama"
    assert ensure_default_config() is None


def test_cli_bootstrap_does_not_overwrite_existing_config(isolated_dirs) -> None:
    home, _project = isolated_dirs
    config_path = home / "config.toml"
    config_path.write_text("[model]\nprovider = \"custom\"\nname = \"custom-model\"\n", encoding="utf-8")

    created = ensure_default_config()

    assert created is None
    assert config_path.read_text(encoding="utf-8") == "[model]\nprovider = \"custom\"\nname = \"custom-model\"\n"


@pytest.mark.asyncio
async def test_cli_run_subcommand_is_not_available(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        await async_main(["run", "hello"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "invalid choice" in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize("stdin_text", ["/exit\n", "/quit\n", ""])
async def test_cli_chat_exit_commands_return_zero(isolated_dirs, monkeypatch, capsys, stdin_text: str) -> None:
    _home, project = isolated_dirs
    monkeypatch.setattr("sys.stdin", _chat_stdin(stdin_text))
    code = await async_main(["chat", "--path", str(project), "--plain"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""


@pytest.mark.asyncio
async def test_cli_chat_uses_ollama_provider(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("cli ok")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    monkeypatch.setattr("sys.stdin", _chat_stdin("hello from cli\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
    captured = capsys.readouterr()

    assert code == 0
    assert "cli ok" in captured.out
    assert captured.err == ""
    assert len(scripted_ollama.requests) == 1
    assert scripted_ollama.requests[0]["model"] == "gemma4"


@pytest.mark.asyncio
async def test_cli_chat_two_inputs_share_one_session(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("first answer")
    scripted_ollama.enqueue_text("second answer")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())
    monkeypatch.setattr("sys.stdin", _chat_stdin("first question\nsecond question\n/exit\n"))

    code = await async_main(["chat", "--path", str(project), "--plain"])
    captured = capsys.readouterr()

    assert code == 0
    assert "first answer" in captured.out
    assert "second answer" in captured.out
    assert captured.err == ""
    assert len(scripted_ollama.requests) == 2
    second_messages = scripted_ollama.requests[1]["messages"]
    assert any(message["role"] == "user" and "first question" in message["content"] for message in second_messages)
    assert any(message["role"] == "assistant" and "first answer" in message["content"] for message in second_messages)
    with sqlite3.connect(home / "sessions.sqlite") as conn:
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_cli_chat_fixed_session_id(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("fixed answer")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())
    monkeypatch.setattr("sys.stdin", _chat_stdin("hello fixed session\n/exit\n"))

    code = await async_main(["chat", "--path", str(project), "--session-id", "sess_fixed_cli", "--plain"])
    captured = capsys.readouterr()

    assert code == 0
    assert "fixed answer" in captured.out
    assert captured.err == ""
    with sqlite3.connect(home / "sessions.sqlite") as conn:
        row = conn.execute("SELECT session_id FROM sessions").fetchone()
    assert row == ("sess_fixed_cli",)


@pytest.mark.asyncio
async def test_cli_path_file_uses_parent_dir(isolated_dirs, monkeypatch, capsys, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    source = project / "src" / "a.py"
    source.parent.mkdir()
    source.write_text("print('x')\n", encoding="utf-8")
    write_config(home, base_url=scripted_ollama.base_url)
    scripted_ollama.enqueue_text("file path ok")
    monkeypatch.setattr("agent_core.api.runtime.default_provider_registry", lambda: scripted_ollama.provider_registry())

    monkeypatch.setattr("sys.stdin", _chat_stdin("hello from file path\n/exit\n"))
    code = await async_main(["chat", "--path", str(source), "--plain"])
    captured = capsys.readouterr()

    assert code == 0
    assert "file path ok" in captured.out
    assert captured.err == ""
    db_path = home / "sessions.sqlite"
    assert db_path.exists()
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

    monkeypatch.setattr("sys.stdin", _chat_stdin("write a file\n1\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
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

    monkeypatch.setattr("sys.stdin", _chat_stdin("write the same file twice\n2\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
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

    monkeypatch.setattr("sys.stdin", _chat_stdin("try writing a file\n3\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
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

    monkeypatch.setattr("sys.stdin", _chat_stdin("try writing two files\n3\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
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
    monkeypatch.setattr("sys.stdin", _chat_stdin("orchestrate cli task\n1\n1\n1\n/exit\n"))

    code = await async_main(["chat", "--path", str(project), "--orchestrator", "--plain"])
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
