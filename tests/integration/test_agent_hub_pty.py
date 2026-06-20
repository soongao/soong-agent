from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_core.agents.workers import WorkerRuntimeState
from agent_core.api.runtime_helpers.agents.worker_executor import WorkerExecutorContext
from agent_hub.backend.workers.executors.claude_code_pty.executor import _claude_code_command
from agent_hub.backend.workers.executors.codex_pty.executor import _codex_command, _prompt_text
from agent_hub.backend.workers.executors.codex_pty import CodexPtyWorkerExecutor
from agent_hub.backend.workers.pty import PtySessionKey, PtySessionManager


@pytest.mark.asyncio
async def test_pty_session_streams_output_and_accepts_active_input(tmp_path) -> None:
    script = (
        "import sys\n"
        "print('ready', flush=True)\n"
        "for line in sys.stdin:\n"
        "    text = line.strip()\n"
        "    print('echo:' + text, flush=True)\n"
        "    if text == 'finish':\n"
        "        print('<<<DONE>>>', flush=True)\n"
        "        break\n"
    )
    manager = PtySessionManager()
    outputs: list[str] = []
    try:
        session = await manager.get_or_create(
            PtySessionKey(core_session_id="sess_pty", worker_id="worker_pty", executor_type="test_pty"),
            command=["python3", "-u", "-c", script],
            cwd=tmp_path,
        )

        async def on_output(text: str) -> None:
            outputs.append(text)

        task = asyncio.create_task(
            session.run_turn(
                worker_run_id="run_worker_pty",
                prompt="hello",
                completion_marker="<<<DONE>>>",
                output_callback=on_output,
            )
        )
        for _ in range(50):
            if "echo:hello" in "".join(outputs):
                break
            await asyncio.sleep(0.01)
        receipt = await manager.write_to_active(core_session_id="sess_pty", worker_id="worker_pty", text="finish\n")
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        await manager.close()

    assert receipt is not None
    assert receipt.worker_run_id == "run_worker_pty"
    assert "echo:hello" in "".join(outputs)
    assert "echo:finish" in "".join(outputs)
    assert "<<<DONE>>>" not in "".join(outputs)
    assert "echo:finish" in result


@pytest.mark.asyncio
async def test_pty_session_replays_startup_output_when_turn_begins(tmp_path) -> None:
    script = (
        "import sys\n"
        "print('startup prompt: Continue? [y/N]', flush=True)\n"
        "for line in sys.stdin:\n"
        "    text = line.strip()\n"
        "    print('input:' + text, flush=True)\n"
        "    if text == 'y':\n"
        "        print('<<<DONE>>>', flush=True)\n"
        "        break\n"
    )
    manager = PtySessionManager()
    outputs: list[str] = []
    try:
        session = await manager.get_or_create(
            PtySessionKey(core_session_id="sess_startup", worker_id="worker_startup", executor_type="test_pty"),
            command=["python3", "-u", "-c", script],
            cwd=tmp_path,
        )
        await asyncio.sleep(0.05)

        async def on_output(text: str) -> None:
            outputs.append(text)

        task = asyncio.create_task(
            session.run_turn(
                worker_run_id="run_worker_startup",
                prompt="ignored",
                completion_marker="<<<DONE>>>",
                output_callback=on_output,
            )
        )
        for _ in range(50):
            if "startup prompt" in "".join(outputs):
                break
            await asyncio.sleep(0.01)
        receipt = await manager.write_to_active(core_session_id="sess_startup", worker_id="worker_startup", text="y\n")
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        await manager.close()

    joined = "".join(outputs)
    assert receipt is not None
    assert "startup prompt: Continue? [y/N]" in joined
    assert "input:y" in joined
    assert "<<<DONE>>>" not in joined
    assert "startup prompt: Continue? [y/N]" in result


@pytest.mark.asyncio
async def test_pty_session_detects_marker_split_by_terminal_repaint_whitespace(tmp_path) -> None:
    marker = "<<<DONE>>>"
    split_marker = "\n".join(marker)
    script = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    print('answer-ok', flush=True)\n"
        f"    print({split_marker!r}, flush=True)\n"
        "    break\n"
    )
    manager = PtySessionManager()
    outputs: list[str] = []
    try:
        session = await manager.get_or_create(
            PtySessionKey(core_session_id="sess_marker", worker_id="worker_marker", executor_type="test_pty"),
            command=["python3", "-u", "-c", script],
            cwd=tmp_path,
        )

        async def on_output(text: str) -> None:
            outputs.append(text)

        result = await asyncio.wait_for(
            session.run_turn(
                worker_run_id="run_worker_marker",
                prompt="go",
                completion_marker=marker,
                output_callback=on_output,
            ),
            timeout=2,
        )
    finally:
        await manager.close()

    joined = "".join(outputs)
    assert "answer-ok" in joined
    assert marker not in joined.replace("\n", "")
    assert "answer-ok" in result


def test_codex_command_uses_local_binary_and_configured_flags(tmp_path) -> None:
    command = _codex_command(
        {
            "binary": "/usr/local/bin/codex",
            "model": "gpt-test",
            "profile": "local",
            "sandbox": "read-only",
            "ask_for_approval": "never",
            "args": ["--search"],
        },
        cwd=tmp_path,
        initial_prompt="hello",
    )

    assert command == [
        "/usr/local/bin/codex",
        "--no-alt-screen",
        "--cd",
        str(tmp_path),
        "--model",
        "gpt-test",
        "--profile",
        "local",
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "--search",
        "hello",
    ]


def test_codex_command_splits_string_override() -> None:
    assert _codex_command({"command": "codex --no-alt-screen --cd /tmp"}, cwd=Path("/ignored"), initial_prompt="hello world") == [
        "codex",
        "--no-alt-screen",
        "--cd",
        "/tmp",
        "hello world",
    ]


def test_claude_code_command_uses_help_confirmed_interactive_flags(tmp_path) -> None:
    command = _claude_code_command(
        {
            "binary": "/usr/local/bin/claude",
            "model": "sonnet",
            "permission_mode": "plan",
            "args": ["--add-dir", str(tmp_path / "docs")],
        },
        cwd=tmp_path,
        initial_prompt="hello\nworld",
    )

    assert command == [
        "/usr/local/bin/claude",
        "--ax-screen-reader",
        "--model",
        "sonnet",
        "--permission-mode",
        "plan",
        "--add-dir",
        str(tmp_path / "docs"),
        "hello\nworld",
    ]


def test_claude_code_command_can_disable_ax_screen_reader(tmp_path) -> None:
    command = _claude_code_command(
        {"binary": "claude", "ax_screen_reader": False},
        cwd=tmp_path,
        initial_prompt=None,
    )

    assert command == ["claude"]


def test_claude_code_command_splits_string_override() -> None:
    assert _claude_code_command({"command": "claude --safe-mode"}, cwd=Path("/ignored"), initial_prompt="hello world") == [
        "claude",
        "--safe-mode",
        "hello world",
    ]


def test_codex_prompt_does_not_embed_full_completion_marker() -> None:
    marker = "<<<AGENTHUB_DONE:run_worker_codex>>>"
    prompt = _prompt_text(_worker_context(), completion_marker=marker)

    assert marker not in prompt
    assert "AGENTHUB_DONE" in prompt
    assert "run_worker_codex" in prompt
    assert "three `>` characters" in prompt


def test_codex_executor_uses_bracketed_paste_for_turn_input(tmp_path) -> None:
    executor = CodexPtyWorkerExecutor(pty_manager=PtySessionManager(), project_dir=tmp_path)

    assert executor.format_turn_input("hello\nworld", {}) == "hello world"
    assert executor.format_turn_input("hello\nworld", {"bracketed_paste": True}) == "\x1b[200~hello\nworld\x1b[201~"
    assert executor.input_suffix({}) == "\r"


@pytest.mark.asyncio
async def test_codex_executor_reuses_session_and_writes_later_turns_to_pty(tmp_path) -> None:
    script = (
        "import sys\n"
        "print('first:' + sys.argv[1], flush=True)\n"
        "print('<<<AGENTHUB_DONE:run_worker_codex_first>>>', flush=True)\n"
        "for line in sys.stdin:\n"
        "    print('next:' + line.strip(), flush=True)\n"
        "    print('<<<AGENTHUB_DONE:run_worker_codex_second>>>', flush=True)\n"
        "    break\n"
    )
    manager = PtySessionManager()
    runtime = _RecordingRuntime()
    executor = CodexPtyWorkerExecutor(pty_manager=manager, project_dir=tmp_path)
    try:
        first = await asyncio.wait_for(
            executor.run(
                runtime,
                _worker_context(
                    worker_run_id="run_worker_codex_first",
                    instruction="First prompt",
                    executor_config={"command": ["python3", "-u", "-c", script], "startup_ready_pattern": ""},
                ),
            ),
            timeout=2,
        )
        second = await asyncio.wait_for(
            executor.run(
                runtime,
                _worker_context(
                    worker_run_id="run_worker_codex_second",
                    instruction="Second prompt",
                    executor_config={"command": ["python3", "-u", "-c", script], "startup_ready_pattern": ""},
                ),
            ),
            timeout=2,
        )
    finally:
        await manager.close()

    assert "First prompt" in first.text
    assert "next:Second prompt" in second.text
    assert "first:Second prompt" not in second.text


class _RecordingRuntime:
    async def _emit_child_run_event(self, **kwargs) -> None:
        return None


def _worker_context(
    *,
    worker_run_id: str = "run_worker_codex",
    instruction: str = "Do the test task.",
    executor_config: dict | None = None,
) -> WorkerExecutorContext:
    return WorkerExecutorContext(
        session_id="sess_test",
        parent_run_id="run_parent",
        parent_agent_id="agent_orchestrator",
        task_id="task_test",
        instruction=instruction,
        worker=WorkerRuntimeState(worker_id="codex_pty_worker", pool_id="default", agent_definition_id="codex_pty_agent"),
        worker_agent_id="agent_worker_codex",
        worker_run_id=worker_run_id,
        worker_start_node=None,  # type: ignore[arg-type]
        worker_stream=None,  # type: ignore[arg-type]
        executor_config=executor_config,
    )
