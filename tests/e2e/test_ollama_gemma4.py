from __future__ import annotations

from io import StringIO

import pytest

from agent_core import AgentRuntime
from agent_cli.cli import async_main
from tests.conftest import write_config
from tests.fixtures.ollama import ollama_server


@pytest.mark.asyncio
async def test_ollama_gemma4_simple_run(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama", base_url=ollama_server, model_name="gemma4")
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("timeout_ms = 1000", "timeout_ms = 60000")
    config_path.write_text(text, encoding="utf-8")
    async with AgentRuntime(project_dir=project) as runtime:
        handle = await runtime.start("Reply with exactly: pong")
        events = [event.event_type async for event in handle.events()]
    assert "loop_completed" in events


@pytest.mark.asyncio
async def test_cli_chat_uses_local_ollama(isolated_dirs, ollama_server, capsys, monkeypatch) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama", base_url=ollama_server, model_name="gemma4")
    monkeypatch.setattr("sys.stdin", StringIO("Reply with exactly: cli-pong\n/exit\n"))
    code = await async_main(["chat", "--path", str(project), "--plain"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert "cli-pong" in captured.out


@pytest.mark.asyncio
async def test_ollama_gemma4_tool_call_list_dir(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama", base_url=ollama_server, model_name="gemma4")
    (project / "marker_tool_file.txt").write_text("marker\n", encoding="utf-8")
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("timeout_ms = 1000", "timeout_ms = 60000")
    config_path.write_text(text, encoding="utf-8")

    async with AgentRuntime(project_dir=project) as runtime:
        handle = await runtime.start(
            "Use the available directory listing tool on the current project directory. "
            "After the tool result, reply with exactly the file name marker_tool_file.txt if it is present.",
            session_id="sess_ollama_tool",
        )
        events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_ollama_tool")

    event_types = [event.event_type for event in events]
    assert "tool_started" in event_types
    assert "tool_completed" in event_types
    assert "loop_completed" in event_types
    assert any(event.payload.get("name") == "code.list_dir" for event in events if event.event_type == "tool_completed")
    assistant_text = "\n".join(
        getattr(block, "text", "")
        for node in replay.nodes
        if node.role == "assistant"
        for block in node.content
        if getattr(block, "type", None) == "text"
    )
    assert "marker_tool_file.txt" in assistant_text


@pytest.mark.asyncio
async def test_ollama_gemma4_memory_extraction_writes_user_memory(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama", base_url=ollama_server, model_name="gemma4", memory_enabled=True)
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("timeout_ms = 1000", "timeout_ms = 60000")
    config_path.write_text(text, encoding="utf-8")

    async with AgentRuntime(project_dir=project) as runtime:
        handle = await runtime.start("记住：我是一个 Go 和 Python 的后端开发", session_id="sess_ollama_memory")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_ollama_memory")

    memory_files = list((home / "memory" / "user").glob("*.md"))
    assert memory_files
    memory_text = "\n".join(path.read_text(encoding="utf-8") for path in memory_files)
    assert "Go" in memory_text or "go" in memory_text
    assert "Python" in memory_text or "python" in memory_text
    completed = [event for event in replay.events if event.event_type == "memory_extraction_completed"]
    assert completed
    assert completed[-1].payload["reason"] == "explicit"
    assert completed[-1].payload["created_memory_ids"]
