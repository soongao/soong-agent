from __future__ import annotations

import pytest

from agent_core import AgentRuntime
from tests.conftest import write_config
from tests.fixtures.ollama import ollama_server


@pytest.mark.asyncio
async def test_ollama_gemma4_simple_run(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama")
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace('base_url = ""', f'base_url = "{ollama_server}"')
    text = text.replace('name = "fake-model"', 'name = "gemma4"')
    text = text.replace("timeout_ms = 1000", "timeout_ms = 60000")
    config_path.write_text(text, encoding="utf-8")
    async with AgentRuntime(project_dir=project) as runtime:
        handle = await runtime.start("Reply with exactly: pong")
        events = [event.event_type async for event in handle.events()]
    assert "loop_completed" in events


@pytest.mark.asyncio
async def test_ollama_gemma4_tool_call_list_dir(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(home, provider="ollama")
    (project / "marker_tool_file.txt").write_text("marker\n", encoding="utf-8")
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace('base_url = ""', f'base_url = "{ollama_server}"')
    text = text.replace('name = "fake-model"', 'name = "gemma4"')
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
