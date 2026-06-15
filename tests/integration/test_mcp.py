from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_core import AgentRuntime
from agent_core.providers import ProviderRegistry
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.runtime import RunStatus
from agent_core.types.tools import ToolCall
from tests.conftest import write_config
from tests.fixtures.fake_provider import FakeProvider


def write_mcp_server(path: Path) -> None:
    path.write_text(
        r'''
from __future__ import annotations

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "write_note",
        "description": "Write a note",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def read_message():
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            break
        key, _, value = stripped.decode("ascii").partition(":")
        if key.lower() == "content-length":
            content_length = int(value.strip())
    if content_length is None:
        return None
    return json.loads(sys.stdin.buffer.read(content_length).decode("utf-8"))


def write_message(message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload)
    sys.stdout.buffer.flush()


while True:
    request = read_message()
    if request is None:
        break
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.1"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        result = {"content": [{"type": "text", "text": f"{name}:{arguments.get('text', '')}"}]}
    else:
        write_message({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "method not found"}})
        continue
    write_message({"jsonrpc": "2.0", "id": request["id"], "result": result})
'''.lstrip(),
        encoding="utf-8",
    )


def write_mcp_config(home: Path, server_script: Path, *, disabled_tools: list[str] | None = None) -> None:
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "local": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                    },
                    "broken": {
                        "command": str(home / "missing-mcp-server"),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    if disabled_tools:
        with (home / "config.toml").open("a", encoding="utf-8") as fh:
            fh.write("\n[tools.mcp]\n")
            fh.write("disabled_tools = " + json.dumps(disabled_tools) + "\n")


@pytest.mark.asyncio
async def test_mcp_tools_are_discovered_and_callable(isolated_dirs, tmp_path: Path) -> None:
    home, project = isolated_dirs
    write_config(home)
    server_script = tmp_path / "fake_mcp_server.py"
    write_mcp_server(server_script)
    write_mcp_config(home, server_script)
    fake = FakeProvider(
        final_text="done",
        tool_call=ToolCall(tool_call_id="call1", name="mcp.local.echo", arguments={"text": "hello"}),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)

    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("use mcp")
        events = [event.event_type async for event in handle.events()]

    tool_names = {tool.name for tool in fake.requests[0].tools}
    assert "mcp.local.echo" in tool_names
    assert "mcp.local.write_note" in tool_names
    assert not any(name.startswith("mcp.broken.") for name in tool_names)
    assert "mcp_server_failed" in events
    assert "tool_completed" in events
    assert handle.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_mcp_disabled_tool_is_hidden_from_provider(isolated_dirs, tmp_path: Path) -> None:
    home, project = isolated_dirs
    write_config(home)
    server_script = tmp_path / "fake_mcp_server.py"
    write_mcp_server(server_script)
    write_mcp_config(home, server_script, disabled_tools=["mcp.local.write_note"])
    fake = FakeProvider(final_text="done")
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)

    async with AgentRuntime(project_dir=project, provider_registry=registry) as runtime:
        handle = await runtime.start("hello")
        _events = [event.event_type async for event in handle.events()]

    tool_names = {tool.name for tool in fake.requests[0].tools}
    assert "mcp.local.echo" in tool_names
    assert "mcp.local.write_note" not in tool_names


@pytest.mark.asyncio
async def test_mcp_write_tool_uses_permission_callback(isolated_dirs, tmp_path: Path) -> None:
    home, project = isolated_dirs
    write_config(home)
    server_script = tmp_path / "fake_mcp_server.py"
    write_mcp_server(server_script)
    write_mcp_config(home, server_script)
    requests = []

    async def deny(request):
        requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    fake = FakeProvider(
        final_text="done",
        tool_call=ToolCall(tool_call_id="call1", name="mcp.local.write_note", arguments={"text": "secret"}),
    )
    registry = ProviderRegistry()
    registry.register("fake", lambda config: fake)

    async with AgentRuntime(project_dir=project, provider_registry=registry, permission_callback=deny) as runtime:
        handle = await runtime.start("write through mcp")
        events = [event.event_type async for event in handle.events()]

    assert requests
    assert requests[0].tool_name == "mcp.local.write_note"
    assert requests[0].permission == "write"
    assert "mcp" in requests[0].tags
    assert "tool_denied" in events
    assert handle.status == RunStatus.COMPLETED
