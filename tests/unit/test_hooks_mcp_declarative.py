from __future__ import annotations

import json
import sys

import pytest

from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.hooks.loader import load_hooks, normalize_hooks
from agent_core.mcp.discovery import McpDiscovery
from agent_core.permissions import PermissionSessionCache
from agent_core.tools.builtin_code import register_builtin_code_tools
from agent_core.tools.declarative import load_declarative_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.tools import ToolCall, ToolDefinition
from tests.conftest import write_config


async def network_echo_handler(_context, args):
    return {"url": args["url"]}


async def make_context(home, project, *, hooks=None) -> ToolExecutionContext:
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    return ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
        hooks=hooks,
    )


@pytest.mark.asyncio
async def test_user_hook_deny_blocks_tool(isolated_dirs) -> None:
    home, project = isolated_dirs
    hooks = [{"tool_name": "code.write_file", "action": "deny", "reason": "blocked"}]
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="code.write_file", arguments={"path": "x.txt", "content": "x"}),
        await make_context(home, project, hooks=hooks),
    )
    assert result.is_error
    assert result.error and result.error.message == "blocked"


def test_load_hooks_user_level_only(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "hooks.json").write_text(json.dumps({"hooks": [{"tool_name": "code.write_file", "action": "deny"}]}), encoding="utf-8")
    project_hooks = project / ".soong-agent" / "hooks.json"
    project_hooks.parent.mkdir()
    project_hooks.write_text(json.dumps({"hooks": [{"tool_name": "code.read_file", "action": "deny"}]}), encoding="utf-8")
    hooks = normalize_hooks(load_hooks(home))
    assert hooks == [{"tool_name": "code.write_file", "action": "deny"}]


def test_normalize_grouped_command_hooks(isolated_dirs) -> None:
    home, _project = isolated_dirs
    config = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": {"tool_name": "code.write_file"},
                    "hooks": [{"type": "command", "command": "python deny.py"}],
                }
            ]
        }
    }
    hooks = normalize_hooks(config)
    assert hooks == [{"tool_name": "code.write_file", "type": "command", "command": "python deny.py", "event_type": "tool_started"}]


@pytest.mark.asyncio
async def test_command_hook_deny_runs_before_permission_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    script = home / "deny_hook.py"
    script.write_text(
        "import json, sys\n"
        "payload=json.load(sys.stdin)\n"
        "assert payload['tool_name']=='code.write_file'\n"
        "print(json.dumps({'decision':'deny','reason':'command blocked','metadata':{'seen': True}}))\n",
        encoding="utf-8",
    )
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    permission_calls = 0

    async def allow(_request):
        nonlocal permission_calls
        permission_calls += 1
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
        hooks=[
            {
                "event_type": "tool_started",
                "tool_name": "code.write_file",
                "type": "command",
                "command": [sys.executable, str(script)],
            }
        ],
    )
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="code.write_file", arguments={"path": "x.txt", "content": "x"}),
        context,
    )
    assert result.is_error
    assert result.error and result.error.message == "command blocked"
    assert result.metadata["hook_summary"]["metadata"] == {"seen": True}
    assert permission_calls == 0
    assert not (project / "x.txt").exists()


@pytest.mark.asyncio
async def test_pre_tool_hook_summary_is_passed_to_permission_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    script = home / "allow_hook.py"
    script.write_text(
        "import json, sys\n"
        "payload=json.load(sys.stdin)\n"
        "print(json.dumps({'decision':'allow','reason':'checked','metadata':{'tool': payload['tool_name']},'logs':['ok']}))\n",
        encoding="utf-8",
    )
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    requests = []

    async def allow(request):
        requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
        hooks=[
            {
                "event_type": "tool_started",
                "tool_name": "code.write_file",
                "type": "command",
                "command": [sys.executable, str(script)],
            }
        ],
    )
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="code.write_file", arguments={"path": "x.txt", "content": "x"}),
        context,
    )

    assert not result.is_error
    assert requests[0].hook_summary == {
        "action": "allow",
        "reason": "checked",
        "metadata": {"tool": "code.write_file"},
        "logs": ["ok"],
        "error": None,
    }


@pytest.mark.asyncio
async def test_command_hook_uses_env_allowlist(isolated_dirs, monkeypatch) -> None:
    home, project = isolated_dirs
    monkeypatch.setenv("SOONG_HOOK_SECRET", "should-not-leak")
    output_path = home / "env_seen.json"
    script = home / "env_hook.py"
    script.write_text(
        "import json, os, pathlib\n"
        f"path=pathlib.Path({str(output_path)!r})\n"
        "path.write_text(json.dumps({'secret': os.environ.get('SOONG_HOOK_SECRET'), 'path': bool(os.environ.get('PATH'))}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)

    async def allow(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=allow,
        permission_cache=PermissionSessionCache(),
        hooks=[
            {
                "event_type": "tool_started",
                "tool_name": "code.write_file",
                "type": "command",
                "command": [sys.executable, str(script)],
            }
        ],
    )
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="code.write_file", arguments={"path": "x.txt", "content": "x"}),
        context,
    )
    seen = json.loads(output_path.read_text(encoding="utf-8"))

    assert not result.is_error
    assert seen == {"secret": None, "path": True}


@pytest.mark.asyncio
async def test_network_tool_confirm_populates_permission_request_host(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    requests = []

    async def callback(request):
        requests.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=callback,
        permission_cache=PermissionSessionCache(),
    )
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="user.fetch",
            description="fetch",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            permission="readonly",
            tags={"network"},
        ),
        network_echo_handler,
    )
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.fetch", arguments={"url": "https://api.example.com/v1"}),
        context,
    )

    assert not result.is_error
    assert requests[0].network_host == "api.example.com"
    assert requests[0].target_scope == "network:api.example.com"


@pytest.mark.asyncio
async def test_network_tool_allowed_host_skips_permission_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    config.permissions.network_policy.allowed_hosts = ["api.example.com"]
    permission_calls = 0

    async def callback(_request):
        nonlocal permission_calls
        permission_calls += 1
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=callback,
        permission_cache=PermissionSessionCache(),
    )
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="user.fetch",
            description="fetch",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            permission="readonly",
            tags={"network"},
        ),
        network_echo_handler,
    )
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.fetch", arguments={"url": "https://api.example.com/v1"}),
        context,
    )

    assert not result.is_error
    assert permission_calls == 0


@pytest.mark.asyncio
async def test_network_tool_policy_deny_blocks_without_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    config.permissions.network_policy.default = "deny"
    permission_calls = 0

    async def callback(_request):
        nonlocal permission_calls
        permission_calls += 1
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = ToolExecutionContext(
        session_id="sess",
        run_id="run",
        agent_id="agent",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=callback,
        permission_cache=PermissionSessionCache(),
    )
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name="user.fetch",
            description="fetch",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            permission="readonly",
            tags={"network"},
        ),
        network_echo_handler,
    )
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.fetch", arguments={"url": "https://api.example.com/v1"}),
        context,
    )

    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"
    assert result.metadata["network_host"] == "api.example.com"
    assert permission_calls == 0


@pytest.mark.asyncio
async def test_post_tool_use_command_hook_observes_result(isolated_dirs) -> None:
    home, project = isolated_dirs
    output_path = home / "post_payload.json"
    script = home / "post_hook.py"
    script.write_text(
        "import json, pathlib, sys\n"
        f"path=pathlib.Path({str(output_path)!r})\n"
        "payload=json.load(sys.stdin)\n"
        "path.write_text(json.dumps(payload), encoding='utf-8')\n"
        "print(json.dumps({'decision':'deny','reason':'ignored by post hook'}))\n",
        encoding="utf-8",
    )
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="code.read_file", arguments={"path": "a.txt"}),
        await make_context(
            home,
            project,
            hooks=[
                {
                    "event_type": "tool_completed",
                    "tool_name": "code.read_file",
                    "type": "command",
                    "command": [sys.executable, str(script)],
                }
            ],
        ),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "file_not_found"

    (project / "a.txt").write_text("hello", encoding="utf-8")
    result = await registry.execute(
        ToolCall(tool_call_id="c2", name="code.read_file", arguments={"path": "a.txt"}),
        await make_context(
            home,
            project,
            hooks=[
                {
                    "event_type": "tool_completed",
                    "tool_name": "code.read_file",
                    "type": "command",
                    "command": [sys.executable, str(script)],
                }
            ],
        ),
    )
    assert not result.is_error
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["event_type"] == "PostToolUse"
    assert payload["tool_name"] == "code.read_file"
    assert payload["result"]["content"][0]["data"]["content"] == "hello"
    assert result.metadata["post_hook_summary"]["decision"] == "deny"


@pytest.mark.asyncio
async def test_declarative_tool_user_level_exec(isolated_dirs) -> None:
    home, project = isolated_dirs
    tools_dir = home / "tools"
    tools_dir.mkdir()
    (tools_dir / "echo.json").write_text(
        json.dumps(
            {
                "name": "user.echo",
                "description": "echo",
                "permission": "readonly",
                "tags": ["declarative"],
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                "command_type": "exec",
                "command": ["python3", "-c", "import sys; print(sys.argv[1])", "{{args.text}}"],
            }
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry()
    load_declarative_tools(registry, home)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.echo", arguments={"text": "hello"}),
        await make_context(home, project),
    )
    assert not result.is_error
    assert result.content[0].data["stdout"].strip() == "hello"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_declarative_tool_large_stdout_uses_artifact(isolated_dirs) -> None:
    home, project = isolated_dirs
    tools_dir = home / "tools"
    tools_dir.mkdir()
    (tools_dir / "large.json").write_text(
        json.dumps(
            {
                "name": "user.large",
                "description": "large",
                "permission": "readonly",
                "tags": ["declarative"],
                "input_schema": {"type": "object", "properties": {}},
                "command_type": "exec",
                "command": ["python3", "-c", "print('x' * 200)"],
                "stdout_limit_bytes": 32,
            }
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry()
    load_declarative_tools(registry, home)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.large", arguments={}),
        await make_context(home, project),
    )
    assert not result.is_error
    assert result.content[0].data["truncated"] is True  # type: ignore[union-attr]
    assert result.metadata["stdout_artifact_id"]
    artifact = home / "sessions" / "sess" / "artifacts" / result.metadata["stdout_artifact_id"]
    assert list(artifact.iterdir())[0].read_text(encoding="utf-8").strip() == "x" * 200


@pytest.mark.asyncio
async def test_declarative_tool_working_dir_outside_allowed_roots_denied(isolated_dirs, tmp_path) -> None:
    home, project = isolated_dirs
    outside = tmp_path / "outside"
    outside.mkdir()
    tools_dir = home / "tools"
    tools_dir.mkdir()
    (tools_dir / "outside.json").write_text(
        json.dumps(
            {
                "name": "user.outside",
                "description": "outside",
                "permission": "readonly",
                "tags": ["declarative"],
                "input_schema": {"type": "object", "properties": {}},
                "command_type": "exec",
                "command": ["python3", "-c", "print('ok')"],
                "working_dir": str(outside),
            }
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry()
    load_declarative_tools(registry, home)
    result = await registry.execute(
        ToolCall(tool_call_id="c1", name="user.outside", arguments={}),
        await make_context(home, project),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "write_outside_allowed_roots"


def test_mcp_discovery_disabled_rules(isolated_dirs) -> None:
    home, project = isolated_dirs
    write_config(home)
    config, _paths = load_runtime_config(project_dir=project)
    config.tools.mcp.disabled_servers.append("s2")
    config.tools.mcp.disabled_tools.append("mcp.s1.bad")
    discovery = McpDiscovery({"servers": {"s1": {}, "s2": {}}}, config.tools)
    assert discovery.available_servers() == ["s1"]
    assert discovery.tool_enabled("mcp.s1.ok")
    assert not discovery.tool_enabled("mcp.s1.bad")
