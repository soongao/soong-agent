from __future__ import annotations

from pathlib import Path
import sys

import pytest

from agent_core.artifacts import ArtifactManager
from agent_core.config import load_runtime_config
from agent_core.permissions import PermissionSessionCache
from agent_core.tools.builtin_code import register_builtin_code_tools
from agent_core.tools.execution import ToolExecutionContext
from agent_core.tools.registry import ToolRegistry
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.tools import ToolCall
from tests.conftest import write_config


async def make_context(home: Path, project: Path, callback=None) -> ToolExecutionContext:
    write_config(home)
    config, paths = load_runtime_config(project_dir=project)
    return ToolExecutionContext(
        session_id="sess_test",
        run_id="run_test",
        agent_id="agent_main",
        agent_role="main",
        project_dir=paths.project_dir,
        home_dir=paths.home_dir,
        config=config,
        artifact_manager=ArtifactManager(home_dir=paths.home_dir),
        permission_callback=callback,
        permission_cache=PermissionSessionCache(),
    )


@pytest.mark.asyncio
async def test_read_file_paging_and_line_cap(isolated_dirs) -> None:
    home, project = isolated_dirs
    long = "x" * 5000
    path = project / "a.txt"
    path.write_text(long + "\nsecond\n", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": str(path), "max_lines": 1}),
        await make_context(home, project),
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert data["truncated"] is True
    assert data["truncated_lines"] == [1]


@pytest.mark.asyncio
async def test_read_file_binary_returns_metadata_without_artifact(isolated_dirs) -> None:
    home, project = isolated_dirs
    path = project / "blob.bin"
    path.write_bytes(b"abc\x00def")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": str(path)}),
        await make_context(home, project),
    )
    data = result.content[0].data  # type: ignore[union-attr]
    assert not result.is_error
    assert data["binary"] is True
    assert data["content"] == ""
    assert not result.metadata.get("output_artifact_id")


@pytest.mark.asyncio
async def test_read_file_allows_non_sensitive_path_outside_project(isolated_dirs, tmp_path) -> None:
    home, project = isolated_dirs
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": str(outside)}),
        await make_context(home, project),
    )
    assert not result.is_error
    assert result.content[0].data["content"] == "outside"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_list_dir_nonrecursive_and_recursive_limit(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "root.txt").write_text("root", encoding="utf-8")
    nested = project / "nested"
    nested.mkdir()
    (nested / "child.txt").write_text("child", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 4096

    shallow = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.list_dir", arguments={"path": str(project), "recursive": False}),
        context,
    )
    shallow_names = {entry["name"] for entry in shallow.content[0].data["entries"]}  # type: ignore[union-attr]
    assert shallow_names == {"nested", "root.txt"}

    recursive = await registry.execute(
        ToolCall(tool_call_id="call2", name="code.list_dir", arguments={"path": str(project), "recursive": True, "limit": 2}),
        context,
    )
    data = recursive.content[0].data  # type: ignore[union-attr]
    assert len(data["entries"]) == 2
    assert data["truncated"] is True


@pytest.mark.asyncio
async def test_search_defaults_to_project_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "needle.txt").write_text("needle\n", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 4096
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.search", arguments={"query": "needle"}),
        context,
    )
    matches = result.content[0].data["matches"]  # type: ignore[union-attr]
    assert any(match["path"].endswith("needle.txt") for match in matches)


@pytest.mark.asyncio
async def test_search_query_starting_with_dash_is_treated_as_pattern(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "dash.txt").write_text("-needle\n", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 4096

    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.search", arguments={"query": "-needle"}),
        context,
    )

    assert not result.is_error
    matches = result.content[0].data["matches"]  # type: ignore[union-attr]
    assert any(match["path"].endswith("dash.txt") and match["text"] == "-needle" for match in matches)


@pytest.mark.asyncio
async def test_sensitive_read_denied_without_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    sensitive = project / ".env"
    sensitive.write_text("TOKEN=secret", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.read_file", arguments={"path": str(sensitive)}),
        await make_context(home, project),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"


@pytest.mark.asyncio
async def test_search_default_project_path_with_sensitive_file_requires_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / ".env").write_text("TOKEN=secret", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)

    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.search", arguments={"query": "TOKEN"}),
        await make_context(home, project),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"


@pytest.mark.asyncio
async def test_list_dir_recursive_sensitive_file_requires_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    secrets = project / "nested"
    secrets.mkdir()
    (secrets / "secret.pem").write_text("secret", encoding="utf-8")
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 4096

    recursive = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.list_dir", arguments={"path": str(project), "recursive": True}),
        context,
    )
    nonrecursive = await registry.execute(
        ToolCall(tool_call_id="call2", name="code.list_dir", arguments={"path": str(project), "recursive": False}),
        context,
    )

    assert recursive.is_error
    assert recursive.error and recursive.error.code.value == "permission_denied"
    assert not nonrecursive.is_error
    entries = nonrecursive.content[0].data["entries"]  # type: ignore[union-attr]
    assert [entry["name"] for entry in entries] == ["nested"]


@pytest.mark.asyncio
async def test_permission_scope_uses_resolved_path(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []
    env_file = project / ".env"
    env_file.write_text("secret", encoding="utf-8")

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call2", name="code.read_file", arguments={"path": ".env"}),
        await make_context(home, project, callback),
    )
    assert result.is_error is False
    assert calls[0].target_scope == str(env_file.resolve())


@pytest.mark.asyncio
async def test_permission_request_args_summary_is_redacted(isolated_dirs, monkeypatch) -> None:
    home, project = isolated_dirs
    monkeypatch.setenv("SOONG_PERMISSION_TEST_SECRET", "permission-secret-value")
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "secret.txt", "content": "api_key=permission-secret-value"},
        ),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert calls
    assert "permission-secret-value" not in calls[0].args_summary
    assert "[REDACTED]" in calls[0].args_summary
    assert not (project / "secret.txt").exists()


@pytest.mark.asyncio
async def test_list_dir_large_output_uses_artifact(isolated_dirs) -> None:
    home, project = isolated_dirs
    for index in range(20):
        (project / f"file_{index}.txt").write_text("x", encoding="utf-8")
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 80
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.list_dir", arguments={"path": str(project), "recursive": False}),
        context,
    )
    assert not result.is_error
    assert result.metadata["output_artifact_id"]
    assert result.content[0].data["truncated"] is True  # type: ignore[union-attr]
    artifact_id = result.metadata["output_artifact_id"]
    artifact_files = list((home / "sessions" / "sess_test" / "artifacts" / artifact_id).iterdir())
    assert artifact_files
    assert "file_0.txt" in artifact_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_run_command_readonly_ls_does_not_request_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "visible.txt").write_text("ok", encoding="utf-8")
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.run_command", arguments={"argv": ["ls"], "cwd": str(project)}),
        await make_context(home, project, callback),
    )

    assert not result.is_error
    assert "visible.txt" in result.content[0].data["stdout"]  # type: ignore[union-attr]
    assert calls == []


@pytest.mark.asyncio
async def test_run_command_write_like_command_still_requests_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.run_command", arguments={"argv": ["python3", "-c", "print('x')"], "cwd": str(project)}),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"
    assert len(calls) == 1
    assert calls[0].tool_name == "code.run_command"


@pytest.mark.asyncio
async def test_run_command_recursive_ls_still_requests_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.DENY)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.run_command", arguments={"argv": ["ls", "-R"], "cwd": str(project)}),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_search_large_output_uses_artifact(isolated_dirs) -> None:
    home, project = isolated_dirs
    for index in range(20):
        (project / f"file_{index}.txt").write_text(f"needle {index}\n", encoding="utf-8")
    context = await make_context(home, project)
    context.config.tools.stdout_limit_bytes = 80
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.search", arguments={"query": "needle", "path": str(project)}),
        context,
    )
    assert not result.is_error
    assert result.metadata["output_artifact_id"]
    assert result.content[0].data["truncated"] is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_write_requires_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "hello"},
        ),
        await make_context(home, project, callback),
    )
    assert not result.is_error
    assert (project / "out.txt").read_text(encoding="utf-8") == "hello"
    assert calls and calls[0].tool_name == "code.write_file"


@pytest.mark.asyncio
async def test_allow_once_write_permission_asks_again_for_same_scope(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project, callback)

    first = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "one"},
        ),
        context,
    )
    second = await registry.execute(
        ToolCall(
            tool_call_id="call2",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "two", "overwrite": True},
        ),
        context,
    )

    assert not first.is_error
    assert not second.is_error
    assert len(calls) == 2
    assert calls[0].target_scope == calls[1].target_scope == str((project / "out.txt").resolve())
    assert (project / "out.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_write_denied_without_callback(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "hello"},
        ),
        await make_context(home, project),
    )
    assert result.is_error
    assert not (project / "out.txt").exists()


@pytest.mark.asyncio
async def test_write_file_conflict_and_create_dirs(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "exists.txt"
    target.write_text("old", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    context = await make_context(home, project, callback)

    conflict = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "exists.txt", "content": "new", "overwrite": False},
        ),
        context,
    )
    assert conflict.is_error
    assert conflict.error and conflict.error.code.value == "path_conflict"
    assert target.read_text(encoding="utf-8") == "old"

    created = await registry.execute(
        ToolCall(
            tool_call_id="call2",
            name="code.write_file",
            arguments={"path": "new/dir/file.txt", "content": "new", "create_dirs": True},
        ),
        context,
    )
    assert not created.is_error
    assert (project / "new" / "dir" / "file.txt").read_text(encoding="utf-8") == "new"
    assert created.content[0].data["created"] is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_write_file_create_dirs_false_reports_missing_parent(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "missing/dir/file.txt", "content": "new", "create_dirs": False},
        ),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "file_not_found"
    assert not (project / "missing").exists()


@pytest.mark.asyncio
async def test_write_without_callback_allow_applies_to_plain_write_tools(isolated_dirs) -> None:
    home, project = isolated_dirs
    context = await make_context(home, project)
    context.config.permissions.write_without_callback = "allow"
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "hello"},
        ),
        context,
    )
    assert not result.is_error
    assert (project / "out.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_write_without_callback_allow_does_not_allow_dangerous_run_command(isolated_dirs) -> None:
    home, project = isolated_dirs
    context = await make_context(home, project)
    context.config.permissions.write_without_callback = "allow"
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "print('x')"]},
        ),
        context,
    )
    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"


@pytest.mark.asyncio
async def test_run_command_allow_for_session_scope_includes_executable_when_cwd_is_present(isolated_dirs) -> None:
    home, project = isolated_dirs
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_FOR_SESSION)

    context = await make_context(home, project, callback)
    registry = ToolRegistry()
    register_builtin_code_tools(registry)

    first = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "print('one')"], "cwd": str(project)},
        ),
        context,
    )
    second = await registry.execute(
        ToolCall(
            tool_call_id="call2",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "print('two')"], "cwd": str(project)},
        ),
        context,
    )
    third = await registry.execute(
        ToolCall(
            tool_call_id="call3",
            name="code.run_command",
            arguments={"argv": ["/bin/echo", "three"], "cwd": str(project)},
        ),
        context,
    )

    assert not first.is_error
    assert not second.is_error
    assert not third.is_error
    assert len(calls) == 2
    assert calls[0].target_scope == f"{sys.executable}:{project.resolve()}"
    assert calls[1].target_scope == f"/bin/echo:{project.resolve()}"


@pytest.mark.asyncio
async def test_run_command_rejects_shell_string_before_handler(isolated_dirs) -> None:
    home, project = isolated_dirs
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.run_command", arguments={"argv": "echo hi"}),
        await make_context(home, project),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "validation_error"


@pytest.mark.asyncio
async def test_run_command_cwd_outside_allowed_roots_denied(isolated_dirs, tmp_path) -> None:
    home, project = isolated_dirs
    outside = tmp_path / "outside"
    outside.mkdir()

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "print('x')"], "cwd": str(outside)},
        ),
        await make_context(home, project, callback),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "write_outside_allowed_roots"


@pytest.mark.asyncio
async def test_edit_file_exact_replace_unique(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "edit.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={"path": "edit.txt", "edits": [{"old": "beta", "new": "BETA"}]},
        ),
        await make_context(home, project, callback),
    )

    assert not result.is_error
    assert result.content[0].data["edits_applied"] == 1  # type: ignore[union-attr]
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\n"


@pytest.mark.asyncio
async def test_edit_file_edits_schema_rejects_bad_items_before_permission(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "edit.txt").write_text("alpha\n", encoding="utf-8")
    calls = []

    async def callback(request):
        calls.append(request)
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={"path": "edit.txt", "edits": [{"old": "alpha"}]},
        ),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "validation_error"
    assert "$.edits[0].new is required" in result.error.message
    assert calls == []
    assert (project / "edit.txt").read_text(encoding="utf-8") == "alpha\n"


@pytest.mark.asyncio
async def test_edit_file_exact_replace_ambiguous_without_replace_all(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "edit.txt"
    target.write_text("same\nsame\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={"path": "edit.txt", "edits": [{"old": "same", "new": "changed"}]},
        ),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "ambiguous_edit"
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


@pytest.mark.asyncio
async def test_edit_file_exact_replace_all(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "edit.txt"
    target.write_text("same\nsame\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={"path": "edit.txt", "edits": [{"old": "same", "new": "changed", "replace_all": True}]},
        ),
        await make_context(home, project, callback),
    )

    assert not result.is_error
    assert result.content[0].data["edits_applied"] == 2  # type: ignore[union-attr]
    assert target.read_text(encoding="utf-8") == "changed\nchanged\n"


@pytest.mark.asyncio
async def test_edit_file_create_if_missing_failure_does_not_leave_empty_file(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={
                "path": "missing.txt",
                "edits": [{"old": "not present", "new": "new"}],
                "create_if_missing": True,
            },
        ),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "text_not_found"
    assert not (project / "missing.txt").exists()


@pytest.mark.asyncio
async def test_edit_file_create_if_missing_success_writes_file(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.edit_file",
            arguments={
                "path": "created.txt",
                "edits": [{"old": "", "new": "created\n"}],
                "create_if_missing": True,
            },
        ),
        await make_context(home, project, callback),
    )

    assert not result.is_error
    assert result.content[0].data["edits_applied"] == 1  # type: ignore[union-attr]
    assert (project / "created.txt").read_text(encoding="utf-8") == "created\n"


@pytest.mark.asyncio
async def test_edit_file_unified_diff_success(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "patch.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    diff = "--- a/patch.txt\n+++ b/patch.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.edit_file", arguments={"path": "patch.txt", "unified_diff": diff}),
        await make_context(home, project, callback),
    )

    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


@pytest.mark.asyncio
async def test_edit_file_unified_diff_path_mismatch(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "patch.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    diff = "--- a/other.txt\n+++ b/other.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.edit_file", arguments={"path": "patch.txt", "unified_diff": diff}),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "patch_path_mismatch"
    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


@pytest.mark.asyncio
async def test_edit_file_unified_diff_rejects_multi_file_patch(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "a.txt").write_text("a\n", encoding="utf-8")
    (project / "b.txt").write_text("b\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    diff = (
        "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+A\n"
        "--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-b\n+B\n"
    )
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.edit_file", arguments={"path": "a.txt", "unified_diff": diff}),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "patch_path_mismatch"
    assert (project / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (project / "b.txt").read_text(encoding="utf-8") == "b\n"


@pytest.mark.asyncio
async def test_edit_file_unified_diff_rejects_delete_patch(isolated_dirs) -> None:
    home, project = isolated_dirs
    target = project / "delete.txt"
    target.write_text("old\n", encoding="utf-8")

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    diff = "deleted file mode 100644\n--- a/delete.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(tool_call_id="call1", name="code.edit_file", arguments={"path": "delete.txt", "unified_diff": diff}),
        await make_context(home, project, callback),
    )

    assert result.is_error
    assert result.error and result.error.code.value == "patch_apply_failed"
    assert target.read_text(encoding="utf-8") == "old\n"


@pytest.mark.asyncio
async def test_permission_callback_exception_defaults_to_deny(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        raise RuntimeError("ui disconnected")

    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.write_file",
            arguments={"path": "out.txt", "content": "hello"},
        ),
        await make_context(home, project, callback),
    )
    assert result.is_error
    assert result.error and result.error.code.value == "permission_denied"
    assert result.metadata["permission_failed"] is True
    assert not (project / "out.txt").exists()


@pytest.mark.asyncio
async def test_run_command_truncated_stdout_sets_result_artifact_id(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = await make_context(home, project, callback)
    context.config.tools.stdout_limit_bytes = 8
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "print('x' * 40)"]},
        ),
        context,
    )
    data = result.content[0].data  # type: ignore[union-attr]
    artifact_id = result.metadata["stdout_artifact_id"]

    assert not result.is_error
    assert data["truncated"] is True
    assert data["stdout_artifact_id"] == artifact_id
    assert data["stderr_artifact_id"] is None
    assert (home / "sessions" / "sess_test" / "artifacts" / artifact_id / "stdout.txt").read_text(encoding="utf-8").strip() == "x" * 40


@pytest.mark.asyncio
async def test_run_command_truncated_stderr_sets_result_artifact_id(isolated_dirs) -> None:
    home, project = isolated_dirs

    async def callback(_request):
        return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)

    context = await make_context(home, project, callback)
    context.config.tools.stderr_limit_bytes = 8
    registry = ToolRegistry()
    register_builtin_code_tools(registry)
    result = await registry.execute(
        ToolCall(
            tool_call_id="call1",
            name="code.run_command",
            arguments={"argv": [sys.executable, "-c", "import sys; print('e' * 40, file=sys.stderr)"]},
        ),
        context,
    )
    data = result.content[0].data  # type: ignore[union-attr]
    artifact_id = result.metadata["stderr_artifact_id"]

    assert not result.is_error
    assert data["truncated"] is True
    assert data["stdout_artifact_id"] is None
    assert data["stderr_artifact_id"] == artifact_id
    assert (home / "sessions" / "sess_test" / "artifacts" / artifact_id / "stderr.txt").read_text(encoding="utf-8").strip() == "e" * 40
