from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
import sqlite3

import pytest

from agent_core import AgentRuntime
from agent_core.storage.migrations import SCHEMA_VERSION
from agent_core.types.content import ArtifactRefBlock
from agent_core.types.permissions import PermissionDecision, PermissionDecisionKind
from agent_core.types.tools import ToolCall, ToolDefinition, ToolResult
from tests.conftest import write_config
from tests.fixtures.scripted_ollama import ScriptedOllama


def _write_ollama_config(home, scripted_ollama: ScriptedOllama, **kwargs):
    return write_config(home, base_url=scripted_ollama.base_url, **kwargs)


def _runtime(project, scripted_ollama: ScriptedOllama, **kwargs) -> AgentRuntime:
    return AgentRuntime(project_dir=project, provider_registry=scripted_ollama.provider_registry(), **kwargs)


def _command_call() -> ToolCall:
    return ToolCall(
        tool_call_id="call1",
        name="code.run_command",
        arguments={"argv": ["python3", "-c", "print('x' * 200)"]},
    )


async def _allow(_request):
    return PermissionDecision(decision=PermissionDecisionKind.ALLOW_ONCE)


@pytest.mark.asyncio
async def test_replay_and_get_node_path(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_replay")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_replay")
        run_replay = await runtime.replay_run(handle.run_id)
        assert replay.nodes
        assert replay.model_requests
        assert replay.model_requests[0]["model"] == "gemma4"
        assert "code.read_file" in replay.model_requests[0]["tool_names"]
        assert run_replay.model_requests
        path = await runtime.get_node_path(replay.nodes[-1].node_id)
        assert path[0].role == "user"
        assert path[-1].role == "assistant"


@pytest.mark.asyncio
async def test_migration_records_schema_version_and_session_tables(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_schema")
        _events = [event async for event in handle.events()]
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT version, applied_at FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone()
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    assert version is not None
    assert version[0] == SCHEMA_VERSION
    assert version[1]
    assert {"sessions", "agents", "runs", "artifacts", "nodes_sess_schema", "events_sess_schema"} <= tables


@pytest.mark.asyncio
async def test_run_end_reasons_stay_within_stable_contract(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("ok")
    scripted_ollama.enqueue_failure_after_delta("partial")
    async with _runtime(project, scripted_ollama) as runtime:
        ok = await runtime.start("ok", session_id="sess_end_reason_ok")
        _ok_events = [event async for event in ok.events()]
        failed = await runtime.start("fail", session_id="sess_end_reason_failed")
        _failed_events = [event async for event in failed.events()]
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    allowed = {
        "completed",
        "max_turns",
        "aborted_streaming",
        "aborted_tools",
        "prompt_too_long",
        "max_output_tokens_recovery",
        "failed",
    }
    with sqlite3.connect(db_path) as conn:
        reasons = [row[0] for row in conn.execute("SELECT end_reason FROM runs WHERE end_reason IS NOT NULL").fetchall()]

    assert reasons
    assert set(reasons) <= allowed


@pytest.mark.asyncio
async def test_root_agent_rows_are_distinct_across_sessions(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("one")
    scripted_ollama.enqueue_text("two")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first", session_id="sess_root_one")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second", session_id="sess_root_two")
        _second_events = [event async for event in second.events()]
        assert runtime.paths is not None
        db_path = runtime.paths.session_db_path

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, root_agent_id FROM sessions WHERE session_id IN (?, ?) ORDER BY session_id",
            ("sess_root_one", "sess_root_two"),
        ).fetchall()
        agents = conn.execute(
            "SELECT agent_id, session_id, agent_type FROM agents WHERE session_id IN (?, ?) ORDER BY session_id",
            ("sess_root_one", "sess_root_two"),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][1] != rows[1][1]
    assert len(agents) == 2
    assert {row[2] for row in agents} == {"main"}


@pytest.mark.asyncio
async def test_replay_session_filters_events_by_seq_bounds(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_replay_bounds")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_replay_bounds")
        seqs = [event.seq for event in replay.events if event.seq is not None]
        assert len(seqs) >= 3
        middle = seqs[len(seqs) // 2]
        bounded = await runtime.replay_session("sess_replay_bounds", from_seq=middle, to_seq=middle)

    assert [event.seq for event in bounded.events] == [middle]
    assert bounded.nodes == replay.nodes


@pytest.mark.asyncio
async def test_replay_run_is_limited_to_single_run(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("first")
    scripted_ollama.enqueue_text("second")
    async with _runtime(project, scripted_ollama) as runtime:
        first = await runtime.start("first prompt", session_id="sess_replay_run")
        _first_events = [event async for event in first.events()]
        second = await runtime.start("second prompt", session_id="sess_replay_run")
        _second_events = [event async for event in second.events()]

        replay = await runtime.replay_run(second.run_id)

    assert replay.run_id == second.run_id
    assert replay.session_id == "sess_replay_run"
    assert replay.nodes
    assert {node.run_id for node in replay.nodes} == {second.run_id}
    assert {event.run_id for event in replay.events} == {second.run_id}
    replay_text = "\n".join(
        getattr(block, "text", "")
        for node in replay.nodes
        for block in node.content
        if getattr(block, "type", None) == "text"
    )
    assert "second prompt" in replay_text
    assert "second" in replay_text
    assert "first prompt" not in replay_text
    assert "first" not in replay_text


@pytest.mark.asyncio
async def test_replay_redacts_sensitive_values_by_default(isolated_dirs, monkeypatch, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    monkeypatch.setenv("SOONG_TEST_SECRET_TOKEN", "secret-token-value")
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("secret-token-value")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("please echo secret-token-value", session_id="sess_redact")
        _events = [event async for event in handle.events()]
        redacted = await runtime.replay_session("sess_redact")
        raw = await runtime.replay_session("sess_redact", include_sensitive=True)

    redacted_text = "\n".join(
        getattr(block, "text", "")
        for node in redacted.nodes
        for block in node.content
        if getattr(block, "type", None) == "text"
    )
    raw_text = "\n".join(
        getattr(block, "text", "")
        for node in raw.nodes
        for block in node.content
        if getattr(block, "type", None) == "text"
    )
    assert "secret-token-value" not in redacted_text
    assert "[REDACTED]" in redacted_text
    assert "secret-token-value" in raw_text


@pytest.mark.asyncio
async def test_debug_mode_persists_raw_provider_artifact_without_polluting_replay_nodes(
    isolated_dirs, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")

    async with _runtime(project, scripted_ollama, debug=True) as runtime:
        handle = await runtime.start("hi", session_id="sess_provider_debug")
        events = [event async for event in handle.events(debug=True)]
        replay = await runtime.replay_session("sess_provider_debug")
        cleanup = await runtime.cleanup_artifacts(session_id="sess_provider_debug", dry_run=True)

    debug_artifacts = [
        artifact
        for artifact in replay.artifacts
        if json.loads(artifact.get("metadata_json") or "{}").get("raw") is True
    ]
    assert any(event.event_type == "provider_debug_artifact_created" for event in events)
    assert debug_artifacts
    artifact = debug_artifacts[0]
    body = __import__("pathlib").Path(artifact["path"]).read_text(encoding="utf-8")
    assert '"request"' in body
    assert '"response"' in body
    assert "raw_debug" not in json.dumps([node.model_dump(mode="json") for node in replay.nodes])
    assert "raw_debug" not in json.dumps([event.model_dump(mode="json") for event in replay.events])
    assert any(candidate["artifact_id"] == artifact["artifact_id"] for candidate in cleanup.candidates)


@pytest.mark.asyncio
async def test_replay_redacts_sensitive_field_names_by_default(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    secret = "field-secret-value"
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="call1", name="test.sensitive_payload", arguments={"api_key": secret})]
    )
    scripted_ollama.enqueue_text("done")

    async with _runtime(project, scripted_ollama) as runtime:
        async def handler(context, args):
            artifact = context.artifact_manager.write_text(
                session_id=context.session_id,
                text=f"api_key={args['api_key']}",
                filename="secret.txt",
                summary="sensitive artifact",
            )
            return ToolResult(
                tool_call_id="ollama_tool_0",
                tool_name="test.sensitive_payload",
                content=[
                    ArtifactRefBlock(
                        artifact_id=artifact.artifact_id,
                        summary=artifact.summary,
                        mime_type=artifact.mime_type,
                    )
                ],
                metadata={"api_key": args["api_key"]},
            )

        runtime.register_tool(
            ToolDefinition(
                name="test.sensitive_payload",
                description="Return sensitive metadata.",
                input_schema={
                    "type": "object",
                    "properties": {"api_key": {"type": "string"}},
                    "required": ["api_key"],
                },
                permission="readonly",
                tags={"readonly"},
            ),
            handler,
        )
        handle = await runtime.start("use the sensitive tool", session_id="sess_field_redact")
        _events = [event async for event in handle.events()]
        redacted = await runtime.replay_session("sess_field_redact")
        raw = await runtime.replay_session("sess_field_redact", include_sensitive=True)

    def render(replay) -> str:
        return "\n".join(
            [
                str(node.model_dump(mode="json"))
                for node in replay.nodes
            ]
            + [str(event.model_dump(mode="json")) for event in replay.events]
            + [str(artifact) for artifact in replay.artifacts]
        )

    redacted_text = render(redacted)
    raw_text = render(raw)
    assert secret not in redacted_text
    assert "[REDACTED]" in redacted_text
    assert secret in raw_text
    assert "field-secret-value" not in redacted.artifacts[0]["path"]
    assert "field-secret-value" not in __import__("pathlib").Path(redacted.artifacts[0]["path"]).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_delete_session_after_completed(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_text("hello")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("hi", session_id="sess_delete")
        _events = [event async for event in handle.events()]
        result = await runtime.delete_session("sess_delete")
        assert result.deleted is True
        replay = await runtime.replay_session("sess_delete")
        assert replay.nodes == []


@pytest.mark.asyncio
async def test_delete_session_removes_artifact_files(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_delete_art")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_delete_art")
        artifact_path = __import__("pathlib").Path(replay.artifacts[0]["path"])
        assert artifact_path.exists()
        result = await runtime.delete_session("sess_delete_art")
        assert result.deleted is True
    assert not artifact_path.exists()


@pytest.mark.asyncio
async def test_delete_session_artifact_delete_failure_returns_error_and_keeps_session(
    isolated_dirs, monkeypatch, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    original_unlink = os.unlink

    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_delete_art_fail")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_delete_art_fail")
        artifact = replay.artifacts[0]
        artifact_path = __import__("pathlib").Path(artifact["path"])

        def fail_unlink(path, *args, **kwargs):
            if str(path) == str(artifact_path):
                raise OSError("permission denied")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", fail_unlink)
        result = await runtime.delete_session("sess_delete_art_fail")
        after = await runtime.replay_session("sess_delete_art_fail")

    assert result.deleted is False
    assert result.error is not None
    assert result.error.code.value == "storage_error"
    assert result.error.details["artifacts"][0]["artifact_id"] == artifact["artifact_id"]
    assert after.nodes
    assert after.artifacts and after.artifacts[0]["artifact_id"] == artifact["artifact_id"]
    assert artifact_path.exists()


@pytest.mark.asyncio
async def test_cleanup_project_tasks_dry_run(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    task_dir = project / ".soong-agent" / "tasks" / "sess"
    task_dir.mkdir(parents=True)
    wal = task_dir / "done.wal.jsonl"
    wal.write_text('{"event_type":"task_completed"}\n', encoding="utf-8")
    async with _runtime(project, scripted_ollama) as runtime:
        result = await runtime.cleanup_project_tasks(project, dry_run=True)
    assert result.candidates and result.candidates[0]["path"] == str(wal)
    assert wal.exists()


@pytest.mark.asyncio
async def test_cleanup_project_tasks_honors_older_than(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    task_dir = project / ".soong-agent" / "tasks" / "sess"
    task_dir.mkdir(parents=True)
    wal = task_dir / "done.wal.jsonl"
    wal.write_text('{"event_type":"task_completed"}\n', encoding="utf-8")
    old_timestamp = (datetime.now(UTC) - timedelta(days=3)).timestamp()
    os.utime(wal, (old_timestamp, old_timestamp))
    async with _runtime(project, scripted_ollama) as runtime:
        old = await runtime.cleanup_project_tasks(project, dry_run=True, older_than=datetime.now(UTC) - timedelta(days=1))
        new = await runtime.cleanup_project_tasks(project, dry_run=True, older_than=datetime.now(UTC) - timedelta(days=5))

    assert old.candidates and old.candidates[0]["path"] == str(wal)
    assert old.candidates[0]["modified_at"]
    assert new.candidates == []
    assert wal.exists()


@pytest.mark.asyncio
async def test_cleanup_project_tasks_include_terminal_kinds_and_delete(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    task_dir = project / ".soong-agent" / "tasks" / "sess"
    task_dir.mkdir(parents=True)
    completed = task_dir / "completed.wal.jsonl"
    failed = task_dir / "failed.wal.jsonl"
    cancelled = task_dir / "cancelled.wal.jsonl"
    active = task_dir / "active.wal.jsonl"
    completed.write_text('{"event_type":"task_completed"}\n', encoding="utf-8")
    failed.write_text('{"event_type":"task_failed"}\n', encoding="utf-8")
    cancelled.write_text('{"event_type":"task_cancelled"}\n', encoding="utf-8")
    active.write_text('{"event_type":"task_step_completed"}\n', encoding="utf-8")

    async with _runtime(project, scripted_ollama) as runtime:
        default = await runtime.cleanup_project_tasks(project, dry_run=True)
        expanded = await runtime.cleanup_project_tasks(
            project,
            dry_run=True,
            include_failed=True,
            include_cancelled=True,
        )
        deleted = await runtime.cleanup_project_tasks(
            project,
            dry_run=False,
            include_failed=True,
            include_cancelled=True,
        )

    assert {candidate["path"] for candidate in default.candidates} == {str(completed)}
    assert {candidate["path"] for candidate in expanded.candidates} == {str(completed), str(failed), str(cancelled)}
    assert set(deleted.deleted) == {str(completed), str(failed), str(cancelled)}
    assert not completed.exists()
    assert not failed.exists()
    assert not cancelled.exists()
    assert active.exists()


@pytest.mark.asyncio
async def test_cleanup_project_tasks_delete_failure_returns_partial_error(
    isolated_dirs, monkeypatch, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    task_dir = project / ".soong-agent" / "tasks" / "sess"
    task_dir.mkdir(parents=True)
    wal = task_dir / "completed.wal.jsonl"
    wal.write_text('{"event_type":"task_completed"}\n', encoding="utf-8")
    original_unlink = os.unlink

    def fail_unlink(path, *args, **kwargs):
        if str(path).endswith("completed.wal.jsonl"):
            raise OSError("permission denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_unlink)

    async with _runtime(project, scripted_ollama) as runtime:
        result = await runtime.cleanup_project_tasks(project, dry_run=False)

    assert result.deleted == []
    assert result.errors
    assert result.errors[0].code.value == "storage_error"
    assert "completed.wal.jsonl" in result.errors[0].message
    assert wal.exists()


@pytest.mark.asyncio
async def test_command_large_output_artifact_cleanup(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_art")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_art")
        assert replay.artifacts
        dry = await runtime.cleanup_artifacts(session_id="sess_art", dry_run=True)
        assert dry.candidates == []
        all_artifacts = await runtime.cleanup_artifacts(session_id="sess_art", dry_run=True, include_all=True)
        assert all_artifacts.candidates
        path = all_artifacts.candidates[0]["path"]
        delete = await runtime.cleanup_artifacts(session_id="sess_art", dry_run=False, include_all=True)
        assert delete.deleted
    assert not __import__("pathlib").Path(path).exists()


@pytest.mark.asyncio
async def test_delete_artifact_removes_file_and_registry(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_delete_artifact")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_delete_artifact")
        artifact = replay.artifacts[0]
        artifact_path = __import__("pathlib").Path(artifact["path"])
        artifact_dir = artifact_path.parent

        result = await runtime.delete_artifact(artifact["artifact_id"])
        after = await runtime.replay_session("sess_delete_artifact")

    assert result.deleted == [artifact["artifact_id"]]
    assert after.artifacts == []
    assert not artifact_path.exists()
    assert not artifact_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_artifacts_delete_failure_returns_partial_error_and_keeps_registry(
    isolated_dirs, monkeypatch, scripted_ollama: ScriptedOllama
) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    original_unlink = os.unlink

    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_art_delete_fail")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_art_delete_fail")
        artifact = replay.artifacts[0]
        artifact_path = __import__("pathlib").Path(artifact["path"])

        def fail_unlink(path, *args, **kwargs):
            if str(path) == str(artifact_path):
                raise OSError("permission denied")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", fail_unlink)
        result = await runtime.cleanup_artifacts(session_id="sess_art_delete_fail", dry_run=False, include_all=True)
        after = await runtime.replay_session("sess_art_delete_fail")

    assert result.deleted == []
    assert result.errors
    assert result.errors[0].code.value == "storage_error"
    assert artifact["artifact_id"] in result.errors[0].message
    assert after.artifacts and after.artifacts[0]["artifact_id"] == artifact["artifact_id"]
    assert artifact_path.exists()


@pytest.mark.asyncio
async def test_cleanup_artifacts_honors_max_bytes_and_removes_registry(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_art_max")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_art_max")
        artifact = replay.artifacts[0]
        artifact_path = __import__("pathlib").Path(artifact["path"])
        artifact_dir = artifact_path.parent
        dry = await runtime.cleanup_artifacts(session_id="sess_art_max", dry_run=True, include_all=True, max_bytes=artifact["size_bytes"] + 1)
        assert dry.candidates == []
        delete = await runtime.cleanup_artifacts(session_id="sess_art_max", dry_run=False, include_all=True, max_bytes=1)
        assert delete.deleted == [artifact["artifact_id"]]
        after = await runtime.replay_session("sess_art_max")

    assert after.artifacts == []
    assert not artifact_path.exists()
    assert not artifact_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_artifacts_honors_older_than(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([_command_call()])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama, permission_callback=_allow) as runtime:
        handle = await runtime.start("run command", session_id="sess_art_time")
        _events = [event async for event in handle.events()]
        future = datetime.now(UTC) + timedelta(days=1)
        past = datetime.now(UTC) - timedelta(days=1)
        old = await runtime.cleanup_artifacts(session_id="sess_art_time", dry_run=True, include_all=True, older_than=future)
        new = await runtime.cleanup_artifacts(session_id="sess_art_time", dry_run=True, include_all=True, older_than=past)

    assert old.candidates
    assert new.candidates == []


@pytest.mark.asyncio
async def test_large_list_dir_output_artifact_is_registered(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    config_path = _write_ollama_config(home, scripted_ollama)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("stdout_limit_bytes = 64", "stdout_limit_bytes = 80"),
        encoding="utf-8",
    )
    for index in range(20):
        (project / f"file_{index}.txt").write_text("x", encoding="utf-8")
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="call1", name="code.list_dir", arguments={"path": str(project)})])
    scripted_ollama.enqueue_text("done")
    async with _runtime(project, scripted_ollama) as runtime:
        handle = await runtime.start("list files", session_id="sess_list_art")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_list_art")

    assert replay.artifacts
    assert replay.artifacts[0]["mime_type"] == "application/json"
    assert __import__("pathlib").Path(replay.artifacts[0]["path"]).exists()


@pytest.mark.asyncio
async def test_tool_result_artifact_ref_block_is_registered_without_metadata(isolated_dirs, scripted_ollama: ScriptedOllama) -> None:
    home, project = isolated_dirs
    _write_ollama_config(home, scripted_ollama)
    scripted_ollama.enqueue_tool_calls([ToolCall(tool_call_id="call1", name="test.artifact_ref", arguments={})])
    scripted_ollama.enqueue_text("done")

    async with _runtime(project, scripted_ollama) as runtime:
        async def handler(context, _args):
            artifact = context.artifact_manager.write_text(
                session_id=context.session_id,
                text="artifact body",
                filename="custom.txt",
                summary="custom artifact",
            )
            return ToolResult(
                tool_call_id="ollama_tool_0",
                tool_name="test.artifact_ref",
                content=[
                    ArtifactRefBlock(
                        artifact_id=artifact.artifact_id,
                        summary=artifact.summary,
                        mime_type=artifact.mime_type,
                    )
                ],
            )

        runtime.register_tool(
            ToolDefinition(
                name="test.artifact_ref",
                description="Return only an artifact reference block.",
                input_schema={"type": "object", "properties": {}, "required": []},
                permission="readonly",
                tags={"readonly"},
            ),
            handler,
        )
        handle = await runtime.start("make artifact", session_id="sess_art_ref")
        _events = [event async for event in handle.events()]
        replay = await runtime.replay_session("sess_art_ref")

    assert replay.artifacts
    assert replay.artifacts[0]["summary"] == "custom artifact"
    assert replay.artifacts[0]["filename"] == "custom.txt"
