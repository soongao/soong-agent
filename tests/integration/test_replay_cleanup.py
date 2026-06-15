from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os

import pytest

from agent_core import AgentRuntime
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
