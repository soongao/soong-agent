from __future__ import annotations

import asyncio
import json
import threading
import time
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent_core.events import make_event
from agent_core.types.permissions import PermissionDecisionKind, PermissionRequest
from agent_core.types.tools import ToolCall
from agent_hub.backend.database import HubDatabase
from agent_hub.backend.errors import HubApiError
from agent_hub.backend.events import HubEventHub, sse_encode
from agent_hub.backend.permissions import PermissionBridge
from agent_hub.backend.app import create_app
from agent_hub.backend.runtime import HubRuntimeBridge
from agent_hub.backend.workers.pty import PtySessionKey
from tests.conftest import write_config


def _allow_pending_permissions(client: TestClient, hub_db_path) -> None:
    conn = sqlite3.connect(hub_db_path)
    try:
        rows = conn.execute(
            """
            SELECT permission_request_id
            FROM permission_requests
            WHERE status='pending'
            ORDER BY created_at ASC
            """
        ).fetchall()
    finally:
        conn.close()
    for (permission_request_id,) in rows:
        response = client.post(
            f"/permissions/{permission_request_id}/decision",
            json={"decision": "allow_once"},
        )
        assert response.status_code == 200


def test_agent_hub_health_bootstraps_config_and_db(isolated_dirs) -> None:
    home, project = isolated_dirs
    (project / "CLAUDE.md").write_text("project instructions\n", encoding="utf-8")
    skills = home / "skills" / "brainstorming"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: brainstorming\ndescription: Think first\n---\nbody\n", encoding="utf-8")
    app = create_app(home_dir=home, project_dir=project)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["status"] == "ready"
        assert data["core_started"] is True
        assert data["config_path"] == str(home / "config.toml")
        assert data["hub_db_path"] == str(home / "hub" / "hub.db")
        assert data["context"]["auto_instruction_paths"] == [str((project / "CLAUDE.md").resolve())]
        assert data["context"]["skill_count"] == 1
        assert data["context"]["skills"][0]["name"] == "brainstorming"
    assert (home / "config.toml").exists()
    assert (home / "hub" / "hub.db").exists()


def test_agent_hub_health_reports_config_error_without_crashing(isolated_dirs) -> None:
    home, project = isolated_dirs
    (home / "config.toml").write_text("[model]\nprovider = 123\n", encoding="utf-8")

    app = create_app(home_dir=home, project_dir=project)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        data = health.json()
        assert data["ok"] is False
        assert data["status"] == "core_failed"
        assert data["error"]["code"] == "config_invalid"
        assert data["core_started"] is False

        unavailable = client.post("/conversations", json={"title": "Should fail"})
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["code"] == "config_invalid"


def test_agent_hub_allows_local_frontend_cors(isolated_dirs) -> None:
    home, project = isolated_dirs
    app = create_app(home_dir=home, project_dir=project)
    with TestClient(app) as client:
        response = client.options(
            "/conversations",
            headers={
                "origin": "http://127.0.0.1:5173",
                "access-control-request-method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_agent_hub_conversation_message_and_worker_routes(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    config_path = home / "config.toml"
    config_path.write_text(
        f"""
[runtime]
cancel_timeout_ms = 1000
max_turns = 8

[model]
provider = "ollama"
base_url = "{scripted_ollama.base_url}"
api_key_env = ""
name = "gemma4"
context_window = 8192
max_output_tokens = 1024
temperature = 0.0
timeout_ms = 1000

[context]
session_db_path = "${{SOONG_AGENT_HOME}}/sessions.sqlite"

[permissions]
readonly_default = "allow"
write_without_callback = "deny"
remember_scope = "session"
allow_for_session_enabled = true

[tools]
declarative_enabled = true
disabled = []
allowed_write_roots = []
allow_tmp_write = false
default_timeout_ms = 1000
max_timeout_ms = 2000
env_allowlist = ["PATH", "HOME", "TMPDIR"]
stdout_limit_bytes = 64
stderr_limit_bytes = 64
sensitive_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "*.pem", "*.key", ".env", ".env.*"]

[[agents.worker_pools]]
pool_id = "default"

[[agents.worker_pools.workers]]
worker_id = "worker_general_1"
agent_definition_id = "default_worker_agent"
allowed_tools = ["agent.task_get", "agent.task_query_steps", "agent.task_claim_step", "agent.task_update_step", "code.read_file"]
""".strip(),
        encoding="utf-8",
    )
    scripted_ollama.enqueue_text("hub response")
    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        created = client.post("/conversations", json={"title": "New conversation"})
        assert created.status_code == 200
        conversation = created.json()
        assert conversation["conversation_id"].startswith("conv_")
        assert client.get("/config/status").json()["model"] == "gemma4"
        fetched_conversation = client.get(f"/conversations/{conversation['conversation_id']}")
        assert fetched_conversation.status_code == 200
        assert fetched_conversation.json()["conversation_id"] == conversation["conversation_id"]
        listed = client.get("/conversations")
        assert listed.status_code == 200
        assert listed.json()["conversations"][0]["conversation_id"] == conversation["conversation_id"]

        worker_create = client.post(
            "/workers",
            json={"worker_id": "hub_worker", "name": "Hub Worker", "system_prompt": "Hub worker prompt."},
        )
        assert worker_create.status_code == 200
        fetched_worker = client.get("/workers/hub_worker")
        assert fetched_worker.status_code == 200
        assert fetched_worker.json()["worker_id"] == "hub_worker"
        assert fetched_worker.json()["queue_length"] == 0
        workers = client.get("/workers").json()["workers"]
        assert any(worker["worker_id"] == "hub_worker" for worker in workers)
        tools = client.get("/tools")
        assert tools.status_code == 200
        assert any(tool["name"] == "code.read_file" for tool in tools.json()["tools"])

        sent = client.post(f"/conversations/{conversation['conversation_id']}/messages", json={"text": "hello hub"})
        assert sent.status_code == 200
        assert sent.json()["status"] == "running"

        for _ in range(50):
            messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
            if any(message["sender_type"] == "orchestrator" and message["status"] == "completed" for message in messages):
                break
            time.sleep(0.02)
        messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
        assert [message["sender_type"] for message in messages] == ["user", "orchestrator"]
        assert messages[-1]["display_text"] == "hub response"
        assert messages[-1]["status"] == "completed"
        assert messages[0]["core_node_id"]

        skills = home / "skills" / "brainstorming"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("---\nname: brainstorming\ndescription: Think first\n---\nSkill body\n", encoding="utf-8")
        skill_load = client.post(
            f"/conversations/{conversation['conversation_id']}/skills/brainstorming/load",
            json={"name": "brainstorming"},
        )
        assert skill_load.status_code == 200
        assert skill_load.json()["loaded"] is True
        assert skill_load.json()["node_id"].startswith("node_")
        assert skill_load.json()["path"].endswith("skills/brainstorming/SKILL.md")

        nodes = client.get(f"/conversations/{conversation['conversation_id']}/branchable-nodes")
        assert nodes.status_code == 200
        assert nodes.json()["nodes"][0]["core_node_id"] == messages[0]["core_node_id"]

        branch = client.post(
            f"/conversations/{conversation['conversation_id']}/branch",
            json={"core_node_id": messages[0]["core_node_id"]},
        )
        assert branch.status_code == 200
        assert branch.json()["switched"] is True

        fork = client.post(
            f"/conversations/{conversation['conversation_id']}/fork",
            json={"core_node_id": messages[0]["core_node_id"], "title": "Forked"},
        )
        assert fork.status_code == 200
        assert fork.json()["conversation_id"].startswith("conv_")

        deleted_conversation = client.delete(f"/conversations/{conversation['conversation_id']}")
        assert deleted_conversation.status_code == 200
        assert deleted_conversation.json()["status"] == "deleted"
        assert client.get(f"/conversations/{conversation['conversation_id']}").status_code == 404
        assert all(
            item["conversation_id"] != conversation["conversation_id"]
            for item in client.get("/conversations").json()["conversations"]
        )

        disabled = client.post("/workers/hub_worker/disable")
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        enabled = client.post("/workers/hub_worker/enable")
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True
        deleted = client.delete("/workers/hub_worker")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_at"] is not None
        default_workers = client.get("/workers").json()["workers"]
        assert all(worker["worker_id"] != "hub_worker" for worker in default_workers)
        deleted_workers = client.get("/workers", params={"include_deleted": True}).json()["workers"]
        deleted_worker = next(worker for worker in deleted_workers if worker["worker_id"] == "hub_worker")
        assert deleted_worker["status"] == "deleted"



@pytest.mark.asyncio
async def test_hub_database_external_worker_session_mapping(isolated_dirs) -> None:
    home, _project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    try:
        stored = await db.upsert_external_worker_session(
            core_session_id="sess_external",
            worker_id="opencode_worker",
            executor_type="opencode",
            external_session_id="oc_session_1",
            metadata={"cwd": "/tmp/project"},
        )
        assert stored["external_session_id"] == "oc_session_1"
        assert stored["metadata"] == {"cwd": "/tmp/project"}
        fetched = await db.get_external_worker_session(
            core_session_id="sess_external",
            worker_id="opencode_worker",
            executor_type="opencode",
        )
        assert fetched is not None
        assert fetched["external_session_id"] == "oc_session_1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_permission_bridge_external_permission_maps_decisions(isolated_dirs) -> None:
    home, _project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = PermissionBridge(db, events)
    try:
        conversation = await db.create_conversation(core_session_id="sess_external_perm")
        bridge.bind_session(core_session_id=conversation.core_session_id, conversation_id=conversation.conversation_id)
        task = asyncio.create_task(
            bridge.external_permission_callback(
                core_session_id=conversation.core_session_id,
                core_run_id="run_worker",
                tool_name="opencode.Edit",
                permission="write",
                target_scope="file.txt",
                args_summary="edit file.txt",
                metadata={"source": "opencode_acp"},
            )
        )
        for _ in range(50):
            row = await (await db.conn.execute("SELECT permission_request_id, metadata_json FROM permission_requests WHERE status='pending'")).fetchone()
            if row is not None:
                break
            await asyncio.sleep(0.01)
        assert row is not None
        assert json.loads(row["metadata_json"])["source"] == "opencode_acp"
        resolved = await bridge.decide(row["permission_request_id"], "allow_for_session")
        assert resolved["status"] == "allowed"
        assert await task == "allow_always"
    finally:
        await bridge.shutdown()
        await db.close()

def test_agent_hub_seeds_editable_default_workers_once(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        workers = client.get("/workers").json()["workers"]
        seeded = {worker["worker_id"]: worker for worker in workers}
        assert {"code_reviewer", "doc_writer", "test_writer", "opencode_worker", "codex_pty_worker"} <= set(seeded)
        assert seeded["code_reviewer"]["source"] == "dynamic"
        assert seeded["code_reviewer"]["metadata"]["agenthub_default_worker"] is True
        assert seeded["doc_writer"]["allowed_tools"] == ["code.read_file", "code.list_dir", "code.search", "code.write_file", "code.edit_file"]
        assert seeded["test_writer"]["enabled"] is True
        assert seeded["opencode_worker"]["metadata"]["worker_executor"]["type"] == "opencode"
        assert seeded["opencode_worker"]["allowed_tools"] == ["opencode.acp"]
        assert seeded["codex_pty_worker"]["metadata"]["worker_executor"]["type"] == "codex_pty"
        assert seeded["codex_pty_worker"]["allowed_tools"] == ["codex.pty"]

        renamed = client.patch(
            "/workers/code_reviewer",
            json={"name": "Custom Reviewer", "system_prompt": seeded["code_reviewer"]["system_prompt"]},
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "Custom Reviewer"
        deleted = client.delete("/workers/doc_writer")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_at"] is not None

    restarted = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(restarted) as client:
        workers = client.get("/workers").json()["workers"]
        by_id = {worker["worker_id"]: worker for worker in workers}
        assert by_id["code_reviewer"]["name"] == "Custom Reviewer"
        assert "doc_writer" not in by_id
        deleted_workers = client.get("/workers", params={"include_deleted": True}).json()["workers"]
        deleted_by_id = {worker["worker_id"]: worker for worker in deleted_workers}
        assert deleted_by_id["doc_writer"]["status"] == "deleted"


@pytest.mark.asyncio
async def test_hub_runtime_routes_worker_reply_to_active_pty_session(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = HubRuntimeBridge(
        db=db,
        events=events,
        permission_bridge=PermissionBridge(db, events),
        project_dir=project,
        home_dir=home,
        provider_registry=scripted_ollama.provider_registry(),
    )
    try:
        await bridge.start()
        conversation = await bridge.create_conversation(title="PTY reply")
        await bridge.runtime.create_worker_config(
            {
                "worker_id": "pty_worker",
                "name": "PTY Worker",
                "system_prompt": "PTY worker.",
                "metadata": {"worker_executor": {"type": "codex_pty", "config": {}}},
            }
        )
        await db.add_conversation_worker(conversation.conversation_id, "pty_worker")

        class FakePtySession:
            def __init__(self) -> None:
                self.received: list[str] = []

            @property
            def running(self) -> bool:
                return True

            async def close(self) -> None:
                return None

            async def write_to_active(self, text: str):
                self.received.append(text)
                from agent_hub.backend.workers.pty import PtyInputReceipt

                return PtyInputReceipt(
                    core_session_id=conversation.core_session_id,
                    worker_id="pty_worker",
                    executor_type="codex_pty",
                    worker_run_id="run_worker_active",
                )

        fake = FakePtySession()
        bridge._pty_manager._sessions[
            PtySessionKey(conversation.core_session_id, "pty_worker", "codex_pty")
        ] = fake  # type: ignore[assignment]

        message, run_id, status = await bridge.send_message(conversation, "@pty_worker Y")
        messages = await db.list_messages(conversation.conversation_id)
    finally:
        await bridge.close()
        await db.close()

    assert fake.received == ["Y"]
    assert run_id == "run_worker_active"
    assert status == "completed"
    assert message.metadata["pty_reply"] is True
    assert [item.sender_type for item in messages] == ["user"]
    assert scripted_ollama.requests == []


def test_agent_hub_worker_api_redacts_direct_api_key(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)
    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        created = client.post(
            "/workers",
            json={
                "worker_id": "secret_worker",
                "name": "Secret Worker",
                "system_prompt": "Do secret work.",
                "model": {
                    "provider": "openai",
                    "name": "qwen2.5:7b",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "secret-key",
                },
            },
        )
        assert created.status_code == 200
        assert created.json()["model"]["api_key"] == "***"

        fetched = client.get("/workers/secret_worker")
        assert fetched.status_code == 200
        assert fetched.json()["model"]["api_key"] == "***"
        listed = client.get("/workers").json()["workers"]
        worker = next(worker for worker in listed if worker["worker_id"] == "secret_worker")
        assert worker["model"]["api_key"] == "***"

        updated = client.patch(
            "/workers/secret_worker",
            json={
                "name": "Renamed Secret Worker",
                "system_prompt": "Do renamed secret work.",
                "model": fetched.json()["model"],
            },
        )
        assert updated.status_code == 200
        assert updated.json()["model"]["api_key"] == "***"
        conn = sqlite3.connect(home / "sessions.sqlite")
        try:
            row = conn.execute("SELECT model_json FROM worker_configs_dynamic WHERE worker_id='secret_worker'").fetchone()
        finally:
            conn.close()
        assert row is not None
        assert json.loads(row[0])["api_key"] == "secret-key"


def test_agent_hub_cancel_queued_run_route(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    (home / "config.toml").write_text(
        f"""
[runtime]
cancel_timeout_ms = 1000
max_turns = 8

[model]
provider = "ollama"
base_url = "{scripted_ollama.base_url}"
api_key_env = ""
name = "gemma4"
context_window = 8192
max_output_tokens = 1024
temperature = 0.0
timeout_ms = 1000

[context]
session_db_path = "${{SOONG_AGENT_HOME}}/sessions.sqlite"

[permissions]
readonly_default = "allow"
write_without_callback = "deny"
remember_scope = "session"
allow_for_session_enabled = true

[tools]
declarative_enabled = true
disabled = []
allowed_write_roots = []
allow_tmp_write = false
default_timeout_ms = 1000
max_timeout_ms = 2000
env_allowlist = ["PATH", "HOME", "TMPDIR"]
stdout_limit_bytes = 64
stderr_limit_bytes = 64
sensitive_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "*.pem", "*.key", ".env", ".env.*"]

[[agents.worker_pools]]
pool_id = "default"

[[agents.worker_pools.workers]]
worker_id = "worker_general_1"
agent_definition_id = "default_worker_agent"
allowed_tools = ["agent.task_get", "agent.task_query_steps", "agent.task_claim_step", "agent.task_update_step", "code.read_file"]
""".strip(),
        encoding="utf-8",
    )
    release_first = threading.Event()

    async def wait_for_release() -> None:
        while not release_first.is_set():
            await __import__("asyncio").sleep(0.01)

    scripted_ollama.enqueue_text("first response", block=wait_for_release)
    scripted_ollama.enqueue_text("second response")
    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        conversation = client.post("/conversations", json={"title": "Cancel"}).json()
        first = client.post(f"/conversations/{conversation['conversation_id']}/messages", json={"text": "first"}).json()
        second = client.post(f"/conversations/{conversation['conversation_id']}/messages", json={"text": "second"}).json()
        assert second["status"] == "queued"
        cancelled = client.post(
            f"/conversations/{conversation['conversation_id']}/cancel",
            json={"core_run_id": second["core_run_id"]},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
        assert first["core_run_id"]
        release_first.set()
        for _ in range(50):
            messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
            if any(message["core_run_id"] == first["core_run_id"] and message["status"] == "completed" for message in messages):
                break
            time.sleep(0.02)
        messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
        assert any(message["core_run_id"] == second["core_run_id"] and message["status"] == "cancelled" for message in messages)


def test_agent_hub_cancel_active_run_route(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    (home / "config.toml").write_text(
        f"""
[runtime]
cancel_timeout_ms = 1000
max_turns = 8

[model]
provider = "ollama"
base_url = "{scripted_ollama.base_url}"
api_key_env = ""
name = "gemma4"
context_window = 8192
max_output_tokens = 1024
temperature = 0.0
timeout_ms = 1000

[context]
session_db_path = "${{SOONG_AGENT_HOME}}/sessions.sqlite"

[permissions]
readonly_default = "allow"
write_without_callback = "deny"
remember_scope = "session"
allow_for_session_enabled = true

[tools]
declarative_enabled = true
disabled = []
allowed_write_roots = []
allow_tmp_write = false
default_timeout_ms = 1000
max_timeout_ms = 2000
env_allowlist = ["PATH", "HOME", "TMPDIR"]
stdout_limit_bytes = 64
stderr_limit_bytes = 64
sensitive_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "*.pem", "*.key", ".env", ".env.*"]

[[agents.worker_pools]]
pool_id = "default"

[[agents.worker_pools.workers]]
worker_id = "worker_general_1"
agent_definition_id = "default_worker_agent"
allowed_tools = ["agent.task_get", "agent.task_query_steps", "agent.task_claim_step", "agent.task_update_step", "code.read_file"]
""".strip(),
        encoding="utf-8",
    )
    release = threading.Event()

    async def wait_for_release() -> None:
        while not release.is_set():
            await asyncio.sleep(0.01)

    scripted_ollama.enqueue_text("active response", block=wait_for_release)
    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        conversation = client.post("/conversations", json={"title": "Cancel active"}).json()
        sent = client.post(f"/conversations/{conversation['conversation_id']}/messages", json={"text": "cancel me"}).json()
        assert sent["status"] == "running"
        cancelled = client.post(
            f"/conversations/{conversation['conversation_id']}/cancel",
            json={"core_run_id": sent["core_run_id"]},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
        release.set()


def test_agent_hub_worker_mention_passes_orchestrator_directive(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    (home / "config.toml").write_text(
        f"""
[runtime]
cancel_timeout_ms = 1000
max_turns = 8

[model]
provider = "ollama"
base_url = "{scripted_ollama.base_url}"
api_key_env = ""
name = "gemma4"
context_window = 8192
max_output_tokens = 1024
temperature = 0.0
timeout_ms = 1000

[context]
session_db_path = "${{SOONG_AGENT_HOME}}/sessions.sqlite"

[permissions]
readonly_default = "allow"
write_without_callback = "deny"
remember_scope = "session"
allow_for_session_enabled = true

[tools]
declarative_enabled = true
disabled = []
allowed_write_roots = []
allow_tmp_write = false
default_timeout_ms = 1000
max_timeout_ms = 2000
env_allowlist = ["PATH", "HOME", "TMPDIR"]
stdout_limit_bytes = 64
stderr_limit_bytes = 64
sensitive_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gcloud", "*.pem", "*.key", ".env", ".env.*"]

[[agents.worker_pools]]
pool_id = "default"

[[agents.worker_pools.workers]]
worker_id = "worker_general_1"
agent_definition_id = "default_worker_agent"
allowed_tools = ["agent.task_get", "agent.task_query_steps", "agent.task_claim_step", "agent.task_update_step", "code.read_file"]
""".strip(),
        encoding="utf-8",
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="create",
                name="agent.task_create",
                arguments={
                    "task_id": "task_hub_worker",
                    "wal_name": "task_hub_worker.wal.jsonl",
                    "title": "Task",
                    "summary": "",
                    "steps": [{"step_id": "s1", "title": "Step"}],
                },
            )
        ]
    )
    scripted_ollama.enqueue_tool_calls(
        [
            ToolCall(
                tool_call_id="dispatch",
                name="agent.dispatch_worker",
                arguments={
                    "task_id": "task_hub_worker",
                    "instruction": "Do the work",
                    "allowed_step_ids": ["s1"],
                },
            )
        ]
    )
    scripted_ollama.enqueue_tool_calls(
        [ToolCall(tool_call_id="claim", name="agent.task_claim_step", arguments={"task_id": "task_hub_worker", "step_id": "s1"})]
    )
    scripted_ollama.enqueue_text("worker done")
    scripted_ollama.enqueue_text("orchestrator summary")

    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        conversation = client.post("/conversations", json={"title": "Mention"}).json()
        created = client.post(
            "/workers",
            json={"worker_id": "hub_worker", "name": "Hub Worker Name", "system_prompt": "Hub worker prompt."},
        )
        assert created.status_code == 200
        added = client.post(
            f"/conversations/{conversation['conversation_id']}/workers",
            json={"worker_id": "hub_worker"},
        )
        assert added.status_code == 200
        listed_workers = client.get(f"/conversations/{conversation['conversation_id']}/workers")
        assert listed_workers.status_code == 200
        assert [worker["worker_id"] for worker in listed_workers.json()["workers"]] == ["hub_worker"]
        sent = client.post(f"/conversations/{conversation['conversation_id']}/messages", json={"text": "@hub_worker inspect this"})
        assert sent.status_code == 200
        hub_db_path = home / "hub" / "hub.db"
        for _ in range(100):
            _allow_pending_permissions(client, hub_db_path)
            messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
            if any(message["sender_type"] == "orchestrator" and message["status"] == "completed" for message in messages):
                break
            time.sleep(0.02)
        messages = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
        worker_messages = [message for message in messages if message["sender_type"] == "worker" and message["worker_id"] == "hub_worker"]
        assert worker_messages
        assert worker_messages[0]["sender_name"] == "Hub Worker Name"
        assert worker_messages[0]["metadata"]["worker_snapshot"]["name"] == "Hub Worker Name"
        assert messages[-1]["display_text"] == "orchestrator summary"
        deleted = client.delete("/workers/hub_worker")
        assert deleted.status_code == 200
        messages_after_delete = client.get(f"/conversations/{conversation['conversation_id']}/messages").json()["messages"]
        worker_after_delete = next(message for message in messages_after_delete if message["sender_type"] == "worker")
        assert worker_after_delete["sender_name"] == "Hub Worker Name"
        assert worker_after_delete["metadata"]["worker_snapshot"]["worker_id"] == "hub_worker"
        system_text = "\n".join(block["content"] for block in scripted_ollama.requests[0]["messages"] if block["role"] == "system")
        assert "The user explicitly mentioned a worker for this run" in system_text
        assert "Worker id: hub_worker" in system_text


def test_agent_hub_worker_mention_errors_do_not_start_runs(isolated_dirs, scripted_ollama) -> None:
    home, project = isolated_dirs
    write_config(home, base_url=scripted_ollama.base_url)

    app = create_app(home_dir=home, project_dir=project, provider_registry=scripted_ollama.provider_registry())
    with TestClient(app) as client:
        conversation = client.post("/conversations", json={"title": "Mention errors"}).json()
        conversation_id = conversation["conversation_id"]

        missing = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@missing_worker inspect this"})
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "worker_not_found"

        empty_body = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@missing_worker"})
        assert empty_body.status_code == 400
        assert empty_body.json()["error"]["code"] == "validation_error"

        client.post("/workers", json={"worker_id": "global_only_worker", "name": "Global Only Worker", "system_prompt": "Global prompt."})
        not_added = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@global_only_worker inspect this"})
        assert not_added.status_code == 409
        assert not_added.json()["error"]["code"] == "worker_not_added"

        client.post("/workers", json={"worker_id": "disabled_worker", "name": "Disabled Worker", "system_prompt": "Disabled prompt."})
        client.post(f"/conversations/{conversation_id}/workers", json={"worker_id": "disabled_worker"})
        client.post("/workers/disabled_worker/disable")
        disabled = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@disabled_worker inspect this"})
        assert disabled.status_code == 409
        assert disabled.json()["error"]["code"] == "worker_disabled"

        client.post("/workers", json={"worker_id": "deleted_worker", "name": "Deleted Worker", "system_prompt": "Deleted prompt."})
        client.post(f"/conversations/{conversation_id}/workers", json={"worker_id": "deleted_worker"})
        client.delete("/workers/deleted_worker")
        deleted = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@deleted_worker inspect this"})
        assert deleted.status_code == 409
        assert deleted.json()["error"]["code"] == "worker_deleted"

        client.post("/workers", json={"worker_id": "ambiguous_one", "name": "ambiguous", "system_prompt": "First prompt."})
        client.post("/workers", json={"worker_id": "ambiguous_two", "name": "ambiguous", "system_prompt": "Second prompt."})
        client.post(f"/conversations/{conversation_id}/workers", json={"worker_id": "ambiguous_one"})
        client.post(f"/conversations/{conversation_id}/workers", json={"worker_id": "ambiguous_two"})
        ambiguous = client.post(f"/conversations/{conversation_id}/messages", json={"text": "@ambiguous inspect this"})
        assert ambiguous.status_code == 409
        assert ambiguous.json()["error"]["code"] == "worker_ambiguous"

        messages = client.get(f"/conversations/{conversation_id}/messages").json()["messages"]
        assert messages == []
        assert scripted_ollama.requests == []


@pytest.mark.asyncio
async def test_agent_hub_maps_queued_run_dequeue_to_running_status(isolated_dirs) -> None:
    home, project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = HubRuntimeBridge(db=db, events=events, permission_bridge=PermissionBridge(db, events), project_dir=project, home_dir=home)
    try:
        conversation = await db.create_conversation(core_session_id="sess_queued_mapping")
        orchestrator_message = await db.create_message(
            conversation_id=conversation.conversation_id,
            sender_type="orchestrator",
            sender_name="Orchestrator",
            status="queued",
            core_session_id=conversation.core_session_id,
            core_run_id="run_queued_mapping",
        )
        event = make_event(
            session_id=conversation.core_session_id,
            agent_id="agent_orchestrator",
            run_id="run_queued_mapping",
            event_type="run_dequeued",
        )
        await bridge._map_core_event(
            conversation_id=conversation.conversation_id,
            orchestrator_message_id=orchestrator_message.message_id,
            event=event,
            text_parts=[],
        )
        updated = await db.get_message(orchestrator_message.message_id)
        assert updated is not None
        assert updated.status == "running"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_agent_hub_maps_permission_failed_to_system_message(isolated_dirs) -> None:
    home, project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = HubRuntimeBridge(db=db, events=events, permission_bridge=PermissionBridge(db, events), project_dir=project, home_dir=home)
    try:
        conversation = await db.create_conversation(core_session_id="sess_permission_failed")
        orchestrator_message = await db.create_message(
            conversation_id=conversation.conversation_id,
            sender_type="orchestrator",
            sender_name="Orchestrator",
            status="running",
            core_session_id=conversation.core_session_id,
            core_run_id="run_permission_failed",
        )
        event = make_event(
            session_id=conversation.core_session_id,
            agent_id="agent_orchestrator",
            run_id="run_permission_failed",
            event_type="permission_failed",
            tool_call_id="call_perm",
            payload={
                "name": "code.write_file",
                "is_error": True,
                "error": {"code": "permission_denied", "message": "permission callback failed"},
            },
        )
        await bridge._map_core_event(
            conversation_id=conversation.conversation_id,
            orchestrator_message_id=orchestrator_message.message_id,
            event=event,
            text_parts=[],
        )
        messages = await db.list_messages(conversation.conversation_id)
        system_messages = [message for message in messages if message.sender_type == "system"]
        assert system_messages
        assert system_messages[-1].status == "failed"
        assert system_messages[-1].display_text == "code.write_file: permission callback failed"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_hub_database_message_crud(isolated_dirs) -> None:
    home, _project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    try:
        conversation = await db.create_conversation(core_session_id="sess_hub_db")
        message = await db.create_message(
            conversation_id=conversation.conversation_id,
            sender_type="user",
            sender_name="You",
            display_text="hello",
        )
        assert message.message_id.startswith("msg_")
        listed = await db.list_messages(conversation.conversation_id)
        assert listed[0].display_text == "hello"
        updated = await db.update_message(message.message_id, status="completed", display_text="done")
        assert updated.display_text == "done"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_hub_event_hub_subscribe_publish_and_sse_encode() -> None:
    events = HubEventHub()
    iterator = events.subscribe("conv_events")
    pending = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    event = await events.publish("message_created", conversation_id="conv_events", payload={"message_id": "msg_events"})
    received = await asyncio.wait_for(pending, timeout=1)
    assert received.id == event.id
    assert received.type == "message_created"
    encoded = sse_encode(received)
    assert f"id: {event.id}" in encoded
    assert "event: message_created" in encoded
    assert '"message_id": "msg_events"' in encoded
    await iterator.aclose()


@pytest.mark.asyncio
async def test_permission_bridge_waits_resolves_and_rejects_duplicate_decision(isolated_dirs) -> None:
    home, project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = PermissionBridge(db, events)
    try:
        conversation = await db.create_conversation(core_session_id="sess_perm_bridge")
        bridge.bind_session(core_session_id="sess_perm_bridge", conversation_id=conversation.conversation_id)
        request = PermissionRequest(
            request_id="perm_req_source",
            session_id="sess_perm_bridge",
            agent_id="agent_main",
            run_id="run_perm_bridge",
            agent_role="orchestrator",
            tool_name="code.write_file",
            permission="write",
            tags=["write"],
            args_summary="write hello.txt",
            target_scope="project",
            cwd=str(project),
        )
        decision_task = asyncio.create_task(bridge.permission_callback(request))
        for _ in range(20):
            rows = await (
                await db.conn.execute("SELECT permission_request_id FROM permission_requests WHERE status='pending'")
            ).fetchall()
            if rows:
                break
            await asyncio.sleep(0.01)
        assert len(rows) == 1
        permission_request_id = rows[0]["permission_request_id"]
        assert not decision_task.done()

        resolved = await bridge.decide(permission_request_id, "allow_for_session")
        assert resolved["status"] == "allowed"
        decision = await asyncio.wait_for(decision_task, timeout=1)
        assert decision.decision == PermissionDecisionKind.ALLOW_FOR_SESSION

        with pytest.raises(HubApiError) as duplicate:
            await bridge.decide(permission_request_id, "deny")
        assert duplicate.value.code == "permission_already_resolved"

        with pytest.raises(HubApiError) as missing:
            await bridge.decide("perm_missing", "deny")
        assert missing.value.code == "permission_not_found"
    finally:
        await bridge.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_permission_bridge_shutdown_cancels_pending_request(isolated_dirs) -> None:
    home, project = isolated_dirs
    db = HubDatabase(home / "hub" / "hub.db")
    await db.open()
    events = HubEventHub()
    bridge = PermissionBridge(db, events)
    try:
        conversation = await db.create_conversation(core_session_id="sess_perm_shutdown")
        bridge.bind_session(core_session_id="sess_perm_shutdown", conversation_id=conversation.conversation_id)
        request = PermissionRequest(
            request_id="perm_req_shutdown",
            session_id="sess_perm_shutdown",
            agent_id="agent_main",
            run_id="run_perm_shutdown",
            agent_role="orchestrator",
            tool_name="code.write_file",
            permission="write",
            tags=["write"],
            args_summary="write shutdown.txt",
            target_scope="project",
            cwd=str(project),
        )
        decision_task = asyncio.create_task(bridge.permission_callback(request))
        for _ in range(20):
            rows = await (
                await db.conn.execute("SELECT permission_request_id FROM permission_requests WHERE status='pending'")
            ).fetchall()
            if rows:
                break
            await asyncio.sleep(0.01)
        assert len(rows) == 1
        permission_request_id = rows[0]["permission_request_id"]

        await bridge.shutdown()
        decision = await asyncio.wait_for(decision_task, timeout=1)
        assert decision.decision == PermissionDecisionKind.DENY
        row = await (
            await db.conn.execute("SELECT status, decision FROM permission_requests WHERE permission_request_id=?", (permission_request_id,))
        ).fetchone()
        assert dict(row) == {"status": "cancelled", "decision": "deny"}
    finally:
        await bridge.shutdown()
        await db.close()
