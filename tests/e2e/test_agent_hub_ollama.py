from __future__ import annotations

import time

from fastapi.testclient import TestClient

from agent_hub.backend.app import create_app
from tests.conftest import write_config
from tests.fixtures.ollama import ollama_server


def test_agent_hub_ollama_normal_flow_uses_configured_provider(isolated_dirs, ollama_server) -> None:
    home, project = isolated_dirs
    write_config(
        home,
        provider="openai",
        base_url=f"{ollama_server}/v1",
        model_name="gemma4",
        worker_pool=True,
    )
    config_path = home / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    text = text.replace('api_key_env = ""\nname = "gemma4"', 'api_key_env = ""\napi_key = "ollama"\nname = "gemma4"')
    text = text.replace("timeout_ms = 1000", "timeout_ms = 60000")
    config_path.write_text(text, encoding="utf-8")

    app = create_app(home_dir=home, project_dir=project)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        health_data = health.json()
        assert health_data["ok"] is True
        assert health_data["provider"] == "openai"
        assert health_data["model"] == "gemma4"
        assert health_data["base_url"] == f"{ollama_server}/v1"

        conversation = client.post("/conversations", json={"title": "Ollama E2E"})
        assert conversation.status_code == 200
        conversation_id = conversation.json()["conversation_id"]

        sent = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"text": "Reply with exactly: hub-pong"},
        )
        assert sent.status_code == 200
        assert sent.json()["status"] in {"running", "queued"}

        messages = []
        for _ in range(120):
            messages = client.get(f"/conversations/{conversation_id}/messages").json()["messages"]
            if any(message["sender_type"] == "orchestrator" and message["status"] == "completed" for message in messages):
                break
            time.sleep(0.5)

        assert any(message["sender_type"] == "user" and message["core_node_id"] for message in messages)
        orchestrator_messages = [message for message in messages if message["sender_type"] == "orchestrator"]
        assert orchestrator_messages
        assert orchestrator_messages[-1]["status"] == "completed"
        assert "hub-pong" in orchestrator_messages[-1]["display_text"]
