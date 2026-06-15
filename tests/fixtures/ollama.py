from __future__ import annotations

import os
import shutil
import subprocess
import time

import httpx
import pytest


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma4"


@pytest.fixture(scope="session")
def ollama_server():
    if shutil.which("ollama") is None:
        _unavailable("ollama command is not installed")
    started = False
    proc: subprocess.Popen | None = None
    if not _healthy():
        proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = True
        deadline = time.time() + 20
        while time.time() < deadline:
            if _healthy():
                break
            time.sleep(0.25)
        else:
            if proc:
                proc.terminate()
            _unavailable("ollama serve did not become healthy")
    if not _model_available(OLLAMA_MODEL):
        _unavailable(f"ollama model {OLLAMA_MODEL!r} is not installed")
    yield OLLAMA_BASE_URL
    if started and proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _healthy() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
        return response.status_code == 200
    except Exception:
        return False


def _model_available(model_name: str) -> bool:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
    except Exception:
        return False
    models = response.json().get("models") or []
    names = {str(model.get("name") or "").split(":", 1)[0] for model in models}
    return model_name in names


def _unavailable(reason: str) -> None:
    if os.environ.get("SOONG_AGENT_REQUIRE_OLLAMA_E2E") in {"1", "true", "yes"}:
        pytest.fail(reason)
    pytest.skip(reason)
