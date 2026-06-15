from __future__ import annotations

import shutil
import subprocess
import time

import httpx
import pytest


@pytest.fixture(scope="session")
def ollama_server():
    if shutil.which("ollama") is None:
        pytest.skip("ollama command is not installed")
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
            pytest.fail("ollama serve did not become healthy")
    yield "http://127.0.0.1:11434"
    if started and proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _healthy() -> bool:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1)
        return response.status_code == 200
    except Exception:
        return False

