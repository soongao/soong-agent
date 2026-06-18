from __future__ import annotations

from pathlib import Path


def test_agenthub_script_uses_agenthub_names_and_project_dir() -> None:
    script = Path("agenthub").read_text(encoding="utf-8")
    package_json = Path("src/agent_hub/frontend/package.json").read_text(encoding="utf-8")

    assert "AGENTHUB_PROJECT_DIR" in script
    assert "AGENTHUB_REPO_ROOT" in script
    assert "AGENTHUB_PYTHONPATH" in script
    assert "python3 -m agent_hub.backend" in script
    assert "concurrently --kill-others --success first" in package_json
    assert "soong-hub" not in script
