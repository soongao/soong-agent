from __future__ import annotations

from agent_core.artifacts import ArtifactManager


def test_artifact_ids_do_not_dedupe_identical_content(isolated_dirs) -> None:
    home, _project = isolated_dirs
    manager = ArtifactManager(home_dir=home)
    first = manager.write_text(session_id="sess", text="same", filename="a.txt")
    second = manager.write_text(session_id="sess", text="same", filename="a.txt")
    assert first.artifact_id != second.artifact_id
    assert first.path.exists()
    assert second.path.exists()


def test_artifact_text_is_redacted(isolated_dirs, monkeypatch) -> None:
    home, _project = isolated_dirs
    monkeypatch.setenv("TEST_API_TOKEN", "super-secret-token")
    manager = ArtifactManager(home_dir=home)
    artifact = manager.write_text(session_id="sess", text="value=super-secret-token", filename="log.txt")
    assert "super-secret-token" not in artifact.path.read_text(encoding="utf-8")
    assert "[REDACTED]" in artifact.path.read_text(encoding="utf-8")
