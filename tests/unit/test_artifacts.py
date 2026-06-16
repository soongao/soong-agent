from __future__ import annotations

from agent_core.artifacts import ArtifactManager
from agent_core.artifacts.redaction import redact_text, redact_value


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


def test_artifact_textual_bytes_are_redacted(isolated_dirs) -> None:
    home, _project = isolated_dirs
    manager = ArtifactManager(home_dir=home)
    artifact = manager.write_bytes(
        session_id="sess",
        data=b'{"api_key":"byte-secret-value"}',
        filename="raw.json",
        mime_type="application/json",
    )

    text = artifact.path.read_text(encoding="utf-8")
    assert "byte-secret-value" not in text
    assert "[REDACTED]" in text


def test_artifact_filename_must_be_single_path_segment(isolated_dirs) -> None:
    home, _project = isolated_dirs
    manager = ArtifactManager(home_dir=home)

    import pytest

    with pytest.raises(ValueError, match="single path segment"):
        manager.write_text(session_id="sess", text="x", filename="../escape.txt")


def test_redact_value_redacts_sensitive_field_names() -> None:
    payload = {
        "api_key": "key-from-field",
        "nested": {
            "accessToken": "token-from-field",
            "normal": "visible",
            "token_count": 12,
        },
        "items": [{"password": "password-from-field"}],
    }

    redacted = redact_value(payload)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["accessToken"] == "[REDACTED]"
    assert redacted["nested"]["normal"] == "visible"
    assert redacted["nested"]["token_count"] == 12
    assert redacted["items"][0]["password"] == "[REDACTED]"


def test_redact_text_redacts_common_secret_assignments_without_env() -> None:
    text = (
        'api_key="key-from-text"\n'
        "password=pass-from-text\n"
        "authorization: Bearer bearer-token-value\n"
        "token_count=12\n"
    )

    redacted = redact_text(text)

    assert "key-from-text" not in redacted
    assert "pass-from-text" not in redacted
    assert "bearer-token-value" not in redacted
    assert "token_count=12" in redacted
    assert redacted.count("[REDACTED]") == 3
