from __future__ import annotations

import pytest

from agent_core.compact import RecoveryCompact, should_compact
from agent_core.errors import AgentCoreError
from agent_core.memory import MemoryCandidate, MemoryExtractionJob, MemoryRecallService, ensure_memory_write_allowed, parse_memory_candidates


def test_compact_policy_and_payload() -> None:
    assert should_compact(estimated_tokens=100, threshold=50)
    payload = RecoveryCompact().compact(texts=["a" * 10, "b" * 10], source_node_ids=["n1"], max_summary_chars=12)
    assert payload.summary == ("a" * 10 + "\nb")
    assert len(payload.summary) == 12
    assert payload.source_node_ids == ["n1"]


def test_memory_extraction_requires_source_node_ids(tmp_path) -> None:
    job = MemoryExtractionJob(home_dir=tmp_path)
    with pytest.raises(AgentCoreError):
        job.apply(
            [MemoryCandidate(category="user", filename="prefs.md", content="x", source_node_ids=[])],
            source_node_seq=1,
        )


def test_parse_memory_candidates_from_model_json() -> None:
    candidates = parse_memory_candidates(
        """
```json
{
  "memories": [
    {
      "decision": "new",
      "category": "user",
      "filename": "prefs.md",
      "summary": "Testing preference",
      "tags": ["testing"],
      "source_node_ids": ["node_1"],
      "content": "likes pytest"
    },
    {"decision": "ignore", "category": "user", "filename": "ignored.md"}
  ]
}
```
""".strip()
    )
    assert len(candidates) == 1
    assert candidates[0].category == "user"
    assert candidates[0].filename == "prefs.md"
    assert candidates[0].source_node_ids == ["node_1"]
    assert candidates[0].tags == ["testing"]


def test_memory_extraction_writes_and_advances_cursor(tmp_path) -> None:
    job = MemoryExtractionJob(home_dir=tmp_path)
    result = job.apply(
        [MemoryCandidate(category="user", filename="prefs.md", content="likes tests", source_node_ids=["n1"], summary="Testing preference")],
        source_node_seq=7,
    )
    path = tmp_path / "memory" / "user" / "prefs.md"
    text = path.read_text(encoding="utf-8")
    assert "source_node_ids:" in text
    assert "  - n1" in text
    assert text.endswith("likes tests")
    assert result.created == [str(path)]
    catalog = tmp_path / "memory" / "MEMORY.md"
    assert "`mem_prefs` [user] Testing preference (user/prefs.md)" in catalog.read_text(encoding="utf-8")
    assert str(catalog) in result.files_changed
    assert result.scan_cursor.node_seq == 7
    duplicate = job.apply(
        [MemoryCandidate(category="user", filename="prefs.md", content="likes tests", source_node_ids=["n1"], summary="Testing preference")],
        source_node_seq=8,
    )
    assert duplicate.duplicate == [str(path)]
    assert duplicate.scan_cursor.node_seq == 8


def test_memory_extraction_invalid_filename_does_not_write_or_advance_cursor(tmp_path) -> None:
    job = MemoryExtractionJob(home_dir=tmp_path)
    with pytest.raises(AgentCoreError):
        job.apply(
            [MemoryCandidate(category="user", filename="../prefs.md", content="x", source_node_ids=["n1"])],
            source_node_seq=9,
        )
    assert job.cursor.node_seq == 0
    assert not (tmp_path / "memory").exists()


def test_memory_writer_boundary(tmp_path) -> None:
    ensure_memory_write_allowed(tmp_path / "memory" / "user" / "a.md", home_dir=tmp_path)
    with pytest.raises(AgentCoreError):
        ensure_memory_write_allowed(tmp_path / "project" / "memory.md", home_dir=tmp_path)


def test_memory_recall_service(tmp_path) -> None:
    memory = tmp_path / "memory" / "reference"
    memory.mkdir(parents=True)
    (memory / "pytest.md").write_text("pytest notes", encoding="utf-8")
    matches = MemoryRecallService(memory_dir=tmp_path / "memory").recall("pytest")
    assert matches and matches[0]["path"].endswith("pytest.md")
