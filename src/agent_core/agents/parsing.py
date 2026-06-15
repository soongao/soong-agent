from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from typing import Any

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.types.agents import AgentDefinition


def parse_agent_definition_file(path: Path) -> AgentDefinition:
    text = path.read_text(encoding="utf-8")
    return parse_agent_definition_text(text, source="user", metadata={"path": str(path)})


def parse_builtin_agent_definition_resource(filename: str) -> AgentDefinition:
    text = resources.files("agent_core.assets.agents").joinpath(filename).read_text(encoding="utf-8")
    return parse_agent_definition_text(text, source="builtin", metadata={"asset_path": f"agents/{filename}"})


def parse_agent_definition_text(text: str, *, source: str, metadata: dict[str, str]) -> AgentDefinition:
    if not text.startswith("---\n"):
        raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, "missing frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, "unterminated frontmatter")
    frontmatter: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    for field in ("id", "name", "description"):
        if not frontmatter.get(field):
            raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"missing {field}")
    output_metadata = dict(metadata)
    output_metadata.update(frontmatter)
    return AgentDefinition(
        agent_definition_id=frontmatter["id"],
        name=frontmatter["name"],
        description=frontmatter["description"],
        body=text[end + 4 :].lstrip("\n"),
        model_profile=_parse_scalar_or_json(frontmatter.get("model_profile")),
        suggested_tools=_parse_list(frontmatter.get("suggested_tools")),
        tags=_parse_list(frontmatter.get("tags")),
        overrides=frontmatter.get("overrides") or None,
        source=source,  # type: ignore[arg-type]
        metadata=output_metadata,
    )


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = _parse_scalar_or_json(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [item.strip() for item in str(parsed).split(",") if item.strip()]


def _parse_scalar_or_json(value: str | None) -> Any:
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped
