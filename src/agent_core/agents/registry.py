from __future__ import annotations

from pathlib import Path

from agent_core.agents.builtin import builtin_agent_definitions
from agent_core.agents.dynamic import load_json_agent_definitions
from agent_core.agents.parsing import parse_agent_definition_file
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.types.agents import AgentDefinition


class AgentDefinitionRegistry:
    INTERNAL_ONLY_IDS = {"default_compact_agent"}

    def __init__(self) -> None:
        self._definitions: dict[str, AgentDefinition] = {}
        self.reset_to_builtin()

    def reset_to_builtin(self) -> None:
        self._definitions = {}
        for definition in builtin_agent_definitions():
            self._definitions[definition.agent_definition_id] = definition

    def code_definitions(self) -> list[AgentDefinition]:
        return [definition for definition in self._definitions.values() if definition.source == "code"]

    def register(self, definition: AgentDefinition, *, source: str = "code") -> None:
        if definition.agent_definition_id in self.INTERNAL_ONLY_IDS and source != "builtin":
            raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, f"cannot override internal agent definition: {definition.agent_definition_id}")
        existing = self._definitions.get(definition.agent_definition_id)
        if existing is None:
            if definition.overrides:
                raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, f"override target not found: {definition.overrides}")
            self._definitions[definition.agent_definition_id] = _with_registry_metadata(definition, source=source)
            return
        if existing.source == source:
            raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, f"duplicate agent definition: {definition.agent_definition_id}")
        if existing.source == "code" and source != "code":
            raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, "cannot override code registered definition")
        _validate_override(definition, existing)
        if source == "user" and existing.source != "builtin":
            raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, "user definitions can only override builtin definitions")
        self._definitions[definition.agent_definition_id] = _with_registry_metadata(
            definition,
            source=source,
            overridden=existing,
        )

    def register_overlay(self, definition: AgentDefinition, *, source: str) -> None:
        if definition.agent_definition_id in self.INTERNAL_ONLY_IDS and source != "builtin":
            raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, f"cannot override internal agent definition: {definition.agent_definition_id}")
        existing = self._definitions.get(definition.agent_definition_id)
        if existing is not None and existing.source == "code" and source != "code":
            raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, "cannot override code registered definition")
        self._definitions[definition.agent_definition_id] = _with_registry_metadata(
            definition,
            source=source,
            overridden=existing,
        )

    def get(self, agent_definition_id: str) -> AgentDefinition | None:
        return self._definitions.get(agent_definition_id)

    def list(self) -> list[AgentDefinition]:
        return sorted(
            [definition for definition in self._definitions.values() if definition.agent_definition_id not in self.INTERNAL_ONLY_IDS],
            key=lambda item: item.agent_definition_id,
        )

    def validate_suggested_tools(self, available_tool_names: set[str]) -> None:
        for definition in self._definitions.values():
            missing = [name for name in definition.suggested_tools if name not in available_tool_names]
            if missing:
                raise AgentCoreError(
                    ErrorCode.INVALID_AGENT_DEFINITION,
                    f"agent definition {definition.agent_definition_id} references unknown suggested_tools: {missing}",
                )

    def load_user_dir(self, path: Path) -> None:
        if not path.exists():
            return
        seen: set[str] = set()
        for file in sorted(path.glob("*.md")):
            definition = parse_agent_definition_file(file)
            if definition.agent_definition_id in seen:
                raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, definition.agent_definition_id)
            seen.add(definition.agent_definition_id)
            self.register(definition, source=definition.source)

    def load_json_dir(self, path: Path) -> list[AgentDefinition]:
        definitions = load_json_agent_definitions(path)
        for definition in definitions:
            self.register_overlay(definition, source=definition.source)
        return definitions


def _validate_override(definition: AgentDefinition, existing: AgentDefinition) -> None:
    if not definition.overrides:
        raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, f"duplicate agent definition: {definition.agent_definition_id}")
    try:
        target_source, target_id = definition.overrides.split(":", 1)
    except ValueError as exc:
        raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, "overrides must use '<source>:<agent_definition_id>'") from exc
    if target_source != existing.source or target_id != existing.agent_definition_id:
        raise AgentCoreError(
            ErrorCode.INVALID_AGENT_OVERRIDE,
            f"override target mismatch: expected {existing.source}:{existing.agent_definition_id}",
        )


def _with_registry_metadata(
    definition: AgentDefinition,
    *,
    source: str,
    overridden: AgentDefinition | None = None,
) -> AgentDefinition:
    metadata = dict(definition.metadata)
    metadata["registry_source"] = source
    if overridden is not None:
        metadata["overrides"] = {
            "agent_definition_id": overridden.agent_definition_id,
            "source": overridden.source,
        }
    return definition.model_copy(update={"source": source, "metadata": metadata})
