from __future__ import annotations

from pathlib import Path

from agent_core.agents.builtin import builtin_agent_definitions
from agent_core.agents.parsing import parse_agent_definition_file
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.types.agents import AgentDefinition


class AgentDefinitionRegistry:
    INTERNAL_ONLY_IDS = {"default_compact_agent"}

    def __init__(self) -> None:
        self._definitions: dict[str, AgentDefinition] = {}
        for definition in builtin_agent_definitions():
            self._definitions[definition.agent_definition_id] = definition

    def register(self, definition: AgentDefinition, *, source: str = "code") -> None:
        if definition.agent_definition_id in self.INTERNAL_ONLY_IDS and source != "builtin":
            raise AgentCoreError(ErrorCode.INVALID_AGENT_OVERRIDE, f"cannot override internal agent definition: {definition.agent_definition_id}")
        existing = self._definitions.get(definition.agent_definition_id)
        if existing and existing.source == "code" and source != "code":
            raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, "cannot override code registered definition")
        self._definitions[definition.agent_definition_id] = definition

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
