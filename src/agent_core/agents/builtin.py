from __future__ import annotations

from agent_core.agents.parsing import parse_builtin_agent_definition_resource
from agent_core.types.agents import AgentDefinition


def builtin_agent_definitions() -> list[AgentDefinition]:
    definitions = [
        parse_builtin_agent_definition_resource("default_sub_agent.md").model_copy(update={"suggested_tools": ["code.read_file", "code.search"]}),
        parse_builtin_agent_definition_resource("default_fork_agent.md").model_copy(update={"suggested_tools": ["code.read_file", "code.search"]}),
        parse_builtin_agent_definition_resource("default_worker_agent.md").model_copy(
            update={
                "suggested_tools": [
                    "agent.task_get",
                    "agent.task_query_steps",
                    "agent.task_claim_step",
                    "agent.task_update_step",
                    "code.read_file",
                    "code.search",
                ]
            }
        ),
        parse_builtin_agent_definition_resource("default_compact_agent.md").model_copy(update={"tags": ["internal"]}),
    ]
    return definitions
