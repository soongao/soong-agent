from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from agent_core.config.models import AgentsConfig, RetryConfig, WorkerConfig, WorkerPoolConfig
from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode
from agent_core.types.agents import AgentDefinition
from agent_core.types.common import StrictModel, validate_safe_id


class ModelOverrideConfig(StrictModel):
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    timeout_ms: int | None = None
    retry: RetryConfig | None = None


class JsonAgentDefinitionConfig(StrictModel):
    agent_definition_id: str = Field(alias="id")
    name: str
    description: str = ""
    body: str | None = None
    system_prompt: str | None = None
    model_profile: str | None = None
    model: ModelOverrideConfig | None = None
    suggested_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    overrides: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_definition_id")
    @classmethod
    def validate_agent_definition_id(cls, value: str) -> str:
        return validate_safe_id(value, field_name="agent_definition_id")

    @field_validator("suggested_tools", "tags")
    @classmethod
    def unique_string_list(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(str(item) for item in value))

    @model_validator(mode="after")
    def validate_body(self) -> "JsonAgentDefinitionConfig":
        if not (self.body or self.system_prompt):
            raise ValueError("agent definition JSON requires body or system_prompt")
        return self


class WorkerConfigCreate(StrictModel):
    worker_id: str
    worker_pool_id: str = "default"
    agent_definition_id: str | None = None
    agent: JsonAgentDefinitionConfig | None = None
    name: str | None = None
    description: str = ""
    system_prompt: str | None = None
    model_profile: str | None = None
    model: ModelOverrideConfig | None = None
    allowed_tools: list[str] | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str) -> str:
        return validate_safe_id(value, field_name="worker_id")

    @field_validator("worker_pool_id")
    @classmethod
    def validate_worker_pool_id(cls, value: str) -> str:
        return validate_safe_id(value, field_name="worker_pool_id")

    @field_validator("agent_definition_id")
    @classmethod
    def validate_agent_definition_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_safe_id(value, field_name="agent_definition_id")

    @field_validator("allowed_tools")
    @classmethod
    def unique_allowed_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(str(item) for item in value))

    @model_validator(mode="after")
    def validate_worker_body_or_definition(self) -> "WorkerConfigCreate":
        if self.agent is not None:
            if self.agent_definition_id and self.agent_definition_id != self.agent.agent_definition_id:
                raise ValueError("agent_definition_id must match agent.agent_definition_id")
            self.agent_definition_id = self.agent.agent_definition_id
            self.name = self.name or self.agent.name
            self.description = self.description or self.agent.description
            self.system_prompt = self.system_prompt or self.agent.body or self.agent.system_prompt
            self.model_profile = self.model_profile or self.agent.model_profile
            self.model = self.model or self.agent.model
            merged_metadata = dict(self.agent.metadata)
            merged_metadata.update(self.metadata)
            merged_metadata["inline_agent"] = True
            merged_metadata["inline_agent_definition"] = {
                "suggested_tools": self.agent.suggested_tools,
                "tags": self.agent.tags,
                "overrides": self.agent.overrides,
            }
            self.metadata = merged_metadata
        if not self.agent_definition_id and not self.system_prompt:
            raise ValueError("worker config requires agent_definition_id, agent, or system_prompt")
        return self


class WorkerConfigUpdate(StrictModel):
    worker_pool_id: str | None = None
    agent_definition_id: str | None = None
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    model_profile: str | None = None
    model: ModelOverrideConfig | None = None
    allowed_tools: list[str] | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("worker_pool_id")
    @classmethod
    def validate_worker_pool_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_safe_id(value, field_name="worker_pool_id")

    @field_validator("agent_definition_id")
    @classmethod
    def validate_agent_definition_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_safe_id(value, field_name="agent_definition_id")

    @field_validator("allowed_tools")
    @classmethod
    def unique_allowed_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(str(item) for item in value))


class WorkerConfigView(StrictModel):
    worker_id: str
    worker_pool_id: str = "default"
    agent_definition_id: str
    name: str
    description: str = ""
    system_prompt: str | None = None
    model_profile: str | None = None
    model: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
    enabled: bool = True
    deleted_at: str | None = None
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_worker_config(self) -> WorkerConfig:
        return WorkerConfig(
            worker_id=self.worker_id,
            agent_definition_id=self.agent_definition_id,
            allowed_tools=self.allowed_tools,
        )


class MentionedWorkerDirective(StrictModel):
    mention: str
    worker_id: str | None = None
    worker_agent_id: str | None = None
    worker_pool_id: str | None = None
    name: str | None = None


class RunDirectives(StrictModel):
    mentioned_worker: MentionedWorkerDirective | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerMentionResolution(StrictModel):
    mention: str
    worker_id: str | None = None
    worker_agent_id: str | None = None
    worker_pool_id: str | None = None
    name: str | None = None
    status: str
    error_code: str | None = None
    error_message: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.worker_id is not None

    def to_directive(self) -> MentionedWorkerDirective:
        if not self.resolved or self.worker_id is None or self.worker_agent_id is None or self.worker_pool_id is None:
            raise ValueError("cannot create directive from unresolved worker mention")
        return MentionedWorkerDirective(
            mention=self.mention,
            worker_id=self.worker_id,
            worker_agent_id=self.worker_agent_id,
            worker_pool_id=self.worker_pool_id,
            name=self.name or self.worker_id,
        )


class WorkerQueueItem(StrictModel):
    queue_id: str
    worker_id: str
    worker_agent_id: str
    session_id: str
    parent_run_id: str
    parent_agent_id: str
    task_id: str
    status: str
    position: int | None = None
    created_at: str
    updated_at: str
    cancelled: bool = False


def synthesized_agent_definition_id(worker_id: str) -> str:
    return f"worker.{worker_id}"


def model_profile_value(model_profile: str | None, model: ModelOverrideConfig | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if model is not None:
        if isinstance(model, ModelOverrideConfig):
            return model.model_dump(mode="python", exclude_none=True)
        return {key: value for key, value in model.items() if value is not None}
    return model_profile


def agent_definition_from_json_config(config: JsonAgentDefinitionConfig, *, path: Path) -> AgentDefinition | None:
    if not config.enabled:
        return None
    metadata = dict(config.metadata)
    metadata.update({"path": str(path), "json_enabled": config.enabled})
    return AgentDefinition(
        agent_definition_id=config.agent_definition_id,
        name=config.name,
        description=config.description,
        body=(config.body if config.body is not None else config.system_prompt) or "",
        model_profile=model_profile_value(config.model_profile, config.model),
        suggested_tools=config.suggested_tools,
        tags=config.tags,
        overrides=config.overrides,
        source="json",
        metadata=metadata,
    )


def agent_definition_from_worker_config(
    config: WorkerConfigCreate | WorkerConfigView,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> AgentDefinition | None:
    if not config.system_prompt:
        return None
    agent_definition_id = config.agent_definition_id or synthesized_agent_definition_id(config.worker_id)
    output_metadata = dict(metadata or {})
    inline_agent_definition = output_metadata.get("inline_agent_definition")
    if not isinstance(inline_agent_definition, dict):
        inline_agent_definition = {}
    output_metadata.update(
        {
            "worker_id": config.worker_id,
            "worker_pool_id": config.worker_pool_id,
            "generated_from_worker_config": True,
        }
    )
    model = config.model
    if isinstance(config, WorkerConfigCreate):
        model = config.model.model_dump(mode="python", exclude_none=True) if config.model is not None else None
    return AgentDefinition(
        agent_definition_id=agent_definition_id,
        name=config.name or config.worker_id,
        description=config.description or "",
        body=config.system_prompt,
        model_profile=model_profile_value(config.model_profile, model),
        suggested_tools=[str(item) for item in inline_agent_definition.get("suggested_tools", [])],
        tags=list(dict.fromkeys(["worker", *[str(item) for item in inline_agent_definition.get("tags", [])]])),
        overrides=inline_agent_definition.get("overrides"),
        source=source,  # type: ignore[arg-type]
        metadata=output_metadata,
    )


def worker_view_from_create(config: WorkerConfigCreate, *, source: str, path: Path | None = None) -> WorkerConfigView:
    metadata = dict(config.metadata)
    if path is not None:
        metadata["path"] = str(path)
    agent_definition_id = config.agent_definition_id or synthesized_agent_definition_id(config.worker_id)
    return WorkerConfigView(
        worker_id=config.worker_id,
        worker_pool_id=config.worker_pool_id,
        agent_definition_id=agent_definition_id,
        name=config.name or config.worker_id,
        description=config.description,
        system_prompt=config.system_prompt,
        model_profile=config.model_profile if config.model is None else None,
        model=config.model.model_dump(mode="python", exclude_none=True) if config.model is not None else None,
        allowed_tools=config.allowed_tools,
        enabled=config.enabled,
        deleted_at=None,
        source=source,
        metadata=metadata,
    )


def load_json_agent_definitions(path: Path) -> list[AgentDefinition]:
    if not path.exists():
        return []
    definitions: list[AgentDefinition] = []
    seen: set[str] = set()
    for file in sorted(path.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            config = JsonAgentDefinitionConfig.model_validate(data)
        except Exception as exc:
            raise AgentCoreError(ErrorCode.INVALID_AGENT_DEFINITION, f"invalid JSON agent definition: {file}") from exc
        if config.agent_definition_id in seen:
            raise AgentCoreError(ErrorCode.DUPLICATE_AGENT_DEFINITION, config.agent_definition_id)
        seen.add(config.agent_definition_id)
        definition = agent_definition_from_json_config(config, path=file)
        if definition is not None:
            definitions.append(definition)
    return definitions


def load_json_worker_configs(path: Path) -> list[WorkerConfigView]:
    if not path.exists():
        return []
    workers: list[WorkerConfigView] = []
    seen: set[str] = set()
    for file in sorted(path.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            config = WorkerConfigCreate.model_validate(data)
        except Exception as exc:
            raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"invalid JSON worker config: {file}") from exc
        if config.worker_id in seen:
            raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"duplicate JSON worker_id: {config.worker_id}")
        seen.add(config.worker_id)
        workers.append(worker_view_from_create(config, source="json", path=file))
    return workers


def config_worker_views(config: AgentsConfig, definitions: dict[str, AgentDefinition]) -> list[WorkerConfigView]:
    views: list[WorkerConfigView] = []
    for pool in config.worker_pools:
        for index, worker in enumerate(pool.workers):
            worker_id = worker.worker_id or f"{pool.pool_id}_{worker.agent_definition_id}_{index}"
            definition = definitions.get(worker.agent_definition_id)
            views.append(
                WorkerConfigView(
                    worker_id=worker_id,
                    worker_pool_id=pool.pool_id,
                    agent_definition_id=worker.agent_definition_id,
                    name=definition.name if definition is not None else worker_id,
                    description=definition.description if definition is not None else "",
                    system_prompt=definition.body if definition is not None and definition.source != "builtin" else None,
                    model_profile=definition.model_profile if isinstance(definition.model_profile, str) else None,
                    model=definition.model_profile if isinstance(definition.model_profile, dict) else None,
                    allowed_tools=worker.allowed_tools,
                    enabled=True,
                    source="config",
                    metadata={},
                )
            )
    return views


def merge_worker_views(*sources: list[WorkerConfigView]) -> list[WorkerConfigView]:
    merged: dict[str, WorkerConfigView] = {}
    order: list[str] = []
    for source in sources:
        seen_in_source: set[str] = set()
        for worker in source:
            if worker.worker_id in seen_in_source:
                raise AgentCoreError(ErrorCode.CONFIG_ERROR, f"duplicate worker_id in {worker.source}: {worker.worker_id}")
            seen_in_source.add(worker.worker_id)
            if worker.worker_id not in merged:
                order.append(worker.worker_id)
            merged[worker.worker_id] = worker
    return [merged[worker_id] for worker_id in order]


def agents_config_from_worker_views(base: AgentsConfig, workers: list[WorkerConfigView]) -> AgentsConfig:
    pools: dict[str, list[WorkerConfig]] = {}
    pool_order: list[str] = []
    for worker in workers:
        if not worker.enabled or worker.deleted_at is not None:
            continue
        if worker.worker_pool_id not in pools:
            pools[worker.worker_pool_id] = []
            pool_order.append(worker.worker_pool_id)
        pools[worker.worker_pool_id].append(worker.to_worker_config())
    return base.model_copy(
        update={
            "worker_pools": [
                WorkerPoolConfig(pool_id=pool_id, workers=pools[pool_id])
                for pool_id in pool_order
            ]
        },
        deep=True,
    )
