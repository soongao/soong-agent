from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from agent_core.types.common import StrictModel


class RetryConfig(StrictModel):
    max_attempts: int = 3
    initial_backoff_ms: int = 500
    max_backoff_ms: int = 8000


class ModelConfig(StrictModel):
    provider: Literal["openai-compatible", "anthropic", "ollama"] | str
    base_url: str | None = None
    api_key_env: str | None = ""
    name: str
    context_window: int = 8192
    max_output_tokens: int = 4096
    temperature: float = 0.2
    timeout_ms: int = 60000
    retry: RetryConfig = Field(default_factory=RetryConfig)


class RuntimeConfig(StrictModel):
    cancel_timeout_ms: int = 10000


class ContextConfig(StrictModel):
    session_db_path: str = "${SOONG_AGENT_HOME}/sessions.sqlite"
    active_path_only: bool = True
    allow_branch_from: str = "user_message"
    branch_summary: bool = False
    reserve_output_tokens: int = 4096
    dynamic_system_budget: int = 12000
    non_system_budget: int | None = None
    task_board_token_budget: int = 1200
    task_recent_changes_limit: int = 20
    task_recent_changes_window_minutes: int = 30


class CompactConfig(StrictModel):
    enabled: bool = True
    reserve_tokens: int = 8000
    keep_recent_tokens: int = 16000
    auto_background: bool = True
    recovery_sync: bool = True
    model_profile: str | dict[str, Any] | None = "compact"
    max_summary_tokens: int = 2048


class MemoryConfig(StrictModel):
    enabled: bool = True
    memory_dir: str = "${SOONG_AGENT_HOME}/memory"
    categories: list[str] = Field(default_factory=lambda: ["user", "feedback", "reference"])
    extract_every_messages: int = 8
    extract_every_tokens: int = 12000
    idle_seconds: int = 120
    catalog_max_tokens: int = 4000
    recall_top_k: int = 5
    memory_context_token_budget: int = 6000
    extract_model_profile: str | dict[str, Any] | None = None
    recall_model_profile: str | dict[str, Any] | None = None


class WorkerConfig(StrictModel):
    worker_id: str | None = None
    agent_definition_id: str
    allowed_tools: list[str] | None = None


class WorkerPoolConfig(StrictModel):
    pool_id: str
    workers: list[WorkerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_worker_ids(self) -> "WorkerPoolConfig":
        seen: set[str] = set()
        for worker in self.workers:
            if worker.worker_id is None:
                continue
            if worker.worker_id in seen:
                raise ValueError(f"duplicate worker_id in pool {self.pool_id}: {worker.worker_id}")
            seen.add(worker.worker_id)
        return self


class AgentsConfig(StrictModel):
    max_children_per_run: int = 4
    default_sub_agent_definition: str = "default_sub_agent"
    default_fork_agent_definition: str = "default_fork_agent"
    worker_pools: list[WorkerPoolConfig] = Field(default_factory=list)
    max_concurrent_children_per_session: int = 8
    default_child_timeout_ms: int = 600000
    child_cancel_timeout_ms: int = 30000


class PlanConfig(StrictModel):
    default_dir: str = "<project>/.soong-agent/plans"
    template_name: str = "default"


class TaskConfig(StrictModel):
    wal_dir: str = "<project>/.soong-agent/tasks"
    task_board_token_budget: int = 1200
    task_recent_changes_limit: int = 20
    task_recent_changes_window_minutes: int = 30
    step_lease_timeout_ms: int = 300000


class NetworkPolicyConfig(StrictModel):
    default: Literal["allow", "confirm", "deny"] = "confirm"
    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)


class PermissionsConfig(StrictModel):
    readonly_default: Literal["allow", "confirm", "deny"] = "allow"
    write_without_callback: Literal["allow", "deny"] = "deny"
    remember_scope: Literal["session"] = "session"
    allow_for_session_enabled: bool = True
    network_policy: NetworkPolicyConfig = Field(default_factory=NetworkPolicyConfig)


class HooksConfig(StrictModel):
    enabled: bool = True
    default_timeout_ms: int = 30000


class ToolsNetworkConfig(StrictModel):
    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)


class ToolOverrideConfig(StrictModel):
    permission: Literal["readonly", "write"] | None = None
    tags: list[str] | None = None
    description: str | None = None


class McpToolsConfig(StrictModel):
    disabled_servers: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    tool_overrides: dict[str, ToolOverrideConfig] = Field(default_factory=dict)
    discovery_cache_ttl_ms: int = 60000


class ToolsConfig(StrictModel):
    declarative_enabled: bool = True
    disabled: list[str] = Field(default_factory=list)
    overrides: dict[str, ToolOverrideConfig] = Field(default_factory=dict)
    allowed_write_roots: list[str] = Field(default_factory=list)
    allow_tmp_write: bool = False
    default_timeout_ms: int = 120000
    max_timeout_ms: int = 600000
    env_allowlist: list[str] = Field(default_factory=lambda: ["PATH", "HOME", "TMPDIR"])
    stdout_limit_bytes: int = 65536
    stderr_limit_bytes: int = 65536
    network: ToolsNetworkConfig = Field(default_factory=ToolsNetworkConfig)
    sensitive_paths: list[str] = Field(
        default_factory=lambda: [
            "~/.ssh",
            "~/.gnupg",
            "~/.aws",
            "~/.config/gcloud",
            "*.pem",
            "*.key",
            ".env",
            ".env.*",
        ]
    )
    mcp: McpToolsConfig = Field(default_factory=McpToolsConfig)


class AgentCoreConfig(StrictModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    context: ContextConfig = Field(default_factory=ContextConfig)
    compact: CompactConfig = Field(default_factory=CompactConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @field_validator("model_overrides", mode="before")
    @classmethod
    def normalize_model_overrides(cls, value: Any) -> Any:
        return value or {}
