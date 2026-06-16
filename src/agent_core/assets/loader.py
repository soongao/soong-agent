from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from agent_core.errors.codes import ErrorCode
from agent_core.errors.exceptions import AgentCoreError


@dataclass(frozen=True)
class PackageAsset:
    asset_id: str
    package: str
    filename: str

    @property
    def resource_path(self) -> str:
        return f"{self.package.replace('agent_core.assets.', '').replace('.', '/')}/{self.filename}"


ASSET_REGISTRY: dict[str, PackageAsset] = {
    "system.core": PackageAsset("system.core", "agent_core.assets.prompts.system", "core.md"),
    "system.tool_protocol": PackageAsset(
        "system.tool_protocol",
        "agent_core.assets.prompts.system",
        "tool_protocol.md",
    ),
    "system.todo": PackageAsset("system.todo", "agent_core.assets.prompts.system", "todo.md"),
    "system.permissions": PackageAsset(
        "system.permissions",
        "agent_core.assets.prompts.system",
        "permissions.md",
    ),
    "system.multi_agent": PackageAsset(
        "system.multi_agent",
        "agent_core.assets.prompts.system",
        "multi_agent.md",
    ),
    "system.memory": PackageAsset("system.memory", "agent_core.assets.prompts.system", "memory.md"),
    "system.compact": PackageAsset("system.compact", "agent_core.assets.prompts.system", "compact.md"),
    "template.config.default": PackageAsset(
        "template.config.default",
        "agent_core.assets.templates",
        "config_default.toml",
    ),
    "template.plan.default": PackageAsset(
        "template.plan.default",
        "agent_core.assets.templates",
        "plan_default.md",
    ),
    "template.task_dag.default": PackageAsset(
        "template.task_dag.default",
        "agent_core.assets.templates",
        "task_dag_default.md",
    ),
    "agent.default_sub_agent": PackageAsset(
        "agent.default_sub_agent",
        "agent_core.assets.agents",
        "default_sub_agent.md",
    ),
    "agent.default_fork_agent": PackageAsset(
        "agent.default_fork_agent",
        "agent_core.assets.agents",
        "default_fork_agent.md",
    ),
    "agent.default_worker_agent": PackageAsset(
        "agent.default_worker_agent",
        "agent_core.assets.agents",
        "default_worker_agent.md",
    ),
    "agent.default_compact_agent": PackageAsset(
        "agent.default_compact_agent",
        "agent_core.assets.agents",
        "default_compact_agent.md",
    ),
}


def get_asset(asset_id: str) -> PackageAsset:
    try:
        return ASSET_REGISTRY[asset_id]
    except KeyError as exc:
        raise AgentCoreError(
            ErrorCode.INTERNAL_ERROR,
            f"unknown package asset: {asset_id}",
        ) from exc


def read_asset(asset_id: str) -> str:
    asset = get_asset(asset_id)
    try:
        return resources.files(asset.package).joinpath(asset.filename).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AgentCoreError(
            ErrorCode.INTERNAL_ERROR,
            f"missing package asset {asset_id}: {asset.resource_path}",
        ) from exc


def list_required_assets() -> list[PackageAsset]:
    return list(ASSET_REGISTRY.values())
