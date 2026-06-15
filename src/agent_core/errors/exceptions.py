from __future__ import annotations

from typing import Any

from agent_core.errors.codes import ErrorCode


class AgentCoreError(Exception):
    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message
        self.details = details or {}
        super().__init__(f"{self.code.value}: {message}")


class ConfigError(AgentCoreError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CONFIG_ERROR, message, details=details)


class ToolExecutionError(AgentCoreError):
    pass


class ProviderError(AgentCoreError):
    pass

