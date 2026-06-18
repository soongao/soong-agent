from __future__ import annotations

from typing import Any, NoReturn

from fastapi.responses import JSONResponse

from agent_core.errors import AgentCoreError


class HubApiError(Exception):
    def __init__(self, *, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def raise_hub_error(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> NoReturn:
    raise HubApiError(status_code=status_code, code=code, message=message, details=details)


def hub_error_response(error: HubApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            }
        },
    )


def agent_core_error_response(error: AgentCoreError) -> JSONResponse:
    return JSONResponse(
        status_code=_status_for_core_error(error),
        content={
            "error": {
                "code": error.code.value,
                "message": str(error),
                "details": error.details,
            }
        },
    )


def _status_for_core_error(error: AgentCoreError) -> int:
    if error.code.value in {"worker_not_available", "task_not_found", "step_not_found", "file_not_found"}:
        return 404
    if error.code.value in {"session_active", "worker_busy", "worker_pool_busy", "worker_queue_full"}:
        return 409
    if error.code.value in {"config_error", "validation_error", "schema_error"}:
        return 400
    return 500
