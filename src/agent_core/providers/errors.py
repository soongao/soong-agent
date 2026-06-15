from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agent_core.config.models import RetryConfig
from agent_core.errors.codes import ErrorCode
from agent_core.types.common import ErrorPayload


@dataclass(frozen=True)
class ProviderErrorInfo:
    code: ErrorCode
    message: str
    retryable: bool
    status_code: int | None = None
    retry_after_ms: int | None = None
    raw: Any | None = None

    def payload(self) -> ErrorPayload:
        details: dict[str, Any] = {"retryable": self.retryable}
        if self.status_code is not None:
            details["status_code"] = self.status_code
        if self.retry_after_ms is not None:
            details["retry_after_ms"] = self.retry_after_ms
        return ErrorPayload(code=self.code, message=self.message, details=details)


def classify_provider_exception(exc: Exception) -> ProviderErrorInfo:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderErrorInfo(code=ErrorCode.PROVIDER_TIMEOUT, message=str(exc), retryable=True, raw=exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        retry_after_ms = _retry_after_ms(exc.response.headers.get("retry-after"))
        if status == 401 or status == 403:
            return ProviderErrorInfo(
                code=ErrorCode.PROVIDER_AUTH_FAILED,
                message=f"provider auth failed with HTTP {status}",
                retryable=False,
                status_code=status,
                raw=exc,
            )
        if status == 429:
            return ProviderErrorInfo(
                code=ErrorCode.PROVIDER_RATE_LIMITED,
                message="provider rate limited request",
                retryable=True,
                status_code=status,
                retry_after_ms=retry_after_ms,
                raw=exc,
            )
        if 500 <= status <= 599:
            return ProviderErrorInfo(
                code=ErrorCode.PROVIDER_ERROR,
                message=f"provider server error HTTP {status}",
                retryable=True,
                status_code=status,
                retry_after_ms=retry_after_ms,
                raw=exc,
            )
        return ProviderErrorInfo(
            code=ErrorCode.PROVIDER_ERROR,
            message=f"provider request failed with HTTP {status}",
            retryable=False,
            status_code=status,
            raw=exc,
        )
    if isinstance(exc, httpx.TransportError):
        return ProviderErrorInfo(code=ErrorCode.PROVIDER_ERROR, message=str(exc), retryable=True, raw=exc)
    return ProviderErrorInfo(code=ErrorCode.PROVIDER_ERROR, message=str(exc), retryable=False, raw=exc)


def retry_delay_seconds(*, attempt_index: int, retry: RetryConfig, retry_after_ms: int | None = None) -> float:
    if retry_after_ms is not None:
        return max(retry_after_ms, 0) / 1000
    initial = max(retry.initial_backoff_ms, 0)
    max_backoff = max(retry.max_backoff_ms, initial)
    return min(initial * (2 ** max(attempt_index - 1, 0)), max_backoff) / 1000


def _retry_after_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value) * 1000)
    except ValueError:
        return None
