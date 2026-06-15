from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.errors.codes import ErrorCode

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


class ErrorPayload(StrictModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def validate_safe_id(value: str, *, field_name: str = "id") -> str:
    if not SAFE_ID_RE.match(value):
        raise ValueError(f"{field_name} contains unsafe characters")
    return value
