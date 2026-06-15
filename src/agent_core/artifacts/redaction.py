from __future__ import annotations

import os
from typing import Any


def redact_text(text: str) -> str:
    redacted = text
    for key, value in os.environ.items():
        if not value:
            continue
        upper = key.upper()
        if any(marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 8:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _redact_model(value)
    return value


def _redact_model(model: Any) -> Any:
    data = redact_value(model.model_dump(mode="python"))
    return model.__class__.model_validate(data)
