from __future__ import annotations

import os
import re
from typing import Any


REDACTED = "[REDACTED]"

_SENSITIVE_KEY_NAMES = {
    "apikey",
    "accesstoken",
    "authtoken",
    "authorization",
    "clientsecret",
    "cookie",
    "csrftoken",
    "idtoken",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessiontoken",
    "setcookie",
    "token",
    "xsrftoken",
}
_INLINE_SECRET_KEY = (
    r"api[_-]?key|access[_-]?token|auth[_-]?token|authorization|"
    r"client[_-]?secret|csrf[_-]?token|id[_-]?token|password|passwd|"
    r"private[_-]?key|refresh[_-]?token|secret|session[_-]?token|"
    r"set[_-]?cookie|token|xsrf[_-]?token"
)
_QUOTED_SECRET_RE = re.compile(
    rf"(?i)([\"']?(?:{_INLINE_SECRET_KEY})[\"']?\s*[:=]\s*[\"'])([^\"']{{3,}})([\"'])"
)
_UNQUOTED_SECRET_RE = re.compile(
    rf"(?i)(\b(?:{_INLINE_SECRET_KEY})\b\s*[:=]\s*)(?!Bearer\b)([^\s,;}}&]+)"
)
_BEARER_RE = re.compile(r"(?i)(\bauthorization\b\s*[:=]\s*bearer\s+)([A-Za-z0-9._~+/=-]{8,})")


def redact_text(text: str) -> str:
    redacted = text
    for key, value in os.environ.items():
        if not value:
            continue
        upper = key.upper()
        if any(marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 8:
            redacted = redacted.replace(value, REDACTED)
    redacted = _BEARER_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    redacted = _QUOTED_SECRET_RE.sub(lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}", redacted)
    redacted = _UNQUOTED_SECRET_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_value(item)
        return redacted
    if hasattr(value, "model_dump"):
        return _redact_model(value)
    return value


def _redact_model(model: Any) -> Any:
    data = redact_value(model.model_dump(mode="python"))
    return model.__class__.model_validate(data)


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if normalized in _SENSITIVE_KEY_NAMES:
        return True
    return normalized.endswith(("secret", "password", "token"))
