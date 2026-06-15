from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from agent_core.types.common import validate_safe_id


def new_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return validate_safe_id(f"{prefix}_{stamp}_{uuid4().hex[:8]}")


def sanitize_session_id(session_id: str) -> str:
    return validate_safe_id(session_id, field_name="session_id")

