from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class SessionState:
    session_id: str
    active_run_id: str | None = None
    active_node_id: str | None = None
    metadata: dict = field(default_factory=dict)
