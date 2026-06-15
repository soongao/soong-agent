from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mcp_config(home_dir: Path) -> dict[str, Any]:
    path = home_dir / "mcp.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

