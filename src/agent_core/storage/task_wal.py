from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TaskWalWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        self.append_many([payload])

    def append_many(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        serialized = "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(serialized)
