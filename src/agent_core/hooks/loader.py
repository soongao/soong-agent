from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_hooks(home_dir: Path) -> dict[str, Any]:
    path = home_dir / "hooks.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_hooks(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config:
        return []
    if isinstance(config.get("hooks"), list):
        return config["hooks"]
    if isinstance(config.get("hooks"), dict):
        normalized: list[dict[str, Any]] = []
        for raw_event_type, rules in config["hooks"].items():
            event_type = _event_type(raw_event_type)
            for rule in rules or []:
                matcher = dict(rule.get("matcher") or {})
                hooks = rule.get("hooks") or []
                if not hooks and (rule.get("type") or rule.get("command") or rule.get("action") or rule.get("decision")):
                    hooks = [rule]
                for hook in hooks:
                    item = dict(matcher)
                    item.update(hook)
                    item["event_type"] = item.get("event_type") or event_type
                    normalized.append(item)
        return normalized
    if isinstance(config.get("rules"), list):
        return config["rules"]
    return []


def _event_type(value: str) -> str:
    mapping = {
        "PreToolUse": "tool_started",
        "PostToolUse": "tool_completed",
        "SessionStart": "session_started",
        "UserPromptSubmit": "user_prompt_submitted",
        "Stop": "stop",
    }
    return mapping.get(value, value)
