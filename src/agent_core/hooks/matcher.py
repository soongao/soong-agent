from __future__ import annotations


from __future__ import annotations

from pathlib import Path

from agent_core.config.validation import is_relative_to


def hook_matches(*, hook: dict, event_type: str, tool_name: str | None = None, target_path: str | None = None) -> bool:
    if hook.get("event_type") and hook["event_type"] != event_type:
        return False
    if hook.get("tool_name") and hook["tool_name"] != tool_name:
        return False
    if hook.get("canonical_tool_name") and hook["canonical_tool_name"] != tool_name:
        return False
    if hook.get("tag") and hook["tag"] not in set(hook.get("_tool_tags", [])):
        return False
    path_prefix = hook.get("path_prefix")
    if path_prefix and target_path:
        try:
            if not is_relative_to(Path(target_path).expanduser().resolve(), Path(path_prefix).expanduser().resolve()):
                return False
        except OSError:
            return False
    return True
