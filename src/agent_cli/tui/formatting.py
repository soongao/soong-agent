from __future__ import annotations


NODE_ID_DISPLAY_LENGTH = 28


def format_message(role: str, text: str) -> str:
    labels = {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
        "event": "EVENT",
        "warning": "WARNING",
        "error": "ERROR",
    }
    label = labels.get(role, role.upper())
    body = text.rstrip()
    return f"{label}\n{body}" if body else label


def compact_time(value) -> str:
    text = str(value or "")
    return text.replace("T", " ")[:19]


def short_id(value: str | None, length: int = 18) -> str:
    if not value:
        return "-"
    return value if len(value) <= length else value[:length]


def format_node_row(node, *, selected: bool = False) -> str:
    marker = "*" if selected or getattr(node, "active", False) else " "
    preview = getattr(node, "content_preview", "") or ""
    return (
        f"{marker} {short_id(node.node_id, NODE_ID_DISPLAY_LENGTH):<{NODE_ID_DISPLAY_LENGTH}} "
        f"{node.role:<9} {node.node_type:<14} "
        f"{preview}"
    ).rstrip()
