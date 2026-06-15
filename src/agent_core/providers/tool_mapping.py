from __future__ import annotations


def to_provider_tool_name(canonical_name: str) -> str:
    return canonical_name.replace(".", "__")


def from_provider_tool_name(provider_name: str, known_names: set[str]) -> str:
    if provider_name in known_names:
        return provider_name
    candidate = provider_name.replace("__", ".")
    return candidate if candidate in known_names else provider_name

