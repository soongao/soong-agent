from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_core.providers.base import ProviderAdapter


ProviderFactory = Callable[[Any], ProviderAdapter]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, key: str, factory: ProviderFactory) -> None:
        self._factories[key] = factory

    def has(self, key: str) -> bool:
        return key in self._factories

    def create(self, key: str, config: Any) -> ProviderAdapter:
        if key not in self._factories:
            raise KeyError(f"provider not registered: {key}")
        return self._factories[key](config)


def default_provider_registry() -> ProviderRegistry:
    from agent_core.providers.anthropic import AnthropicProvider
    from agent_core.providers.ollama import OllamaProvider
    from agent_core.providers.openai_compatible import OpenAICompatibleProvider

    registry = ProviderRegistry()
    registry.register("openai-compatible", OpenAICompatibleProvider)
    registry.register("anthropic", AnthropicProvider)
    registry.register("ollama", OllamaProvider)
    return registry

