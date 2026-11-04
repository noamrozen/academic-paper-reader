import functools

from ..config import DEFAULT_PROVIDER
from .anthropic_provider import AnthropicProvider
from .base import Provider, StepResult, TextDelta, ThinkingDelta
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider

# Provider classes by id. Keys belong to users (auth.user_keys), so instances are made per key.
PROVIDERS: dict[str, type[Provider]] = {p.id: p for p in (AnthropicProvider, OpenAIProvider, GeminiProvider,
                                                          OpenRouterProvider)}


@functools.lru_cache(maxsize=256)
def get_provider(provider_id: str, api_key: str) -> Provider:
    """The provider's client for this key, kept so that its connections are reused."""
    return PROVIDERS[provider_id](api_key)


def provider_for(keys: dict[str, str], provider_id: str) -> Provider | None:
    """The provider with the key in keys ({provider id: API key}), or None without one."""
    return get_provider(provider_id, keys[provider_id]) if provider_id in PROVIDERS and keys.get(provider_id) else None


def default_provider(keys: dict[str, str]) -> Provider | None:
    """The configured DEFAULT_PROVIDER if keys has its key, otherwise the first provider that has one."""
    for provider_id in (DEFAULT_PROVIDER, *PROVIDERS):
        if p := provider_for(keys, provider_id):
            return p
    return None


__all__ = ["PROVIDERS", "Provider", "StepResult", "TextDelta", "ThinkingDelta", "default_provider", "provider_for"]
