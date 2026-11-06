import asyncio
import functools
import logging
import random

from ..config import DEFAULT_PROVIDER
from .anthropic_provider import AnthropicProvider
from .base import Provider, ProviderError, StepResult, TextDelta, ThinkingDelta
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


log = logging.getLogger(__name__)
RETRIES = 3
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}  # 529: Anthropic is overloaded
MAX_RETRY_WAIT = 30.0


def _status(e: Exception) -> int | None:
    """The HTTP status in an SDK's error: status_code (Anthropic, OpenAI, ProviderError) or code (Gemini)."""
    for attr in ("status_code", "code", "status"):
        value = getattr(e, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_wait(e: Exception, attempt: int) -> float:
    response = getattr(e, "response", None)
    retry_after = (getattr(response, "headers", None) or {}).get("retry-after", "")
    try:
        return min(float(retry_after), MAX_RETRY_WAIT)
    except ValueError:
        return min(2 ** (attempt + 1) + random.random(), MAX_RETRY_WAIT)


async def stream_step(provider: Provider, **kwargs):
    """provider.stream_step, tried again after a pause when the provider is rate limiting or
    overloaded, as long as nothing has streamed yet (after that, a retry would repeat the start of
    the answer). Out of credits (402) becomes a plain message."""
    for attempt in range(RETRIES + 1):
        started = False
        try:
            async for ev in provider.stream_step(**kwargs):
                started = True
                yield ev
            return
        except Exception as e:
            status = _status(e)
            if status == 402:
                raise ProviderError(f"{provider.label}: the account behind this API key is out of credits"
                                    " (top it up, or switch provider in Settings)", 402) from e
            if started or attempt == RETRIES or status not in RETRY_STATUSES:
                raise
            wait = _retry_wait(e, attempt)
            log.warning("%s answered %s; trying again in %.0f s", provider.label, status, wait)
            await asyncio.sleep(wait)


__all__ = ["PROVIDERS", "Provider", "ProviderError", "StepResult", "TextDelta", "ThinkingDelta", "default_provider",
           "provider_for", "stream_step"]
