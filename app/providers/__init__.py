from ..config import DEFAULT_PROVIDER
from .anthropic_provider import AnthropicProvider
from .base import Provider, StepResult, TextDelta, ThinkingDelta
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider

PROVIDERS: dict[str, Provider] = {p.id: p for p in (AnthropicProvider(), OpenAIProvider(), GeminiProvider(),
                                              OpenRouterProvider())}



def default_provider() -> Provider | None:
    """The configured DEFAULT_PROVIDER if it has a key, otherwise the first provider that does."""
    p = PROVIDERS.get(DEFAULT_PROVIDER)
    if p and p.available():
        return p
    return next((p for p in PROVIDERS.values() if p.available()), None)


__all__ = ["PROVIDERS", "Provider", "StepResult", "TextDelta", "ThinkingDelta", "default_provider"]
