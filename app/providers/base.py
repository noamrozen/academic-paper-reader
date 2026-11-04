"""Provider interface.

Conversations are stored in a provider-neutral format:

  {"role": "user",      "content": [{"type": "paper"}, {"type": "text", "text": ...}]}
  {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", "id", "name", "input"}],
                        "raw": {"provider": ..., "model": ..., "content": <native output>}}
  {"role": "tool",      "content": [{"type": "tool_result", "tool_use_id", "name", "content", "is_error"}]}

`{"type": "paper"}` is a placeholder that each provider expands into the same paper text, as the
first block of the first user message, so the prefix is identical across conversations. Search
sessions have no paper: no placeholder, an empty paper_text and paper_id None.

`raw` holds the provider's native output (thinking signatures, reasoning items, Gemini thought
signatures). When the same provider continues the conversation it is replayed verbatim; otherwise
the neutral blocks are used.
"""

from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class StepResult:
    message: dict  # neutral assistant message, including "raw"
    usage: dict = field(default_factory=dict)  # input, cached_input, cache_write, output
    stop_reason: str = ""


class Provider:
    """One instance per API key (see get_provider)."""

    id: str
    label: str

    def __init__(self, api_key: str):
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        raise NotImplementedError

    async def check_key(self):
        """Raises if the provider rejects this instance's key."""
        await self.list_models()

    def stream_step(
        self, *, model: str, system: str, paper_text: str, paper_id: str | None,
        messages: list[dict], tools: list[dict], reasoning: bool = True,
    ) -> AsyncIterator[TextDelta | ThinkingDelta | StepResult]:
        """One model call. Yields deltas, then exactly one StepResult. reasoning=False turns
        reasoning off where the provider can do that for any model (OpenRouter, Anthropic)."""
        raise NotImplementedError


def raw_for(msg: dict, provider: str) -> list | None:
    raw = msg.get("raw")
    if raw and raw.get("provider") == provider:
        return raw["content"]
    return None


def tool_result_text(block: dict) -> str:
    return block["content"] if not block.get("is_error") else f"Error: {block['content']}"
