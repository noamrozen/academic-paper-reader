import re

import anthropic

from ..config import CLAUDE_PAPER_CACHE_TTL
from .base import Provider, StepResult, TextDelta, ThinkingDelta, raw_for

# Models that reject adaptive thinking (pre-4.6 generations).
_NO_ADAPTIVE = re.compile(r"claude-3|haiku-4-5|-4-5|-4-1|-4-0|-4-2\d")
# Models that take the server-side refusal fallback.
_FALLBACKS = ("claude-opus-5", "claude-fable")


class AnthropicProvider(Provider):
    id = "anthropic"
    label = "Claude"

    def __init__(self, api_key: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def list_models(self):
        return [m.id async for m in self._client.models.list(limit=100)]

    def _convert(self, messages, paper_text):
        out = []
        for msg in messages:
            if msg["role"] == "assistant":
                content = raw_for(msg, self.id) or [
                    {"type": "text", "text": b["text"]} if b["type"] == "text"
                    else {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]}
                    for b in msg["content"] if b["type"] in ("text", "tool_use") and (b.get("text") or b["type"] == "tool_use")
                ]
                role = "assistant"
            elif msg["role"] == "tool":
                content = [
                    {"type": "tool_result", "tool_use_id": b["tool_use_id"], "content": b["content"],
                     "is_error": bool(b.get("is_error"))}
                    for b in msg["content"]
                ]
                role = "user"
            else:
                content = []
                for b in msg["content"]:
                    if b["type"] == "paper":
                        # Explicit breakpoint at the end of the shared prefix (tools + system + paper).
                        content.append({"type": "text", "text": paper_text,
                                        "cache_control": {"type": "ephemeral", "ttl": CLAUDE_PAPER_CACHE_TTL}})
                    else:
                        content.append({"type": "text", "text": b["text"]})
                role = "user"
            if not content:
                continue
            if out and out[-1]["role"] == role:  # e.g. a follow-up after an interrupted turn
                out[-1]["content"] = list(out[-1]["content"]) + content
            else:
                out.append({"role": role, "content": content})
        return out

    async def stream_step(self, *, model, system, paper_text, paper_id, messages, tools, reasoning=True):
        kwargs = dict(
            model=model,
            max_tokens=32000,
            system=system,
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                   for t in tools],
            messages=self._convert(messages, paper_text),
            # Automatic breakpoint for the growing conversation tail (5m; after the 1h paper marker).
            cache_control={"type": "ephemeral"},
        )
        if reasoning and not _NO_ADAPTIVE.search(model):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if model.startswith(_FALLBACKS):
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        async with self._client.beta.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield TextDelta(event.delta.text)
                    elif event.delta.type == "thinking_delta":
                        yield ThinkingDelta(event.delta.thinking)
            final = await stream.get_final_message()

        raw, neutral = [], []
        for b in final.content:
            if b.type == "text":
                raw.append({"type": "text", "text": b.text})
                neutral.append({"type": "text", "text": b.text})
            elif b.type == "thinking":
                raw.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
            elif b.type == "redacted_thinking":
                raw.append({"type": "redacted_thinking", "data": b.data})
            elif b.type == "tool_use":
                raw.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                neutral.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        stop = final.stop_reason or ""
        if stop == "refusal":
            neutral.append({"type": "text", "text": "\n\n_The model declined to answer this request._"})
        if stop == "max_tokens":
            # A truncated tool call must not run.
            neutral = [b for b in neutral if b["type"] != "tool_use"]
            raw = [b for b in raw if b["type"] != "tool_use"]
        u = final.usage
        usage = {
            "input": (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
            "cached_input": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
            # Writes with the 1h TTL (the paper block, by default) cost more than 5m ones.
            "cache_write_1h": getattr(getattr(u, "cache_creation", None), "ephemeral_1h_input_tokens", 0) or 0,
            "output": u.output_tokens or 0,
        }
        yield StepResult(
            message={"role": "assistant", "content": neutral,
                     "raw": {"provider": self.id, "model": final.model, "content": raw}},
            usage=usage, stop_reason=stop,
        )
