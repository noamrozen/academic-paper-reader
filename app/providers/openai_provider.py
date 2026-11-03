import json
import re

import openai

from ..config import OPENAI_API_KEY
from .base import Provider, StepResult, TextDelta, ThinkingDelta, raw_for

_REASONING = re.compile(r"^(o\d|gpt-5)")
_CHAT_MODEL = re.compile(r"^(gpt-|o\d|chatgpt-)")
_NOT_CHAT = re.compile(r"audio|realtime|transcribe|tts|image|search|embedding|instruct|codex")


class OpenAIProvider(Provider):
    """OpenAI via the Responses API (stateless: store=False, reasoning items replayed encrypted)."""

    id = "openai"
    label = "OpenAI"

    def __init__(self):
        self._client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def available(self):
        return self._client is not None

    async def list_models(self):
        ids = [m.id async for m in self._client.models.list()]
        return sorted((i for i in ids if _CHAT_MODEL.match(i) and not _NOT_CHAT.search(i)), reverse=True)

    def _convert(self, messages, paper_text):
        items = []
        for msg in messages:
            if msg["role"] == "assistant":
                raw = raw_for(msg, self.id)
                if raw is not None:
                    items.extend(raw)
                    continue
                for b in msg["content"]:
                    if b["type"] == "text" and b["text"]:
                        items.append({"type": "message", "role": "assistant",
                                      "content": [{"type": "output_text", "text": b["text"]}]})
                    elif b["type"] == "tool_use":
                        items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                      "arguments": json.dumps(b["input"])})
            elif msg["role"] == "tool":
                for b in msg["content"]:
                    items.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                                  "output": b["content"] if not b.get("is_error") else f"Error: {b['content']}"})
            else:
                parts = [{"type": "input_text", "text": paper_text if b["type"] == "paper" else b["text"]}
                         for b in msg["content"]]
                items.append({"role": "user", "content": parts})
        return items

    async def stream_step(self, *, model, system, paper_text, paper_id, messages, tools, reasoning=True):
        kwargs = dict(
            model=model,
            instructions=system,
            input=self._convert(messages, paper_text),
            tools=[{"type": "function", "name": t["name"], "description": t["description"],
                    "parameters": t["parameters"], "strict": False} for t in tools],
            stream=True,
            store=False,
            # Routes requests sharing the paper prefix (or the search prompt) to the same cache.
            prompt_cache_key=f"paper:{paper_id}" if paper_id else "search",
        )
        if _REASONING.match(model):
            kwargs["reasoning"] = {"summary": "auto"}
            kwargs["include"] = ["reasoning.encrypted_content"]

        final = None
        stream = await self._client.responses.create(**kwargs)
        async for ev in stream:
            if ev.type == "response.output_text.delta":
                yield TextDelta(ev.delta)
            elif ev.type == "response.reasoning_summary_text.delta":
                yield ThinkingDelta(ev.delta)
            elif ev.type == "response.reasoning_summary_part.done":
                yield ThinkingDelta("\n\n")
            elif ev.type in ("response.completed", "response.incomplete"):
                final = ev.response
            elif ev.type == "response.failed":
                err = ev.response.error
                raise RuntimeError(f"OpenAI response failed: {err.message if err else 'unknown error'}")
            elif ev.type == "error":
                raise RuntimeError(f"OpenAI error: {ev.message}")
        if final is None:
            raise RuntimeError("OpenAI stream ended without a response")

        raw, neutral = [], []
        for item in final.output:
            if item.type == "reasoning":
                raw.append({"type": "reasoning", "id": item.id,
                            "summary": [{"type": "summary_text", "text": s.text} for s in item.summary],
                            "encrypted_content": item.encrypted_content})
            elif item.type == "message":
                text = "".join(c.text for c in item.content if c.type == "output_text")
                refusal = "".join(c.refusal for c in item.content if c.type == "refusal")
                text = text or (f"_The model declined: {refusal}_" if refusal else "")
                raw.append({"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}]})
                neutral.append({"type": "text", "text": text})
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_invalid_json": item.arguments}
                raw.append({"type": "function_call", "call_id": item.call_id, "name": item.name,
                            "arguments": item.arguments})
                neutral.append({"type": "tool_use", "id": item.call_id, "name": item.name, "input": args})

        u = final.usage
        usage = {
            "input": u.input_tokens if u else 0,
            "cached_input": (u.input_tokens_details.cached_tokens or 0) if u and u.input_tokens_details else 0,
            "cache_write": 0,
            "output": u.output_tokens if u else 0,
        }
        yield StepResult(
            message={"role": "assistant", "content": neutral,
                     "raw": {"provider": self.id, "model": model, "content": raw}},
            usage=usage, stop_reason=final.status or "",
        )
