import json

import openai

from ..config import OPENROUTER_API_KEY, OPENROUTER_MAX_TOKENS
from .base import Provider, StepResult, TextDelta, ThinkingDelta, raw_for, tool_result_text

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(Provider):
    """Any OpenRouter model via its OpenAI-compatible Chat Completions API.

    Reasoning is requested uniformly (`reasoning.enabled`) and the returned `reasoning_details` are
    replayed on the next call, which models with signed/encrypted reasoning need across tool calls.
    """

    id = "openrouter"
    label = "OpenRouter"

    def __init__(self):
        self._client = openai.AsyncOpenAI(
            base_url=BASE_URL, api_key=OPENROUTER_API_KEY,
            default_headers={"X-Title": "Academic Paper Reader"},
        ) if OPENROUTER_API_KEY else None

    def available(self):
        return self._client is not None

    async def list_models(self):
        models = [m async for m in self._client.models.list()]
        # Only models that support tool calling; batch variants can't stream.
        return sorted(m.id for m in models
                      if "tools" in ((m.model_extra or {}).get("supported_parameters") or [])
                      and not m.id.endswith(":batch"))

    def _convert(self, messages, paper_text, model):
        # Anthropic models need an explicit breakpoint; other providers cache prefixes automatically.
        cache = {"cache_control": {"type": "ephemeral"}} if model.startswith("anthropic/") else {}
        out = []
        for msg in messages:
            if msg["role"] == "assistant":
                # Reasoning signatures are model-specific, so only replay to the model that wrote them.
                raw = raw_for(msg, self.id) if (msg.get("raw") or {}).get("model") == model else None
                if raw is not None:
                    out.append(raw)
                    continue
                text = "".join(b["text"] for b in msg["content"] if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in msg["content"] if b["type"] == "tool_use"]
                out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
            elif msg["role"] == "tool":
                for b in msg["content"]:
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": tool_result_text(b)})
            else:
                parts = [{"type": "text", "text": paper_text, **cache} if b["type"] == "paper"
                         else {"type": "text", "text": b["text"]} for b in msg["content"]]
                out.append({"role": "user", "content": parts})
        return out

    async def stream_step(self, *, model, system, paper_text, paper_id, messages, tools, reasoning=True):
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *self._convert(messages, paper_text, model)],
            tools=[{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                     "parameters": t["parameters"]}} for t in tools],
            # OpenRouter reserves credit for max_tokens up front, so don't default to the model's maximum.
            max_tokens=OPENROUTER_MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"reasoning": {"enabled": reasoning}},
        )

        text, reasoning_details, calls = [], [], {}
        finish, usage = "", None
        async for chunk in stream:
            extra = chunk.model_extra or {}
            if extra.get("error"):
                err = extra["error"]
                raise RuntimeError(f"OpenRouter error: {err.get('message', err) if isinstance(err, dict) else err}")
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            dextra = delta.model_extra or {}
            if dextra.get("reasoning"):
                yield ThinkingDelta(dextra["reasoning"])
            for d in dextra.get("reasoning_details") or []:
                _merge_reasoning(reasoning_details, d)
            if delta.content:
                text.append(delta.content)
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    c["id"] = tc.id
                if tc.function and tc.function.name:
                    c["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    c["arguments"] += tc.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
                if finish == "error":
                    raise RuntimeError("OpenRouter: the upstream provider failed mid-response")

        finish = finish or "interrupted"  # the stream ended without a finish reason
        text = "".join(text)
        neutral = [{"type": "text", "text": text}] if text else []
        raw_calls = []
        # A truncated tool call must not run.
        for c in ([] if finish == "length" else [calls[i] for i in sorted(calls)]):
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": c["arguments"]}
            neutral.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": args})
            raw_calls.append({"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}})
        raw = {"role": "assistant", "content": text or None}
        if raw_calls:
            raw["tool_calls"] = raw_calls
        if reasoning_details:
            raw["reasoning_details"] = reasoning_details

        details = usage.prompt_tokens_details if usage else None
        dextra = (details.model_extra or {}) if details else {}
        yield StepResult(
            message={"role": "assistant", "content": neutral,
                     "raw": {"provider": self.id, "model": model, "content": raw}},
            usage={
                "input": usage.prompt_tokens if usage else 0,
                "cached_input": (details.cached_tokens or 0) if details else 0,
                "cache_write": dextra.get("cache_write_tokens") or 0,
                "output": usage.completion_tokens if usage else 0,
                "cost": (usage.model_extra or {}).get("cost") if usage else None,  # USD, as billed
            },
            stop_reason=finish,
        )


def _merge_reasoning(details: list[dict], d: dict):
    """Stream fragments of one reasoning block share type and index; join them into one block."""
    last = details[-1] if details else None
    if last and last.get("type") == d.get("type") and last.get("index") == d.get("index"):
        for key in ("text", "summary"):
            if d.get(key):
                last[key] = (last.get(key) or "") + d[key]
        for key, value in d.items():
            if key not in ("text", "summary") and value is not None:
                last[key] = value
    else:
        details.append(dict(d))
