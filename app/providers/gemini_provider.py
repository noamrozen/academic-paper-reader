import uuid

from google import genai
from google.genai import types

from ..config import GEMINI_API_KEY
from .base import Provider, StepResult, TextDelta, ThinkingDelta, raw_for

_LOCAL_ID = "local_"  # prefix for call ids we generate when Gemini returns none
# Documented placeholder for function calls Gemini did not produce itself (e.g. history written by
# another provider); thinking models otherwise reject function calls without a thought signature.
_SKIP_SIGNATURE = b"skip_thought_signature_validator"


class GeminiProvider(Provider):
    id = "gemini"
    label = "Gemini"

    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    def available(self):
        return self._client is not None

    async def list_models(self):
        out = []
        async for m in await self._client.aio.models.list():
            if "generateContent" in (m.supported_actions or []) and "gemini" in m.name:
                out.append(m.name.removeprefix("models/"))
        return sorted(out, reverse=True)

    def _convert(self, messages, paper_text):
        contents = []
        for msg in messages:
            if msg["role"] == "assistant":
                raw = raw_for(msg, self.id)
                if raw is not None:
                    parts = [types.Part.model_validate(p) for p in raw]
                else:
                    parts = []
                    for b in msg["content"]:
                        if b["type"] == "text" and b["text"]:
                            parts.append(types.Part(text=b["text"]))
                        elif b["type"] == "tool_use":
                            parts.append(types.Part(
                                function_call=types.FunctionCall(
                                    name=b["name"], args=b["input"],
                                    id=None if b["id"].startswith(_LOCAL_ID) else b["id"]),
                                thought_signature=_SKIP_SIGNATURE))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif msg["role"] == "tool":
                parts = [
                    types.Part(function_response=types.FunctionResponse(
                        id=None if b["tool_use_id"].startswith(_LOCAL_ID) else b["tool_use_id"],
                        name=b["name"],
                        response={"error": b["content"]} if b.get("is_error") else {"output": b["content"]},
                    ))
                    for b in msg["content"]
                ]
                contents.append(types.Content(role="user", parts=parts))
            else:
                parts = [types.Part(text=paper_text if b["type"] == "paper" else b["text"]) for b in msg["content"]]
                contents.append(types.Content(role="user", parts=parts))
        return contents

    async def stream_step(self, *, model, system, paper_text, paper_id, messages, tools, reasoning=True):
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"])
                for t in tools
            ])],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        parts: list[types.Part] = []  # accumulated model output, signatures included
        usage_meta = None
        finish = ""
        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=self._convert(messages, paper_text), config=config)
        async for chunk in stream:
            if chunk.usage_metadata:
                usage_meta = chunk.usage_metadata
            if not chunk.candidates:
                continue
            cand = chunk.candidates[0]
            if cand.finish_reason:
                finish = str(cand.finish_reason.value if hasattr(cand.finish_reason, "value") else cand.finish_reason)
            for p in (cand.content.parts if cand.content and cand.content.parts else []):
                if p.function_call:
                    parts.append(p)
                elif p.text is not None:
                    if p.text:
                        yield (ThinkingDelta if p.thought else TextDelta)(p.text)
                    last = parts[-1] if parts else None
                    if (last is not None and last.text is not None and bool(last.thought) == bool(p.thought)
                            and not last.thought_signature):
                        last.text += p.text
                        if p.thought_signature:
                            last.thought_signature = p.thought_signature
                    else:
                        parts.append(p.model_copy())
                elif p.thought_signature and parts:
                    parts[-1].thought_signature = p.thought_signature

        neutral = []
        for p in parts:
            if p.function_call:
                fc = p.function_call
                neutral.append({"type": "tool_use", "id": fc.id or f"{_LOCAL_ID}{uuid.uuid4().hex[:8]}",
                                "name": fc.name, "input": dict(fc.args or {})})
            elif p.text and not p.thought:
                if neutral and neutral[-1]["type"] == "text":
                    neutral[-1]["text"] += p.text
                else:
                    neutral.append({"type": "text", "text": p.text})
        if not parts:
            neutral.append({"type": "text", "text": f"_Gemini returned no content (finish reason: {finish or 'unknown'})._"})

        um = usage_meta
        usage = {
            "input": (um.prompt_token_count or 0) if um else 0,
            "cached_input": (um.cached_content_token_count or 0) if um else 0,
            "cache_write": 0,
            "output": ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
        }
        raw = [p.model_dump(mode="json", exclude_none=True) for p in parts]
        yield StepResult(
            message={"role": "assistant", "content": neutral,
                     "raw": {"provider": self.id, "model": model, "content": raw}},
            usage=usage, stop_reason=finish,
        )
