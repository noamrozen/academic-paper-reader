"""Speech to text for voice questions: Whisper (or another speech-to-text model) on OpenRouter.

The browser does the listening (static/app.js): it tells speech from pauses, sends each phrase
here once you pause after it, for its final text, and the phrase still being spoken now and then
for a quick draft. For a question about a paper, the paper's title and its glossary terms go in
Whisper's prompt, which biases it toward spelling them the way the paper does ("GRPO", "KL").
"""

import base64
import io
import re
import wave

import httpx

from . import db, usage
from .config import (HTTP_HEADERS, OPENROUTER_API_KEY, TRANSCRIBE_DRAFT_MODEL, TRANSCRIBE_LANGUAGE,
                     TRANSCRIBE_MODEL)

URL = "https://openrouter.ai/api/v1/audio/transcriptions"
SAMPLE_RATE = 16000  # of the PCM the browser sends (static/mic-worklet.js)
PROMPT_CHARS = 600  # Whisper reads at most ~224 prompt tokens
CONTEXT_CHARS = 200  # of the question so far, appended to the prompt for the next phrase
# OpenRouter forwards a prompt only as a provider's own option, and picks the provider itself:
# these are the ones serving Whisper (DeepInfra, the cheapest, usually gets it).
PROMPT_PROVIDERS = ("deepinfra", "groq", "together")
_SPOKEN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9@'-]*(?: [A-Za-z0-9@'-]+){0,3}")  # not symbols like θ or d_k


def available() -> bool:
    return bool(OPENROUTER_API_KEY)


def paper_prompt(paper_id: str | None) -> str | None:
    paper = paper_id and db.get_paper(paper_id)
    if not paper:
        return None
    glossary = db.get_analysis(paper_id, "glossary") or {}
    terms, seen = [], set()
    for t in glossary.get("entries") or {}:
        if _SPOKEN_TERM.fullmatch(t) and sum(c.isalpha() for c in t) > 1 and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)
    # Acronyms first: they're what Whisper most often gets wrong ("GRPO" heard as "GRPL").
    terms.sort(key=lambda t: not any(sum(c.isupper() for c in w) > 1 for w in re.split(r"[\s-]+", t)))
    prompt = f"A question about the paper \"{paper['title']}\"."
    if terms:
        prompt += " Terms: " + ", ".join(terms)
    return prompt[:PROMPT_CHARS].rsplit(",", 1)[0] if len(prompt) > PROMPT_CHARS else prompt


def _wav(pcm16: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16)
    return buf.getvalue()


async def transcribe(pcm16: bytes, *, paper_id: str | None, context: str = "", draft: bool = False) -> str:
    """One phrase of a spoken question (16 kHz 16-bit mono PCM). context is the question's text so
    far. A draft is replaced soon anyway: a faster model, and no prompt, which a draft model is apt
    to read out as if it had been spoken."""
    model = TRANSCRIBE_DRAFT_MODEL if draft else TRANSCRIBE_MODEL
    body = {"model": model, "input_audio": {"data": base64.b64encode(_wav(pcm16)).decode(), "format": "wav"},
            "temperature": 0}
    if TRANSCRIBE_LANGUAGE:
        body["language"] = TRANSCRIBE_LANGUAGE
    if not draft:
        prompt = " ".join(p for p in (paper_prompt(paper_id), context[-CONTEXT_CHARS:]) if p)
        if prompt:
            body["provider"] = {"options": {p: {"prompt": prompt} for p in PROMPT_PROVIDERS}}
    # A phrase takes about a second; now and then a request stalls, and a second try is quicker.
    timeout = 8 + len(pcm16) / (2 * SAMPLE_RATE) / 4
    async with httpx.AsyncClient(timeout=timeout, headers=HTTP_HEADERS) as client:
        for attempt in range(2):
            try:
                r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
            except httpx.TimeoutException:
                if attempt:
                    raise RuntimeError(f"{model}: no answer from OpenRouter in {timeout:.0f} s") from None
                continue
            if r.status_code < 500 or attempt:
                break
    if r.is_error:
        try:
            message = r.json()["error"]["message"]
        except Exception:
            message = r.text[:200]
        raise RuntimeError(f"{model}: {message}")
    data = r.json()
    u = data.get("usage") or {}
    await usage.record("openrouter", model, {
        "input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0), "cost": u.get("cost"),
    }, kind="transcription", paper_id=paper_id)
    return (data.get("text") or "").strip()
