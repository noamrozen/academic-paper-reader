"""Speech to text for voice questions: Whisper, run locally on the CPU by faster-whisper.

The model loads on first use (and is downloaded once, to the Hugging Face cache). For a question
about a paper, the paper's title and its glossary terms go in Whisper's initial prompt, which
biases it toward spelling them the way the paper does ("GRPO", "chain-of-thought", "KL").

Questions are transcribed live, while they're spoken (LiveTranscript): the browser streams the
microphone as 16 kHz PCM, Silero VAD (bundled with faster-whisper) finds the speech in it, and
- the phrase still being spoken gets a quick draft from a smaller model, redone as more arrives;
- a phrase followed by a pause is transcribed once, at full quality, and kept. Since that happens
  during the silence that ends the question, its text is ready when the recording stops;
- a long enough silence after speech ends the question, so the browser stops recording.
"""

import re
import threading

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from . import db
from .config import VOICE_END_SILENCE, WHISPER_DRAFT_MODEL, WHISPER_LANGUAGE, WHISPER_MODEL

PROMPT_CHARS = 600  # Whisper reads at most ~224 prompt tokens
_SPOKEN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9@'-]*(?: [A-Za-z0-9@'-]+){0,3}")  # not symbols like θ or d_k

_models: dict[str, WhisperModel] = {}
_lock = threading.Lock()


def _get_model(name: str) -> WhisperModel:
    with _lock:
        if name not in _models:
            _models[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _models[name]


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


SAMPLE_RATE = 16000
PAUSE = 0.6  # s of silence that closes a phrase (its text is then final)
NO_SPEECH = 8.0  # s with no speech at all before giving up
MIN_DRAFT = 0.4  # s of speech before a phrase gets a draft
MAX_AUDIO = 120.0  # s per question
PAD = int(0.2 * SAMPLE_RATE)  # kept around detected speech, so word edges aren't clipped
CONTEXT_CHARS = 200  # of the question so far, appended to the prompt for the next phrase
_VAD = VadOptions(min_silence_duration_ms=150, speech_pad_ms=0, min_speech_duration_ms=120)


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


class LiveTranscript:
    """One spoken question, fed audio as it's recorded. add() runs on the event loop; step() and
    finish() do the heavy work and run on a worker thread, one at a time."""

    def __init__(self, prompt: str | None):
        self.prompt = prompt
        self._chunks: list[np.ndarray] = []
        self._chunks_lock = threading.Lock()
        self.audio = np.zeros(0, np.float32)
        self.done = 0  # samples whose text is final
        self.final = ""  # their text
        self.draft = ""  # text of the phrase being spoken
        self.speech_end = None  # sample where the latest speech ended (None: none heard yet)
        self._stepped = 0  # audio length at the last step

    def add(self, pcm16: bytes):
        with self._chunks_lock:
            self._chunks.append(np.frombuffer(pcm16, np.int16))

    @property
    def text(self) -> str:
        return _join(self.final, self.draft)

    def step(self) -> str | None:
        """Process the audio added since the last step: "end" when the question is over (after a
        silence, or none spoken, or too long), "text" when self.text changed, else None."""
        self._take()
        if len(self.audio) - self._stepped < 0.25 * SAMPLE_RATE:
            return None
        self._stepped = len(self.audio)
        if len(self.audio) >= MAX_AUDIO * SAMPLE_RATE:
            return "end"
        rest = self.audio[self.done:]
        regions = [(r["start"], r["end"]) for r in get_speech_timestamps(rest, _VAD)]
        if regions:
            self.speech_end = self.done + regions[-1][1]
        if self.speech_end is None:
            return "end" if len(self.audio) >= NO_SPEECH * SAMPLE_RATE else None
        if len(self.audio) - self.speech_end >= VOICE_END_SILENCE * SAMPLE_RATE:
            return "end"
        if not regions:
            return None  # a pause after a phrase that's already final
        # Close every phrase followed by a pause, the last one too once the speaker has paused
        # after it; only what's still being spoken stays a draft.
        if len(rest) - regions[-1][1] >= PAUSE * SAMPLE_RATE:
            cut = len(rest)
        else:
            cut = next((
                (a_end + b_start) // 2
                for (_, a_end), (b_start, _) in reversed(list(zip(regions, regions[1:])))
                if b_start - a_end >= PAUSE * SAMPLE_RATE
            ), None)
        if cut is not None:
            self.final = _join(self.final, self._transcribe(rest[:cut], [r for r in regions if r[1] <= cut], WHISPER_MODEL))
            self.done += cut
            rest = rest[cut:]
            regions = [(a - cut, b - cut) for a, b in regions if a >= cut]
        # Whisper guesses wildly from a fraction of a word ("," or "What does this").
        heard = sum(b - a for a, b in regions) >= MIN_DRAFT * SAMPLE_RATE
        self.draft = self._transcribe(rest, regions, WHISPER_DRAFT_MODEL) if heard else ""
        return "text"

    def finish(self) -> str:
        """The whole question, final: the last phrase transcribed at full quality."""
        self._take()
        rest = self.audio[self.done:]
        regions = [(r["start"], r["end"]) for r in get_speech_timestamps(rest, _VAD)]
        self.draft = ""
        self.final = _join(self.final, self._transcribe(rest, regions, WHISPER_MODEL))
        return self.final

    def _take(self):
        with self._chunks_lock:
            chunks, self._chunks = self._chunks, []
        if chunks:
            new = np.concatenate(chunks).astype(np.float32) / 32768
            self.audio = np.concatenate([self.audio, new])

    def _transcribe(self, audio: np.ndarray, regions: list[tuple[int, int]], model: str) -> str:
        if not regions:
            return ""  # Whisper makes words up from silence ("Thank you.")
        audio = audio[max(0, regions[0][0] - PAD):regions[-1][1] + PAD]
        prompt = _join(self.prompt, self.final[-CONTEXT_CHARS:]) or None
        # A draft is replaced soon anyway: one greedy pass, without Whisper's retries at higher
        # temperatures (which can take seconds), and no longer than the speech can hold.
        draft = {"beam_size": 1, "temperature": 0.0, "max_new_tokens": int(len(audio) / SAMPLE_RATE * 6) + 10} \
            if model != WHISPER_MODEL else {"beam_size": 5}
        segments, _ = _get_model(model).transcribe(
            audio, language=WHISPER_LANGUAGE, initial_prompt=prompt,
            condition_on_previous_text=False, without_timestamps=True, **draft,
        )
        return " ".join(s.text.strip() for s in segments).strip()
