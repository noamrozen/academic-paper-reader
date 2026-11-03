import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("APR_DATA_DIR", ROOT / "data"))
PAPERS_DIR = DATA_DIR / "papers"
DB_PATH = DATA_DIR / "app.db"
STATIC_DIR = ROOT / "static"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Provider selected for new conversations when it has a key; otherwise the first one that does.
DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "openrouter")

DEFAULT_MODELS = {
    "anthropic": os.environ.get("ANTHROPIC_DEFAULT_MODEL", "claude-opus-5"),
    "openai": os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-5"),
    "gemini": os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-3.8-flash"),
    "openrouter": os.environ.get("OPENROUTER_DEFAULT_MODEL", "deepseek/deepseek-v4.1-flash"),
}

# OpenRouter reserves credit for the full max_tokens before each call.
OPENROUTER_MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "16384"))

CLAUDE_PAPER_CACHE_TTL = os.environ.get("CLAUDE_PAPER_CACHE_TTL", "1h")

# Voice questions: a Whisper model run locally by faster-whisper (downloaded on first use).
# "small.en" transcribes a short question in about two seconds on an Apple M-series CPU;
# "large-v3-turbo" is more robust to accents and noise but around three times slower.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")
# Drafts shown while a phrase is still being spoken: a smaller, faster model (~0.4 s per draft).
WHISPER_DRAFT_MODEL = os.environ.get("WHISPER_DRAFT_MODEL", "base.en")
# Spoken language; empty detects it (needs a multilingual model, i.e. one without ".en").
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en") or None
# Seconds of silence after speaking that end a voice question (recording stops by itself).
VOICE_END_SILENCE = float(os.environ.get("VOICE_END_SILENCE", "1.5"))

# Optional spending budgets (USD) for this app's model calls per day, week (from Monday) and
# month; the Usage panel shows how much of each is used.
BUDGETS = {period: float(v) for period, v in (
    ("day", os.environ.get("BUDGET_DAILY")), ("week", os.environ.get("BUDGET_WEEKLY")),
    ("month", os.environ.get("BUDGET_MONTHLY")),
) if v}

# Maximum number of model calls (tool-use rounds) per user turn.
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "12"))

HTTP_HEADERS = {"User-Agent": "academic-paper-reader/0.1 (+https://github.com/)"}

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
