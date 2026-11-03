import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ON_VERCEL = bool(os.environ.get("VERCEL"))
# Vercel can only write to /tmp, which is wiped now and then: there it holds only caches (PDFs and
# figure crops, fetched again when missing), and the database must be on Turso (below).
DATA_DIR = Path(os.environ.get("APR_DATA_DIR") or ("/tmp/apr" if ON_VERCEL else ROOT / "data"))
PAPERS_DIR = DATA_DIR / "papers"
DB_PATH = DATA_DIR / "app.db"
STATIC_DIR = ROOT / "static"

# A Turso (hosted libSQL) database to use instead of the SQLite file at DB_PATH.
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

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

# Voice questions: speech-to-text models on OpenRouter (needs OPENROUTER_API_KEY). Each phrase gets
# its final text from TRANSCRIBE_MODEL once you pause after it (Whisper large-v3: ~0.7 s for a short
# phrase, about $0.0005 per minute of speech, and it follows the prompt with the paper's terms).
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "openai/whisper-large-v3")
# Drafts shown while a phrase is still being spoken: a faster, cheaper model.
TRANSCRIBE_DRAFT_MODEL = os.environ.get("TRANSCRIBE_DRAFT_MODEL", "openai/whisper-large-v3-turbo")
# Spoken language (ISO-639-1); empty detects it.
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "en") or None
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
