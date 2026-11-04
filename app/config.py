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

# A Turso (hosted libSQL) database to use instead of the SQLite file at DB_PATH. Only a production
# deployment on Vercel uses it by itself: a local run, `vercel dev` or a preview deployment that has
# the production database's settings (in .env, or from a Vercel integration connected to every
# environment) would read it, and migrate it to a schema the deployed app doesn't know. Anywhere
# else, USE_TURSO=1 opts in (e.g. for a preview's own database).
USE_TURSO = os.environ.get("VERCEL_ENV") == "production" or os.environ.get("USE_TURSO") == "1"
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL") if USE_TURSO else None
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN") if USE_TURSO else None

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# The keys above are the server's: only admins' requests use them. Other users add their own
# (stored encrypted, see crypto.py), under the same names.
SERVER_KEYS = {name: key for name, key in (
    ("anthropic", ANTHROPIC_API_KEY), ("openai", OPENAI_API_KEY), ("gemini", GEMINI_API_KEY),
    ("openrouter", OPENROUTER_API_KEY), ("tavily", TAVILY_API_KEY),
) if key}

# Accounts. An admin is a user whose verified email is listed here (comma-separated).
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
# Signs nothing by itself, but encrypts the API keys users save: changing it loses them. Required on
# Vercel; a local run makes one in the data directory.
SECRET_KEY = os.environ.get("SECRET_KEY")
# "Sign in with Google": an OAuth client of type "Web application" (see "Accounts" in the README).
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
# The app's address, for links in emails and Google's redirect; by default, the address a request came to.
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
# Emails (verifying an address, resetting a password) go through Resend if RESEND_API_KEY is set, else
# through SMTP if SMTP_HOST is; with neither, their links are only written to the log.
EMAIL_FROM = os.environ.get("EMAIL_FROM", "Paper Reader <onboarding@resend.dev>")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

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

# Without it a deployment would quietly use a SQLite file in /tmp, which each instance has its own
# of: data, and sign-ins, would vanish between requests.
if ON_VERCEL and USE_TURSO and not TURSO_DATABASE_URL:
    raise RuntimeError("Set TURSO_DATABASE_URL (and TURSO_AUTH_TOKEN) for this environment, or unset USE_TURSO")

if not SECRET_KEY:
    if ON_VERCEL:
        raise RuntimeError("Set SECRET_KEY in the Vercel project's environment variables"
                           " (e.g. the output of: python3 -c 'import secrets; print(secrets.token_urlsafe(32))')")
    _secret_file = DATA_DIR / "secret_key"
    if not _secret_file.exists():
        import secrets
        _secret_file.write_text(secrets.token_urlsafe(32))
    SECRET_KEY = _secret_file.read_text().strip()
