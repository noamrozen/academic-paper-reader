# Academic Paper Reader

Read arXiv papers next to an LLM research assistant. Paste a link, highlight a passage, and ask about it.

- Paste an arXiv link or ID (`https://arxiv.org/abs/1706.03762`, `arxiv.org/pdf/1706.03762v7`, `1706.03762`, or any direct PDF URL) and press Enter. The paper is downloaded, its text is extracted, and the PDF is rendered with selectable text.
- Or type a paper's **name** and press Enter to search arXiv and [OpenAlex](https://openalex.org)
  (no key needed). Pick a result with the mouse or ↑/↓ and Enter to open it. A result opens from arXiv
  when there's an arXiv version; otherwise from an open-access PDF, checked to actually download (badge
  **PDF**). Results with neither are shown greyed out as **unsupported**; ↗ opens any result's page
  in a new tab.
- Highlight text in the paper and a popover asks for your question. **Enter** starts a new conversation about the passage. **⌘/Ctrl+Enter** asks in the current conversation. Clicking into the chat box instead attaches the highlight to your next message.
- When a paper is ingested, the default model also builds three lookups in the background: a
  **notation glossary** (symbols, acronyms, defined terms), an index of **numbered elements**
  (definitions, theorems, lemmas, equations, tables, figures, sections, appendices) and the
  **bibliography**. References in the PDF are then **linked automatically**: click "Thm. 3.1",
  "Eq. (5)", "§3.2", a citation like "[18]" or "Kalchbrenner et al., 2016", or a symbol, and a popover
  shows the referenced statement, the definition, or the cited paper (title, authors, venue). Theorems,
  lemmas, propositions, corollaries and claims proved in the paper also get **Go to proof**, which
  scrolls to where the proof starts (even in an appendix) and briefly highlights it. Papers indexed
  before this existed are re-indexed once, the next time they're opened. A
  cited paper's title opens it in the reader with a fresh session; its arXiv ID is found by title
  search on first use and cached. "Figure N"/"Table N" show the figure or table itself, cropped from
  the PDF. Selecting any text also looks it up, which covers terms the automatic links skip (plain
  words, references split oddly by the PDF's text layer). The **Notation** button lists the whole
  glossary, and **Elements** lists every numbered element in page order (filter by text or kind, with
  figure and table crops, and **Go to page** / **Go to proof**). **Summarize** starts a new chat that
  asks for a summary of the paper and streams the answer. The lookups stream in: links appear as items arrive, before the model finishes. A reply
  cut off at the model's output limit is continued from the items it produced. Click the
  **Analyzing paper** pill to see each lookup's progress; one that still fails stays listed there
  (as **Analysis incomplete**), with **Continue** to pick up from the kept items or **Retry**.
- **Ask by voice:** the 🎙 button next to Send (and in the highlight popover) records a question.
  It's transcribed while you speak, so the text grows in the box as you talk, and recording stops
  by itself once you've been quiet for 1.5 s (`VOICE_END_SILENCE`), or when you click 🎙 again.
  The text isn't sent, so you can check it first. The browser tells speech from pauses and sends
  each phrase to the server once you pause after it, where [Whisper](https://github.com/openai/whisper)
  large-v3 on [OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/stt.md)
  (`TRANSCRIBE_MODEL`) gives it its final text; the phrase still being spoken gets quick drafts from
  `whisper-large-v3-turbo` (`TRANSCRIBE_DRAFT_MODEL`). It needs `OPENROUTER_API_KEY` (the button is
  hidden without one) and costs about $0.0005 per minute of speech, shown under Usage. For a
  question about a paper, Whisper is primed with the paper's title and glossary terms, so acronyms
  and method names tend to come out spelled as the paper spells them. The browser only allows the
  microphone over HTTPS or on localhost.
- **arXiv versions:** a paper stays the arXiv version it was added as: its PDF's URL (with the
  version) and SHA-256 are stored, and a copy downloaded again must match, so the PDF always fits
  the stored text, notation and elements. When arXiv has a newer version, a notice above the paper
  offers it: **Get v*N*** processes it like a new paper in the old one's place, and moves the old
  version, with its conversations, to the **Archive** folder (as `<id>v<n>`).
- Ask follow-ups in the chat panel. Every conversation is saved in the sidebar under its paper.
- The home page is a **search session**: a chat with no paper in context, where you ask the assistant to
  find papers (a topic, a half-remembered title, related work). It searches arXiv and the web and links
  each paper it finds; clicking a link opens the paper in the reader, and Back returns to the search.
  Past searches are listed under the sidebar's **Searches** tab, next to **Papers**; the sidebar switches
  to the tab that matches what's open. **+ Search** (or the **Paper Reader** title) starts a new one.
- Organize the sidebar with **folders** (+ Folder; double-click a name to rename it). Drag a paper onto
  a folder, or use its ⇄ menu. Click a folder's name or ▾/▸ (Unfiled too) to fold it. Clicking a paper opens it with
  its chats unfolded; clicking the paper that's already open folds or unfolds its chats. A folded paper
  shows its chat count.
- Delete a paper with its × (on hover); this also deletes its chats, its PDF and cached figure crops.
  Deleting a folder that holds papers asks whether to keep them (they become unfiled) or delete them too.
- Minimize the chat panel with » (or ⌘/Ctrl+`\`) to read the PDF at full width. It reopens when you ask
  from a highlight.
- Pick the provider (Claude, OpenAI, Gemini, OpenRouter) and model per conversation, under **Settings** (⚙) at the foot of the sidebar; it applies to every chat, about a paper or a search. Its tooltip shows the current choice, and each answer shows the model that wrote it. OpenRouter gives one key access to models from every vendor; the model field lists all its tool-capable models.
- **Usage** (next to Settings) shows what the model calls have cost today, this week or this month, with a
  chart of recent days/weeks/months and a breakdown by model. Paper analyses count too, and spending stays
  recorded when a paper or chat is deleted. OpenRouter calls use the cost OpenRouter billed; other providers'
  calls are priced from each model's token prices in OpenRouter's public model catalog (cache reads and writes
  included). Set `BUDGET_DAILY` / `BUDGET_WEEKLY` / `BUDGET_MONTHLY` (USD) to see how much of a budget is used;
  an OpenRouter key's own credit limit is shown too. A dot on the button warns at 80% of any budget.
- The assistant is agentic. It can search the web, fetch pages, search arXiv, and read cited papers. Any paper it finds shows up as a card with **Open** (opens it in the reader with a fresh session) and **↗** (opens it in a new browser tab). Inline arXiv links in answers get an `open` button too.
- When the assistant mentions another paper (e.g. one the current paper cites), it links it as
  `[Title](paper:<arXiv ID or PDF URL>)`, looking each ID up with `search_arxiv` first. Clicking the link
  opens that paper in the reader with a fresh session (downloading it if it's new); **↗** opens it in a
  new tab. Each link also shows the title arXiv has for its ID whenever that differs from the link text,
  so a wrong ID is visible at a glance. Any page can link `#/open/<arXiv ID or PDF URL>` the same way.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # add at least one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY
uv sync
uv run uvicorn app.main:app --port 8000
```

Then open http://localhost:8000.

Web search uses DuckDuckGo by default, with no key. Set `TAVILY_API_KEY` for better results.

### Deploying to Vercel

Vercel runs the app as a function whose files are read-only except `/tmp`, which is wiped now and
then, so the database lives on [Turso](https://turso.tech) (hosted SQLite):

1. In the Vercel project, **Storage → Create Database → Turso Cloud**, connected to the project
   for all environments. It adds `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` to the project.
2. Put the same two variables in `.env` (a local run then uses that database too).
3. Copy the local library into it: `uv run python scripts/migrate_to_turso.py`.
4. Add the provider keys (`OPENROUTER_API_KEY`, ...) to the project's environment variables, and
   redeploy.

On Vercel, `/tmp` only caches PDFs and figure crops, downloaded again (and checked against their
SHA-256) when missing; arXiv PDFs go to the browser straight from arXiv, since a Vercel response is
capped at 4.5 MB.

## Prefix caching design

Every conversation about a paper sends the same prefix: the tool definitions, then a fixed system prompt, then a first user message that opens with the paper.

```
tools (fixed) → system prompt (fixed) → user: [ <paper>…full text…</paper> ][ <paper_elements> ][ highlighted passage + question ] → …
```

- The paper text is extracted once at ingest and **frozen in SQLite**, so the paper block is byte-identical in every conversation about that paper. The prompt and tool list contain nothing per-request (no dates, no IDs).
- The paper is a separate content block. The highlight and question come after it, so they never break the shared prefix.
- `<paper_elements>` lists the paper's numbered elements (link id and label, no contents) so answers can link them as `[Theorem 3.1](ref:theorem-3.1)`; clicking one scrolls the PDF there. It goes after the paper, not in the system prompt: it depends on the paper and is still growing while the index is built, and anything before the paper that changes would miss the paper's cache. It's stored with the conversation's first message, so it stays fixed for that conversation (one started before indexing finished has a partial list, or none).
- **Claude:** an explicit `cache_control` breakpoint on the paper block (1h TTL by default, `CLAUDE_PAPER_CACHE_TTL`), plus automatic caching for the growing conversation tail.
- **OpenAI:** automatic prefix caching, with `prompt_cache_key=paper:<id>` so requests for the same paper hit the same cache.
- **Gemini:** implicit (automatic) prefix caching.
- **OpenRouter:** a `cache_control` breakpoint on the paper block for `anthropic/*` models; other upstreams cache automatically.
- Each answer shows `prompt N tokens (X% cached)`, so you can see cache hits directly.

Caches are scoped per model, so switching models mid-conversation means one cache miss.

Search sessions have no paper block: they send the tools, a fixed search prompt
(`SEARCH_SYSTEM_PROMPT`, sharing the paper-link instructions with the paper prompt) and the chat
(`prompt_cache_key=search` on OpenAI). They're stored as conversations with a `NULL` `paper_id`.

## Architecture

```
app/
  main.py           FastAPI routes; /api/chat streams Server-Sent Events
  agent.py          agent loop: model call → run tools in parallel → repeat (max MAX_AGENT_STEPS)
  papers.py         arXiv ID parsing, metadata (arXiv API), search by name (arXiv + OpenAlex),
                    PDF download, text extraction (PyMuPDF)
  analysis.py       per-paper glossary, numbered-element index and bibliography (default model, after ingest);
                    arXiv lookup for cited papers
  figures.py        crops a figure/table out of the PDF by its caption (PyMuPDF, no model)
  transcribe.py     voice questions: one phrase at a time to Whisper on OpenRouter, primed with the
                    paper's terms (POST /api/transcribe); the browser finds the phrases (Listener, app.js)
  usage.py          spending: records each model call's tokens and cost, sums them per day/week/month
  tools.py          web_search, fetch_url, search_arxiv, read_arxiv_paper (provider-neutral schemas)
  prompts.py        system prompt and the paper / question message formats
  db.py             SQLite: papers, folders, conversations, messages, usage
  turso.py          Turso (hosted SQLite) behind the sqlite3 interface db.py uses, when TURSO_DATABASE_URL is set
  providers/        anthropic (Messages API, adaptive thinking), openai (Responses API, stateless),
                    gemini (google-genai), openrouter (Chat Completions),
                    all behind one stream_step() interface
static/             single-page UI (vanilla JS, pdf.js, marked, KaTeX; no build step)
data/               downloaded PDFs + app.db (gitignored)
```

Messages are stored in a provider-neutral format. Each assistant message also keeps the provider's
native output (Claude thinking signatures, OpenAI encrypted reasoning items, Gemini thought signatures).
When the same provider continues the conversation, that output is replayed exactly. When you switch
providers, the neutral form is used instead.

## Notes

- Text extraction loses figures, and complex math can come out garbled. The model is told to say so rather than guess.
- Default provider: OpenRouter (`DEFAULT_PROVIDER`), falling back to the first provider with a key.
- Default models: `claude-opus-5`, `gpt-5`, `gemini-3.8-flash`, `deepseek/deepseek-v4.1-flash` (OpenRouter). Override them in `.env`. The model field lists whatever models your key can access.
- OpenRouter reserves credit for the full output limit up front, so requests are capped at `OPENROUTER_MAX_TOKENS` (default 16384, reasoning included). Lower it if you get 402 errors on a small balance.
- On Claude Opus 5 / Fable, server-side refusal fallback (`fallbacks: "default"`) is enabled.
