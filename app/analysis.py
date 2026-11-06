"""Per-paper lookups generated once by the default model right after ingest:

- glossary: {term: description} for the paper's notation, acronyms and defined terms.
- refs: numbered elements (theorems, equations, tables, sections, ...) keyed "kind number", so a
  selected cross-reference like "Thm. 3.1" can show its content, plus where theorem-like
  elements are proved.
- bib: the reference list, so a selected citation like "[18]" can show the cited paper and open it.
  arXiv IDs are found lazily (resolve_bib) by title search in code, not trusted from the model.

Replies are parsed while they stream: complete items are saved every few seconds (status still
"pending"), so the reader can use them before the reply ends. A reply cut off at the output limit
is continued from the items it produced; if it stays incomplete, they are kept under a "failed"
status and retry() continues from them.

Each user has their own copy of a paper (db.py), analyzed on their own key (auth.user_keys).

A run holds a lease in the database while it works, so a server instance never starts an analysis
that another instance is running. An instance that stops mid-run (Vercel freezes or ends
instances) leaves its lease to expire; the next look at the paper's analyses continues the run
from the items it saved (resume_stalled).
"""

import asyncio
import functools
import json
import logging
import re
import time
import uuid

from . import db, papers, usage
from .config import DEFAULT_MODELS
from .prompts import BIB_PROMPT, CONTINUE_PROMPT, GLOSSARY_PROMPT, REFS_PROMPT, SYSTEM_PROMPT, paper_message_text
from .providers import StepResult, TextDelta, ThinkingDelta, default_provider, stream_step
from .tools import TOOLS

log = logging.getLogger(__name__)
_tasks: dict[tuple[str, str, str], asyncio.Task] = {}  # by (user id, paper id, analysis)
TRUNCATED = {"length", "max_tokens", "MAX_TOKENS", "incomplete"}  # each provider's "hit the output limit"
SAVE_EVERY = 2.0  # seconds between saves of a streaming reply's complete items
LEASE_TTL = 180.0  # seconds a run's lease lasts unless renewed; how long a stalled run waits to be taken over
RENEW_EVERY = 30.0


def _lease_key(user_id: str, paper_id: str, name: str) -> str:
    return f"analysis:{user_id}:{paper_id}:{name}"


async def ensure(user: dict, paper_id: str, retry_failed: bool = False):
    """Start each missing analysis in the background unless it exists or is already running."""
    if default_provider(user["keys"]) is None:
        return
    for name in JOBS:
        if (user["id"], paper_id, name) in _tasks:
            continue
        current = await db.aio.get_analysis(user["id"], paper_id, name) or {}
        status = current.get("status") or ""
        if (status == "ready" and not _outdated(name, current["entries"])) or (status.startswith("failed") and not retry_failed):
            continue
        # A "pending" left behind by a stopped run continues from the items it had saved.
        await _start(user, paper_id, name, resume=current["entries"] if resumable(name, current.get("entries")) else None)


async def resume_stalled(user: dict, paper_id: str, analyses: dict[str, dict]):
    """Continues any "pending" analysis (analyses: as db.get_analysis returns them) that no
    instance is running any more."""
    pending = [name for name, a in analyses.items() if a and a.get("status") == "pending"
               and (user["id"], paper_id, name) not in _tasks]
    if not pending or default_provider(user["keys"]) is None:
        return
    held = await db.aio.active_leases([_lease_key(user["id"], paper_id, n) for n in pending])
    for name in pending:
        if _lease_key(user["id"], paper_id, name) not in held:
            entries = analyses[name].get("entries")
            await _start(user, paper_id, name, resume=entries if resumable(name, entries) else None)


async def running(user_id: str, paper_id: str) -> bool:
    """Whether any analysis of the paper is running, on this server instance or another."""
    if any(key[:2] == (user_id, paper_id) for key in _tasks):
        return True
    return bool(await db.aio.active_leases([_lease_key(user_id, paper_id, n) for n in JOBS]))


def resumable(name: str, entries: dict | None) -> bool:
    """Whether a run can continue from these entries; an outdated index starts over instead
    (keeping its old entries visible meanwhile)."""
    return bool(entries) and not _outdated(name, entries)


async def retry(user: dict, paper_id: str, name: str) -> bool:
    """Run an analysis again, continuing from the items it kept. False if it's running already."""
    if default_provider(user["keys"]) is None or (user["id"], paper_id, name) in _tasks:
        return False
    current = await db.aio.get_analysis(user["id"], paper_id, name)
    if current is None:
        return False
    return await _start(user, paper_id, name, resume=current["entries"] if resumable(name, current["entries"]) else None)


async def _start(user: dict, paper_id: str, name: str, resume: dict | None) -> bool:
    """Starts the run unless another one holds its lease. The status becomes "pending" before the
    run starts; a failure to start leaves the lease to expire and the status as it was."""
    key = (user["id"], paper_id, name)
    holder = uuid.uuid4().hex
    if key in _tasks or not await db.aio.acquire_lease(_lease_key(*key), holder, LEASE_TTL):
        return False
    if key in _tasks:  # started here while the lease was being taken
        await db.aio.release_lease(_lease_key(*key), holder)
        return False
    await db.aio.set_analysis(user["id"], paper_id, name, "pending")
    task = asyncio.create_task(_generate(user, paper_id, name, resume or {}, holder))
    _tasks[key] = task
    task.add_done_callback(lambda _, key=key: _tasks.pop(key, None))
    return True


def _outdated(name: str, entries: dict) -> bool:
    """Refs indexed before proof locations existed: index them again so theorems get theirs.
    (The old entries stay in the db, and usable, until the new ones are ready.)"""
    return name == "refs" and any(e["kind"] in PROVABLE and "proof" not in e for e in entries.values())


async def _generate(user: dict, paper_id: str, name: str, base: dict, holder: str):
    """base: items kept from an earlier, cut-off run, which the model is asked to continue.
    holder: this run's lease (see _start), renewed while it runs and released at the end."""
    lease = _lease_key(user["id"], paper_id, name)
    last_renew = time.monotonic()

    async def renew():
        nonlocal last_renew
        if time.monotonic() - last_renew >= RENEW_EVERY:
            last_renew = time.monotonic()
            await db.aio.acquire_lease(lease, holder, LEASE_TTL)

    try:
        await _run(user, paper_id, name, base, renew)
    finally:
        try:
            await db.aio.release_lease(lease, holder)
        except Exception:
            log.warning("Could not release %s", lease, exc_info=True)


async def _run(user: dict, paper_id: str, name: str, base: dict, renew):
    prompt, parse = JOBS[name]
    user_id = user["id"]
    provider = default_provider(user["keys"])
    model = DEFAULT_MODELS[provider.id]
    paper = await db.aio.get_paper(user_id, paper_id, with_text=True)
    paper_text = paper_message_text(paper)
    # Starting over (an outdated index): the old entries stay usable until new ones replace them.
    keep = {} if base else ((await db.aio.get_analysis(user_id, paper_id, name)) or {}).get("entries") or {}
    try:
        for attempt in range(3):
            # Same system prompt, tools and paper block as a conversation, so this call also warms
            # the prompt cache for the reader's first question.
            messages = [{"role": "user", "content": [
                {"type": "paper"}, {"type": "text", "text": prompt + _continuation(name, base)}]}]
            chunks, saved, last_save, result, thinking = [], 0, time.monotonic(), None, 0
            # No reasoning: it's billed as output and listing items from the paper doesn't need it.
            async for ev in stream_step(provider, model=model, system=SYSTEM_PROMPT, paper_text=paper_text,
                                        paper_id=paper_id, messages=messages, tools=TOOLS, reasoning=False):
                await renew()
                if isinstance(ev, TextDelta):
                    chunks.append(ev.text)
                    if time.monotonic() - last_save >= SAVE_EVERY:
                        last_save = time.monotonic()
                        partial = parse("".join(chunks), paper["text"])
                        if len(partial) > saved:
                            saved = len(partial)
                            await _save(user_id, paper_id, name, "pending", {**keep, **_merge(name, base, partial)})
                elif isinstance(ev, ThinkingDelta):
                    thinking += len(ev.text)
                elif isinstance(ev, StepResult):
                    result = ev
            await usage.record(user_id, provider.id, model, result.usage, kind="analysis", paper_id=paper_id)
            text ="".join(b["text"] for b in result.message["content"] if b["type"] == "text")
            sizes = f"{len(text)} chars of reply, {thinking} of reasoning, {result.usage.get('output', 0)} output tokens"
            entries = parse(text, paper["text"])
            truncated = result.stop_reason in TRUNCATED
            if not truncated and base and _empty_json(text):  # a continuation with nothing left to add
                entries = dict(base)
            if entries and not truncated:
                entries = _merge(name, base, entries)
                await _save(user_id, paper_id, name, "ready", entries)
                log.info("%s for %s: %d entries (%s %s)", name, paper_id, len(entries), provider.id, model)
                return
            if entries:  # cut off: continue after the items it produced
                base = _merge(name, base, entries)
                await _save(user_id, paper_id, name, "pending", {**keep, **base})
                log.warning("%s for %s: cut off at the output limit with %d items (attempt %d; %s)",
                            name, paper_id, len(base), attempt + 1, sizes)
            else:
                log.warning("%s for %s: unparseable reply (attempt %d, stop=%s; %s)",
                            name, paper_id, attempt + 1, result.stop_reason, sizes)
        if base:  # (replacing any outdated entries, so that a retry continues from these)
            await _save(user_id, paper_id, name, f"failed: the reply was still cut off at the output limit ({len(base)} items kept)", base)
        else:
            await db.aio.set_analysis(user_id, paper_id, name, "failed: the model did not return valid JSON")
    except Exception as e:
        log.exception("%s for %s failed", name, paper_id)
        # Whatever streamed before the error was saved as it arrived, so it's kept for a retry.
        await db.aio.set_analysis(user_id, paper_id, name, f"failed: {type(e).__name__}: {e}"[:500])


def _merge(name: str, base: dict, new: dict) -> dict:
    """Items from a continuation appended to the kept ones, skipping repeats. Bibliography entries
    are keyed by position, so they are renumbered after the kept ones."""
    if name != "bib":
        return {**base, **{k: v for k, v in new.items() if k not in base}}
    out = dict(base)
    seen = {e["title"].lower() for e in base.values()}
    for e in new.values():
        if e["title"].lower() not in seen:
            seen.add(e["title"].lower())
            out[str(len(out))] = e
    return out


def _continuation(name: str, base: dict) -> str:
    if not base:
        return ""
    if name == "glossary":
        names = list(base)
    elif name == "refs":
        names = [e["label"] for e in base.values()]
    else:
        names = [f"[{e['number']}] {e['title']}" if e.get("number") else e["title"] for e in base.values()]
    return CONTINUE_PROMPT.format(count=len(names), done="; ".join(names), last=names[-1],
                                  shape="object" if name == "glossary" else "array")


async def _save(user_id: str, paper_id: str, name: str, status: str, entries: dict):
    """Saves entries, keeping the arXiv lookups resolve_bib already made for bibliography entries."""
    if name == "bib":
        current = ((await db.aio.get_analysis(user_id, paper_id, "bib")) or {}).get("entries") or {}
        for k, e in entries.items():
            c = current.get(k)
            if c and c.get("lookup") and not e.get("lookup") and c["title"] == e["title"]:
                e["arxiv_id"], e["lookup"] = c["arxiv_id"], c["lookup"]
    await db.aio.set_analysis(user_id, paper_id, name, status, entries)


def _empty_json(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:```(?:json)?)?\s*(?:\{\s*\}|\[\s*\])\s*(?:```)?\s*", text))


def _json_items(text: str) -> list:
    """Objects of a JSON array, read one by one so a reply cut off mid-array still yields its
    complete items (long reference lists can hit the output limit). A continuation that leaves
    out the opening "[" is read from its first object."""
    start, first = text.find("["), text.find("{")
    if first >= 0 and (start < 0 or first < start):
        start = first - 1
    if start < 0 and first < 0:
        return []
    decoder, items, i = json.JSONDecoder(), [], start + 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] == "]":
            return items
        try:
            item, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return items
        items.append(item)


def _json_pairs(text: str) -> list[tuple]:
    """Key/value pairs of a JSON object, read one by one so a reply cut off mid-object still
    yields its complete pairs. A continuation that leaves out the opening "{" is read from its
    first key."""
    start, quote = text.find("{"), text.find('"')
    if quote >= 0 and (start < 0 or quote < start):
        start = quote - 1
    if start < 0:
        return []
    decoder, pairs, i = json.JSONDecoder(), [], start + 1

    def skip(i, chars=" \t\r\n"):
        while i < len(text) and text[i] in chars:
            i += 1
        return i
    while True:
        i = skip(i, " \t\r\n,")
        if i >= len(text) or text[i] == "}":
            return pairs
        try:
            key, i = decoder.raw_decode(text, i)
            i = skip(i)
            if i >= len(text) or text[i] != ":":
                return pairs
            value, i = decoder.raw_decode(text, skip(i + 1))
        except json.JSONDecodeError:
            return pairs
        pairs.append((key, value))


_NUMBER_ONLY = re.compile(r"[$€£]?\s*[\d.,]+\s*%?")  # table values a padded reply adds, e.g. "41.3", "$0.064"


def parse_glossary(text: str, paper_text: str = "") -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in _json_pairs(text)
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            and not _NUMBER_ONLY.fullmatch(k.strip())}


# Canonical kinds; the frontend's reference parser maps selections onto the same names.
REF_KINDS = {
    "definition", "theorem", "lemma", "proposition", "corollary", "conjecture", "claim", "assumption",
    "hypothesis", "remark", "example", "problem", "algorithm", "equation", "table", "figure", "section",
}
PROVABLE = {"theorem", "lemma", "proposition", "corollary", "claim"}  # these get a "proof" location
_KIND_ALIASES = {"subsection": "section", "subsubsection": "section", "appendix": "section", "eq": "equation",
                 "fig": "figure", "tab": "table", "def": "definition", "thm": "theorem", "lem": "lemma",
                 "prop": "proposition", "cor": "corollary"}


def ref_key(kind: str, number: str) -> str:
    return f"{kind} {ref_number(number)}"


def ref_number(number: str) -> str:
    return re.sub(r"[\s()]", "", str(number)).strip(".").upper()


def parse_refs(text: str, paper_text: str = "") -> dict[str, dict]:
    out = {}
    for item in _json_items(text):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        kind = _KIND_ALIASES.get(kind, kind)
        number, content = ref_number(item.get("number", "")), str(item.get("content", "")).strip()
        if kind not in REF_KINDS or not number or not content:
            continue
        page = item.get("page")
        entry = {
            "kind": kind, "number": number, "content": content,
            "label": str(item.get("label") or f"{kind.title()} {number}").strip(),
            "page": _page(page),
        }
        if kind in PROVABLE:
            entry["proof"] = _proof(item.get("proof"), paper_text)
        out.setdefault(ref_key(kind, number), entry)
    return out


def _page(page) -> int | None:
    page = int(page) if isinstance(page, str) and page.strip().isdigit() else page
    return page if isinstance(page, int) and page > 0 else None


def _proof(proof, paper_text: str) -> dict | None:
    """{"page", "start"} of a proof. The page comes from finding the start text in the paper
    (the model's page number is only a hint); None if neither is usable."""
    if not isinstance(proof, dict):
        return None
    start = " ".join(str(proof.get("start") or "").split())
    page = _find_page(paper_text, start, _page(proof.get("page"))) or _page(proof.get("page"))
    return {"page": page, "start": start} if page else None


@functools.lru_cache(maxsize=8)  # a streaming reply is re-parsed every few seconds
def _page_words(paper_text: str) -> dict[int, str]:
    parts = re.split(r"\[Page (\d+)\]\n", paper_text)  # ["", "1", text, "2", text, ...]
    return {int(n): " ".join(papers.title_words(t)) for n, t in zip(parts[1::2], parts[2::2])}


def _find_page(paper_text: str, snippet: str, hint: int | None) -> int | None:
    words = " ".join(papers.title_words(snippet)[:8])
    if len(words.split()) < 3:
        return None  # too short to place reliably
    pages = _page_words(paper_text)
    for n in sorted(pages, key=lambda n: (n != hint, n)):
        if f" {words} " in f" {pages[n]} ":
            return n
    return None


def parse_bib(text: str, paper_text: str = "") -> dict[str, dict]:
    def s(v):
        return " ".join(str(v).split()) if v is not None else ""
    out = {}
    for item in _json_items(text):
        if not isinstance(item, dict) or not s(item.get("title")):
            continue
        authors = [s(a) for a in item.get("authors") or [] if s(a)] if isinstance(item.get("authors"), list) else []
        printed = papers.parse_arxiv_id(s(item.get("arxiv_id"))) if item.get("arxiv_id") else None
        out[str(len(out))] = {
            "number": s(item.get("number")).strip("[]() ") or None,
            "authors": authors,
            "first_author": s(item.get("first_author")) or (authors[0].split()[-1] if authors else ""),
            "year": s(item.get("year")),
            "title": s(item.get("title")),
            "venue": s(item.get("venue")),
            "url": s(item.get("url")) or None,
            "arxiv_id": printed,
            "lookup": "printed" if printed else None,  # None until resolve_bib has searched arXiv
        }
    return out


JOBS = {
    "glossary": (GLOSSARY_PROMPT, parse_glossary),
    "refs": (REFS_PROMPT, parse_refs),
    "bib": (BIB_PROMPT, parse_bib),
}

_bib_locks: dict[tuple[str, str], asyncio.Lock] = {}


async def resolve_bib(user_id: str, paper_id: str, ids: list[str]) -> dict[str, dict]:
    """Find arXiv IDs for the given bibliography entries by title search, caching the outcome
    (an ID, or "none") in the paper's bib so each entry is searched at most once."""
    async with _bib_locks.setdefault((user_id, paper_id), asyncio.Lock()):
        bib = ((await db.aio.get_analysis(user_id, paper_id, "bib")) or {}).get("entries") or {}
        found = {}
        for i in ids:
            e = bib.get(i)
            if not e or e.get("lookup"):
                continue
            try:
                # A paper already looked up (by anyone) with this exact title needs no request to arXiv.
                results = await papers.find_by_title(e["title"]) or await papers.search_arxiv(e["title"], max_results=5)
            except Exception:
                log.warning("arXiv lookup failed for %s entry %s", paper_id, i, exc_info=True)
                continue  # leave unresolved; a later selection retries
            surname = papers.title_words(e.get("first_author") or "")
            best = None
            for r in results:
                score = papers.title_match(e["title"], r["title"])
                author_ok = bool(surname) and any(surname[-1] in papers.title_words(a) for a in r["authors"][:3])
                if score >= 0.9 or (score >= 0.75 and author_ok):
                    if not best or score > best[0]:
                        best = (score, r)
            found[i] = (e["title"], *((best[1]["arxiv_id"], "arxiv") if best else (None, "none")))
        # The bibliography may still be streaming in: apply the lookups to the current entries and
        # keep its status.
        current = await db.aio.get_analysis(user_id, paper_id, "bib") or {}
        bib = current.get("entries") or {}
        for i, (title, arxiv_id, lookup) in found.items():
            if i in bib and bib[i]["title"] == title:
                bib[i]["arxiv_id"], bib[i]["lookup"] = arxiv_id, lookup
        if found:
            await db.aio.set_analysis(user_id, paper_id, "bib", current["status"], bib)
        return {i: bib[i] for i in ids if i in bib}
