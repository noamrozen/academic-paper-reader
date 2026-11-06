import asyncio
import json
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, auth, db, figures, papers, transcribe, usage
from .agent import NothingToResume, run_turn
from .auth import User
from .config import DEFAULT_MODELS, ON_VERCEL, STATIC_DIR, VOICE_END_SILENCE
from .crypto import token_hash
from .providers import PROVIDERS, default_provider, provider_for

db.init()
app = FastAPI(title="Academic Paper Reader")
app.include_router(auth.router)
_model_cache: dict[tuple[str, str], list[str]] = {}  # (provider, hash of the key) -> models
_no_such_arxiv: set[str] = set()  # IDs arXiv didn't know (found ones are kept in the database)
MAX_PHRASE_BYTES = 2 * 60 * 16000 * 2  # two minutes of 16 kHz 16-bit audio


@app.middleware("http")
async def revalidate_ui(request, call_next):
    # No build step means no hashed filenames, so make browsers revalidate the UI on every load
    # (cheap: unchanged files still return 304) instead of running a stale cached app.js.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class PaperIn(BaseModel):
    url: str
    # Metadata for a non-arXiv PDF found by search (arXiv papers use arXiv's own metadata).
    title: str | None = None
    authors: list[str] = []
    published: str | None = None


class Selection(BaseModel):
    text: str
    page: int | None = None


class ChatIn(BaseModel):
    conversation_id: str | None = None
    paper_id: str | None = None
    provider: str | None = None
    model: str | None = None
    question: str = ""
    selection: Selection | None = None
    # Continue the conversation's paused answer (a "paused" event) instead of asking a question.
    resume: bool = False


class BibResolveIn(BaseModel):
    ids: list[str]


class FolderIn(BaseModel):
    name: str
    parent_id: str | None = None  # None for a top-level folder


class FolderPatch(BaseModel):
    name: str | None = None
    parent_id: str | None = None  # when given, None moves the folder to the top level


class PaperPatch(BaseModel):
    folder_id: str | None = None  # None moves the paper out of any folder


class ConversationPatch(BaseModel):
    title: str | None = None
    provider: str | None = None
    model: str | None = None


@app.get("/api/config")
def config(user: User):
    """The providers the user has a key for (theirs, or the server's for an admin), and voice."""
    return {"default_provider": (p := default_provider(user["keys"])) and p.id, "providers": [
        {"id": p.id, "label": p.label, "available": bool(user["keys"].get(p.id)), "default_model": DEFAULT_MODELS[p.id]}
        for p in PROVIDERS.values()
    ], "voice": {"end_silence": VOICE_END_SILENCE} if transcribe.available(user) else None,
       "user": auth.public(user)}


@app.get("/api/providers/{provider_id}/models")
async def models(provider_id: str, user: User):
    p = provider_for(user["keys"], provider_id)
    if not p:
        raise HTTPException(404, "Provider not configured")
    key = (provider_id, token_hash(user["keys"][provider_id]))  # what a key can see differs by account
    if key not in _model_cache:
        try:
            _model_cache[key] = await p.list_models()
        except Exception as e:
            raise HTTPException(502, f"Could not list models: {e}")
    return {"models": _model_cache[key]}


@app.post("/api/papers")
async def add_paper(body: PaperIn, user: User):
    try:
        paper = await papers.ingest(user["id"], body.url, body.model_dump(exclude={"url"}))
    except papers.PaperError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to load paper: {e}")
    # Covers new papers and papers ingested before an analysis existed. Failed analyses wait for
    # an explicit retry (each one re-reads the whole paper).
    await analysis.ensure(user, paper["id"])
    return paper


@app.get("/api/search")
async def search(q: str, user: User):
    """Papers matching a name. Only results with an arXiv copy (arxiv_id set) can be opened."""
    if not q.strip():
        return {"results": []}
    try:
        return {"results": await papers.search_papers(q.strip()[:300])}
    except Exception as e:
        raise HTTPException(502, f"Paper search failed: {e}")


@app.get("/api/arxiv/titles")
async def arxiv_titles(ids: str, user: User):
    """Titles for arXiv IDs, so the UI can show what each paper link really points to."""
    wanted = {i: papers.parse_arxiv_id(i) for i in ids.split(",")[:50] if i.strip()}
    lookup = sorted({a for a in wanted.values() if a and a not in _no_such_arxiv})
    try:
        found = await papers.arxiv_metadata_many(lookup)  # from the database when looked up before
    except Exception as e:
        raise HTTPException(502, f"arXiv lookup failed: {e}")
    _no_such_arxiv.update(a for a in lookup if a not in found)
    return {i: found[a]["title"] if a in found else None for i, a in wanted.items()}


@app.get("/api/papers")
def list_papers(user: User):
    return db.list_papers(user["id"])


@app.get("/api/folders")
def list_folders(user: User):
    return db.list_folders(user["id"])


def _folder_name(body: FolderIn | FolderPatch) -> str:
    name = " ".join(body.name.split())[:80]
    if not name:
        raise HTTPException(400, "Folder name is required")
    return name


@app.post("/api/folders")
def create_folder(body: FolderIn, user: User):
    if body.parent_id and body.parent_id not in {f["id"] for f in db.list_folders(user["id"])}:
        raise HTTPException(400, "Unknown parent folder")
    return db.create_folder(user["id"], _folder_name(body), body.parent_id)


@app.patch("/api/folders/{folder_id}")
def update_folder(folder_id: str, body: FolderPatch, user: User):
    """Renames the folder and/or moves it into another folder (parent_id null: to the top level)."""
    folder_ids = {f["id"] for f in db.list_folders(user["id"])}
    if folder_id not in folder_ids:
        raise HTTPException(404)
    if "parent_id" in body.model_fields_set:
        if body.parent_id:
            if body.parent_id not in folder_ids:
                raise HTTPException(400, "Unknown parent folder")
            if body.parent_id in db.folder_subtree_ids(user["id"], folder_id):
                raise HTTPException(400, "Can't move a folder into itself or one of its subfolders")
        db.set_folder_parent(user["id"], folder_id, body.parent_id)
    if body.name is not None:
        db.rename_folder(user["id"], folder_id, _folder_name(body))
    return {"ok": True}


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str, user: User, with_papers: bool = False):
    """Deletes the folder; its papers and subfolders move up to its parent, or with
    ?with_papers=true the subfolders and every paper in them are deleted too."""
    if with_papers:
        for paper_id in db.folder_paper_ids(user["id"], folder_id):
            _delete_paper(user["id"], paper_id)
        db.delete_folder_tree(user["id"], folder_id)
    else:
        db.delete_folder(user["id"], folder_id)
    return {"ok": True}


def _delete_paper(user_id: str, paper_id: str) -> bool:
    paper = db.get_paper(user_id, paper_id)
    if not paper or not db.delete_paper(user_id, paper_id):
        return False
    # The cached files are shared by every user's copy of the same PDF.
    if not (paper.get("pdf_sha256") and db.pdf_in_use(paper["pdf_sha256"])):
        papers.pdf_path(paper).unlink(missing_ok=True)
        shutil.rmtree(figures.crops_dir(paper), ignore_errors=True)
    return True


@app.delete("/api/papers/{paper_id:path}")
def delete_paper(paper_id: str, user: User):
    """Deletes the paper, its conversations, its PDF and cached figure crops."""
    if not _delete_paper(user["id"], paper_id):
        raise HTTPException(404)
    return {"ok": True}


@app.patch("/api/papers/{paper_id:path}")
def move_paper(paper_id: str, body: PaperPatch, user: User):
    if body.folder_id and body.folder_id not in {f["id"] for f in db.list_folders(user["id"])}:
        raise HTTPException(400, "Unknown folder")
    if not db.set_paper_folder(user["id"], paper_id, body.folder_id):
        raise HTTPException(404)
    return db.get_paper(user["id"], paper_id)


@app.get("/api/papers/{paper_id:path}/newer")
async def paper_newer(paper_id: str, user: User):
    """{"version": n} when arXiv has a newer version of the paper than the one in the library."""
    paper = await db.aio.get_paper(user["id"], paper_id)
    if not paper:
        raise HTTPException(404)
    try:
        return {"version": await papers.newer_version(paper)}
    except Exception:  # arXiv unreachable: ask again next time
        return {"version": None}


@app.post("/api/papers/{paper_id:path}/update")
async def update_paper(paper_id: str, user: User):
    """Replaces the paper with its latest arXiv version, processed like a new paper; the version
    it was moves to the Archive folder with its conversations."""
    if await analysis.running(user["id"], paper_id):
        raise HTTPException(409, "Wait until this paper's analysis is done")
    try:
        paper = await papers.update_to_latest(user["id"], paper_id)
    except papers.PaperError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to get the new version: {e}")
    await analysis.ensure(user, paper["id"])
    return paper


@app.get("/api/papers/{paper_id:path}/pdf")
async def paper_pdf(paper_id: str, user: User, sha: str | None = None):
    paper = await db.aio.get_paper(user["id"], paper_id)
    if not paper:
        raise HTTPException(404)
    # Vercel caps a response at 4.5 MB, smaller than many papers: the browser gets arXiv PDFs from
    # arXiv itself (which allows that).
    if ON_VERCEL and not paper_id.startswith("url-"):
        return RedirectResponse(papers.pdf_url(paper))
    try:
        path = await papers.ensure_pdf(paper)
    except papers.PaperError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not download the PDF: {e}")
    # Asked for by the file's hash (?sha=), the browser can keep its copy; otherwise it checks
    # each time (a paper updated to a newer arXiv version keeps its id).
    cache = "max-age=31536000, immutable" if sha and sha == paper.get("pdf_sha256") else "no-cache"
    return FileResponse(path, media_type="application/pdf", headers={"Cache-Control": cache})


@app.post("/api/papers/{paper_id:path}/bib/resolve")
async def resolve_bib(paper_id: str, body: BibResolveIn, user: User):
    """arXiv IDs for selected citations, looked up on first use and cached."""
    return await analysis.resolve_bib(user["id"], paper_id, body.ids[:20])


@app.get("/api/papers/{paper_id:path}/crop")
async def paper_crop(paper_id: str, kind: str, number: str, user: User):
    """PNG of a figure or table cropped from the PDF (located by its caption), cached on disk."""
    if kind not in ("figure", "table") or not figures.valid_number(number):
        raise HTTPException(404)
    paper = await db.aio.get_paper(user["id"], paper_id)
    if not paper:
        raise HTTPException(404)
    if not figures.crop_path(paper, kind, number).exists():
        try:
            await papers.ensure_pdf(paper)
        except papers.PaperError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(502, f"Could not download the PDF: {e}")
    refs = ((await db.aio.get_analysis(user["id"], paper_id, "refs")) or {}).get("entries") or {}
    page = (refs.get(f"{kind} {number.upper()}") or {}).get("page")
    png = await asyncio.to_thread(figures.crop, paper, kind, number, page)
    if not png:
        raise HTTPException(404, "Could not locate it in the PDF")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/papers/{paper_id:path}/analysis/{name}/retry")
async def retry_analysis(paper_id: str, name: str, user: User):
    """Runs a lookup again, continuing from the items it kept. (Async: it starts a task on the event loop.)"""
    if name not in db.ANALYSES or not await db.aio.get_paper(user["id"], paper_id):
        raise HTTPException(404)
    if not await analysis.retry(user, paper_id, name):
        raise HTTPException(409, "It is already running, or you have no API key yet (Settings → Account)")
    return {"ok": True}


@app.get("/api/papers/{paper_id:path}/analysis")
async def paper_analysis(paper_id: str, user: User):
    """Model-generated lookups for the reader: {"glossary"|"refs"|"bib": {status, entries, resumable}}.
    The reader polls this while any is pending, which also continues a run whose server stopped."""
    out = {name: await db.aio.get_analysis(user["id"], paper_id, name) for name in db.ANALYSES}
    if out["glossary"] is None:
        raise HTTPException(404)
    await analysis.resume_stalled(user, paper_id, out)
    for name, a in out.items():
        a["resumable"] = analysis.resumable(name, a["entries"])  # a retry continues instead of starting over
    return out


@app.get("/api/papers/{paper_id:path}")
def get_paper(paper_id: str, user: User):
    paper = db.get_paper(user["id"], paper_id)
    if not paper:
        raise HTTPException(404)
    return paper


@app.get("/api/conversations")
def list_conversations(user: User, paper_id: str | None = None):
    return db.list_conversations(user["id"], paper_id)


def _conversation(user: dict, conv_id: str) -> dict:
    conv = db.get_conversation(user["id"], conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str, user: User):
    conv = _conversation(user, conv_id)
    msgs = db.get_messages(conv_id)
    for m in msgs:
        m.pop("raw", None)
    return {**conv, "messages": msgs}


@app.patch("/api/conversations/{conv_id}")
def patch_conversation(conv_id: str, body: ConversationPatch, user: User):
    if body.provider and body.provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    _conversation(user, conv_id)
    db.update_conversation(user["id"], conv_id, **body.model_dump())
    return db.get_conversation(user["id"], conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, user: User):
    db.delete_conversation(user["id"], conv_id)
    return {"ok": True}


@app.get("/api/usage")
async def usage_summary(user: User, tz: str | None = None):
    """The user's spending on model calls per day, week and month (in the time zone tz), with any budgets."""
    return await usage.summary(user, tz)


@app.post("/api/transcribe")
async def transcribe_phrase(request: Request, user: User, paper_id: str | None = None, context: str = "",
                            draft: bool = False):
    """One phrase of a voice question, as 16 kHz 16-bit mono PCM: its text. context is the question's
    text so far; draft asks for a quick draft of a phrase still being spoken. With paper_id, the
    paper's terms guide spelling."""
    if not transcribe.available(user):
        raise HTTPException(400, "Voice questions need an OpenRouter key (Settings → Account)")
    pcm = await request.body()
    if len(pcm) > MAX_PHRASE_BYTES:
        raise HTTPException(413, "Recording too long")
    try:
        return {"text": await transcribe.transcribe(user, pcm, paper_id=paper_id, context=context, draft=draft)}
    except Exception as e:
        raise HTTPException(502, f"Transcription failed: {e or type(e).__name__}")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn, user: User):
    if body.provider and body.provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    if body.resume and not body.conversation_id:
        raise HTTPException(400, "Nothing to continue")
    if body.conversation_id:
        conv = await db.aio.get_conversation(user["id"], body.conversation_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
        if (body.provider and body.provider != conv["provider"]) or (body.model and body.model != conv["model"]):
            await db.aio.update_conversation(user["id"], conv["id"], provider=body.provider, model=body.model)
            conv = await db.aio.get_conversation(user["id"], conv["id"])
    else:
        # Without a paper_id this starts a search session: a chat with no paper in context.
        if body.paper_id and not await db.aio.get_paper(user["id"], body.paper_id):
            raise HTTPException(400, "Unknown paper")
        provider = body.provider or ((p := default_provider(user["keys"])) and p.id)
        if not provider_for(user["keys"], provider):
            raise HTTPException(400, "Add an API key first: Settings → Account")
        title = (body.question.strip() or (body.selection.text if body.selection else "")).strip()
        title = title[:80] + ("…" if len(title) > 80 else "")
        conv = await db.aio.create_conversation(user["id"], body.paper_id, title or "New conversation", provider,
                                                body.model or DEFAULT_MODELS[provider])

    if not provider_for(user["keys"], conv["provider"]):
        raise HTTPException(400, f"Add a {PROVIDERS[conv['provider']].label} key first (Settings → Account),"
                                 " or pick a provider you have a key for")

    async def stream():
        yield _sse({"type": "conversation", "conversation": conv})
        try:
            async for ev in run_turn(user, conv, body.question, body.selection.model_dump() if body.selection else None,
                                     resume=body.resume):
                yield _sse(ev)
        except asyncio.CancelledError:
            raise
        except NothingToResume as e:
            yield _sse({"type": "error", "message": str(e)})
            yield _sse({"type": "done"})
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
