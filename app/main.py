import asyncio
import json
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, db, figures, papers, transcribe, usage
from .agent import run_turn
from .config import DEFAULT_MODELS, ON_VERCEL, STATIC_DIR, VOICE_END_SILENCE
from .providers import PROVIDERS, default_provider

db.init()
app = FastAPI(title="Academic Paper Reader")
_model_cache: dict[str, list[str]] = {}
_arxiv_titles: dict[str, str | None] = {}  # arXiv ID -> title (None: no such paper)
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
    question: str
    selection: Selection | None = None


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
def config():
    return {"default_provider": (p := default_provider()) and p.id, "providers": [
        {"id": p.id, "label": p.label, "available": p.available(), "default_model": DEFAULT_MODELS[p.id]}
        for p in PROVIDERS.values()
    ], "voice": {"end_silence": VOICE_END_SILENCE} if transcribe.available() else None}


@app.get("/api/providers/{provider_id}/models")
async def models(provider_id: str):
    p = PROVIDERS.get(provider_id)
    if not p or not p.available():
        raise HTTPException(404, "Provider not configured")
    if provider_id not in _model_cache:
        try:
            _model_cache[provider_id] = await p.list_models()
        except Exception as e:
            raise HTTPException(502, f"Could not list models: {e}")
    return {"models": _model_cache[provider_id]}


@app.post("/api/papers")
async def add_paper(body: PaperIn):
    try:
        paper = await papers.ingest(body.url, body.model_dump(exclude={"url"}))
    except papers.PaperError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to load paper: {e}")
    # Covers new papers and papers ingested before an analysis existed. Failed analyses wait for
    # an explicit retry (each one re-reads the whole paper).
    analysis.ensure(paper["id"])
    return paper


@app.get("/api/search")
async def search(q: str):
    """Papers matching a name. Only results with an arXiv copy (arxiv_id set) can be opened."""
    if not q.strip():
        return {"results": []}
    try:
        return {"results": await papers.search_papers(q.strip()[:300])}
    except Exception as e:
        raise HTTPException(502, f"Paper search failed: {e}")


@app.get("/api/arxiv/titles")
async def arxiv_titles(ids: str):
    """Titles for arXiv IDs, so the UI can show what each paper link really points to."""
    wanted = {i: papers.parse_arxiv_id(i) for i in ids.split(",")[:50] if i.strip()}
    missing = sorted({a for a in wanted.values() if a and a not in _arxiv_titles})
    if missing:
        try:
            found = await papers.arxiv_query({"id_list": ",".join(missing), "max_results": len(missing)})
        except Exception as e:
            raise HTTPException(502, f"arXiv lookup failed: {e}")
        titles = {r["arxiv_id"]: r["title"] for r in found}
        for a in missing:
            _arxiv_titles[a] = titles.get(a)
    return {i: _arxiv_titles.get(a) if a else None for i, a in wanted.items()}


@app.get("/api/papers")
def list_papers():
    return db.list_papers()


@app.get("/api/folders")
def list_folders():
    return db.list_folders()


def _folder_name(body: FolderIn | FolderPatch) -> str:
    name = " ".join(body.name.split())[:80]
    if not name:
        raise HTTPException(400, "Folder name is required")
    return name


@app.post("/api/folders")
def create_folder(body: FolderIn):
    if body.parent_id and body.parent_id not in {f["id"] for f in db.list_folders()}:
        raise HTTPException(400, "Unknown parent folder")
    return db.create_folder(_folder_name(body), body.parent_id)


@app.patch("/api/folders/{folder_id}")
def update_folder(folder_id: str, body: FolderPatch):
    """Renames the folder and/or moves it into another folder (parent_id null: to the top level)."""
    if folder_id not in {f["id"] for f in db.list_folders()}:
        raise HTTPException(404)
    if "parent_id" in body.model_fields_set:
        if body.parent_id:
            if body.parent_id not in {f["id"] for f in db.list_folders()}:
                raise HTTPException(400, "Unknown parent folder")
            if body.parent_id in db.folder_subtree_ids(folder_id):
                raise HTTPException(400, "Can't move a folder into itself or one of its subfolders")
        db.set_folder_parent(folder_id, body.parent_id)
    if body.name is not None:
        db.rename_folder(folder_id, _folder_name(body))
    return {"ok": True}


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str, with_papers: bool = False):
    """Deletes the folder; its papers and subfolders move up to its parent, or with
    ?with_papers=true the subfolders and every paper in them are deleted too."""
    if with_papers:
        for paper_id in db.folder_paper_ids(folder_id):
            _delete_paper(paper_id)
        db.delete_folder_tree(folder_id)
    else:
        db.delete_folder(folder_id)
    return {"ok": True}


def _delete_paper(paper_id: str) -> bool:
    if not db.delete_paper(paper_id):
        return False
    papers.pdf_path(paper_id).unlink(missing_ok=True)
    shutil.rmtree(figures.crops_dir(paper_id), ignore_errors=True)
    return True


@app.delete("/api/papers/{paper_id:path}")
def delete_paper(paper_id: str):
    """Deletes the paper, its conversations, its PDF and cached figure crops."""
    if not _delete_paper(paper_id):
        raise HTTPException(404)
    return {"ok": True}


@app.patch("/api/papers/{paper_id:path}")
def move_paper(paper_id: str, body: PaperPatch):
    if body.folder_id and body.folder_id not in {f["id"] for f in db.list_folders()}:
        raise HTTPException(400, "Unknown folder")
    if not db.set_paper_folder(paper_id, body.folder_id):
        raise HTTPException(404)
    return db.get_paper(paper_id)


@app.get("/api/papers/{paper_id:path}/newer")
async def paper_newer(paper_id: str):
    """{"version": n} when arXiv has a newer version of the paper than the one in the library."""
    paper = db.get_paper(paper_id)
    if not paper:
        raise HTTPException(404)
    try:
        return {"version": await papers.newer_version(paper)}
    except Exception:  # arXiv unreachable: ask again next time
        return {"version": None}


@app.post("/api/papers/{paper_id:path}/update")
async def update_paper(paper_id: str):
    """Replaces the paper with its latest arXiv version, processed like a new paper; the version
    it was moves to the Archive folder with its conversations."""
    if analysis.running(paper_id):
        raise HTTPException(409, "Wait until this paper's analysis is done")
    try:
        paper = await papers.update_to_latest(paper_id)
    except papers.PaperError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to get the new version: {e}")
    analysis.ensure(paper["id"])
    return paper


@app.get("/api/papers/{paper_id:path}/pdf")
async def paper_pdf(paper_id: str, sha: str | None = None):
    paper = db.get_paper(paper_id)
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
async def resolve_bib(paper_id: str, body: BibResolveIn):
    """arXiv IDs for selected citations, looked up on first use and cached."""
    return await analysis.resolve_bib(paper_id, body.ids[:20])


@app.get("/api/papers/{paper_id:path}/crop")
async def paper_crop(paper_id: str, kind: str, number: str):
    """PNG of a figure or table cropped from the PDF (located by its caption), cached on disk."""
    paper = db.get_paper(paper_id)
    if kind not in ("figure", "table") or not paper:
        raise HTTPException(404)
    if not figures.crop_path(paper_id, kind, number).exists():
        try:
            await papers.ensure_pdf(paper)
        except papers.PaperError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(502, f"Could not download the PDF: {e}")
    refs = (db.get_analysis(paper_id, "refs") or {}).get("entries") or {}
    page = (refs.get(f"{kind} {number.upper()}") or {}).get("page")
    png = await asyncio.to_thread(figures.crop, paper_id, kind, number, page)
    if not png:
        raise HTTPException(404, "Could not locate it in the PDF")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/papers/{paper_id:path}/analysis/{name}/retry")
async def retry_analysis(paper_id: str, name: str):
    """Runs a lookup again, continuing from the items it kept. (Async: it starts a task on the event loop.)"""
    if name not in db.ANALYSES or not db.get_paper(paper_id):
        raise HTTPException(404)
    if not analysis.retry(paper_id, name):
        raise HTTPException(409, "It is already running, or no model is configured")
    return {"ok": True}


@app.get("/api/papers/{paper_id:path}/analysis")
def paper_analysis(paper_id: str):
    """Model-generated lookups for the reader: {"glossary"|"refs"|"bib": {status, entries, resumable}}."""
    out = {name: db.get_analysis(paper_id, name) for name in db.ANALYSES}
    if out["glossary"] is None:
        raise HTTPException(404)
    for name, a in out.items():
        a["resumable"] = analysis.resumable(name, a["entries"])  # a retry continues instead of starting over
    return out


@app.get("/api/papers/{paper_id:path}")
def get_paper(paper_id: str):
    paper = db.get_paper(paper_id)
    if not paper:
        raise HTTPException(404)
    return paper


@app.get("/api/conversations")
def list_conversations(paper_id: str | None = None):
    return db.list_conversations(paper_id)


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404)
    msgs = db.get_messages(conv_id)
    for m in msgs:
        m.pop("raw", None)
    return {**conv, "messages": msgs}


@app.patch("/api/conversations/{conv_id}")
def patch_conversation(conv_id: str, body: ConversationPatch):
    if body.provider and body.provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    db.update_conversation(conv_id, **body.model_dump())
    return db.get_conversation(conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    db.delete_conversation(conv_id)
    return {"ok": True}


@app.get("/api/usage")
async def usage_summary(tz: str | None = None):
    """Spending on model calls per day, week and month (in the time zone tz), with any budgets."""
    return await usage.summary(tz)


@app.post("/api/transcribe")
async def transcribe_phrase(request: Request, paper_id: str | None = None, context: str = "", draft: bool = False):
    """One phrase of a voice question, as 16 kHz 16-bit mono PCM: its text. context is the question's
    text so far; draft asks for a quick draft of a phrase still being spoken. With paper_id, the
    paper's terms guide spelling."""
    if not transcribe.available():
        raise HTTPException(400, "Voice questions need OPENROUTER_API_KEY")
    pcm = await request.body()
    if len(pcm) > MAX_PHRASE_BYTES:
        raise HTTPException(413, "Recording too long")
    try:
        return {"text": await transcribe.transcribe(pcm, paper_id=paper_id, context=context, draft=draft)}
    except Exception as e:
        raise HTTPException(502, f"Transcription failed: {e or type(e).__name__}")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn):
    if body.conversation_id:
        conv = db.get_conversation(body.conversation_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
        if (body.provider and body.provider != conv["provider"]) or (body.model and body.model != conv["model"]):
            db.update_conversation(conv["id"], provider=body.provider, model=body.model)
            conv = db.get_conversation(conv["id"])
    else:
        # Without a paper_id this starts a search session: a chat with no paper in context.
        if body.paper_id and not db.get_paper(body.paper_id):
            raise HTTPException(400, "Unknown paper")
        provider = body.provider or ((p := default_provider()) and p.id)
        if not provider:
            raise HTTPException(400, "No LLM provider configured; set an API key in .env")
        title = (body.question.strip() or (body.selection.text if body.selection else "")).strip()
        title = title[:80] + ("…" if len(title) > 80 else "")
        conv = db.create_conversation(body.paper_id, title or "New conversation", provider,
                                      body.model or DEFAULT_MODELS[provider])

    if not PROVIDERS[conv["provider"]].available():
        raise HTTPException(400, f"Provider {conv['provider']} has no API key configured")

    async def stream():
        yield _sse({"type": "conversation", "conversation": conv})
        try:
            async for ev in run_turn(conv, body.question, body.selection.model_dump() if body.selection else None):
                yield _sse(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
