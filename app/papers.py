"""Resolve, download and extract papers (arXiv first, any PDF URL as a fallback)."""

import asyncio
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import urlparse

import httpx
import pymupdf

from . import db
from .config import HTTP_HEADERS, PAPERS_DIR

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ARXIV_URL = re.compile(
    rf"arxiv\.org/(?:abs|pdf|html|format)/({_NEW_ID}|{_OLD_ID})", re.IGNORECASE
)
_BARE_ID = re.compile(rf"^(?:arxiv:)?({_NEW_ID}|{_OLD_ID})$", re.IGNORECASE)

_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class PaperError(Exception):
    pass


def parse_arxiv_id(s: str) -> str | None:
    s = s.strip()
    m = _ARXIV_URL.search(s) or _BARE_ID.match(s)
    if not m:
        return None
    # Drop the version suffix so every reference to a paper maps to one stored copy (latest version).
    return re.sub(r"v\d+$", "", m.group(1).removesuffix(".pdf"))


def paper_id_for(source: str) -> tuple[str, str | None]:
    """Return (paper_id, arxiv_id). Non-arXiv URLs get a stable hash id."""
    arxiv_id = parse_arxiv_id(source)
    if arxiv_id:
        return arxiv_id, arxiv_id
    return "url-" + hashlib.sha1(source.strip().encode()).hexdigest()[:12], None


def pdf_path(paper_id: str):
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)  # a cache on Vercel: /tmp can be emptied
    return PAPERS_DIR / (paper_id.replace("/", "_") + ".pdf")


def pdf_url(paper: dict) -> str:
    """Where the paper's PDF was downloaded from: for arXiv, the exact version the text came from.
    (Papers added before that was recorded: their arXiv ID, or their URL if not on arXiv.)"""
    if paper.get("pdf_url"):
        return paper["pdf_url"]
    return paper["source_url"] if paper["id"].startswith("url-") else f"https://arxiv.org/pdf/{paper['id']}"


async def ensure_pdf(paper: dict):
    """The paper's PDF on disk, downloaded again if it's gone (on Vercel, /tmp is only a cache).
    A download that isn't the file the paper was made from is refused: the stored text, notation
    and elements describe that file (a web page's PDF can change; an arXiv version doesn't)."""
    path = pdf_path(paper["id"])
    if not path.exists():
        pdf = await download_pdf(pdf_url(paper))
        if paper.get("pdf_sha256") and hashlib.sha256(pdf).hexdigest() != paper["pdf_sha256"]:
            raise PaperError(f"The PDF at {pdf_url(paper)} has changed since this paper was added")
        path.write_bytes(pdf)
    return path


def archived(paper_id: str) -> bool:
    """An earlier version kept when the paper was updated: its id has the version ("2210.02747v2")."""
    return parse_arxiv_id(paper_id) is not None and paper_id != parse_arxiv_id(paper_id)


LATEST_TTL = 24 * 3600  # seconds between asking arXiv for a paper's latest version


async def newer_version(paper: dict) -> int | None:
    """The latest arXiv version, if it's newer than the paper's."""
    if not paper.get("arxiv_version") or archived(paper["id"]):
        return None
    latest = paper.get("latest_version")
    if not latest or time.time() - (paper.get("latest_checked_at") or 0) > LATEST_TTL:
        meta = await arxiv_metadata(paper["id"])
        latest = meta and meta["version"]
        if not latest:
            return None
        db.set_latest_version(paper["id"], latest)
    return latest if latest > paper["arxiv_version"] else None


async def arxiv_query(params: dict) -> list[dict]:
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=30, follow_redirects=True) as client:
        r = await client.get(ARXIV_API, params=params)
        r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for entry in root.findall("a:entry", ATOM):
        id_url = entry.findtext("a:id", "", ATOM)
        arxiv_id = parse_arxiv_id(id_url)
        if not arxiv_id:
            continue
        version = re.search(r"v(\d+)$", id_url)
        out.append({
            "arxiv_id": arxiv_id,
            "version": int(version.group(1)) if version else None,  # the latest
            "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
            "authors": [a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)],
            "abstract": " ".join(entry.findtext("a:summary", "", ATOM).split()),
            "published": entry.findtext("a:published", "", ATOM)[:10],
            "updated": entry.findtext("a:updated", "", ATOM)[:10],  # when the latest version came out
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return out


async def arxiv_metadata(arxiv_id: str) -> dict | None:
    results = await arxiv_query({"id_list": arxiv_id, "max_results": 1})
    return results[0] if results else None


async def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    if parse_arxiv_id(query):
        return await arxiv_query({"id_list": parse_arxiv_id(query), "max_results": 1})
    # Quote the query as a title phrase first; fall back to an all-fields search.
    phrase = query.replace('"', "")
    results = await arxiv_query(
        {"search_query": f'ti:"{phrase}"', "max_results": max_results, "sortBy": "relevance"}
    )
    if not results:
        terms = " AND ".join(f"all:{w}" for w in re.findall(r"\w+", phrase)[:12])
        results = await arxiv_query(
            {"search_query": terms, "max_results": max_results, "sortBy": "relevance"}
        )
    return results


def title_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def title_match(a: str, b: str) -> float:
    """Jaccard similarity of the two titles' words."""
    wa, wb = set(title_words(a)), set(title_words(b))
    return len(wa & wb) / max(len(wa | wb), 1)


_ARXIV_DOI = re.compile(rf"10\.48550/arxiv\.({_NEW_ID}|{_OLD_ID})", re.IGNORECASE)


async def _openalex_search(query: str, max_results: int) -> list[dict]:
    params = {"search": query, "per_page": max_results,
              "select": "display_name,publication_year,authorships,doi,primary_location,best_oa_location,locations"}
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=30, follow_redirects=True) as client:
        r = await client.get(OPENALEX_API, params=params)
        r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        if not w.get("display_name"):
            continue
        # OpenAlex merges preprints into the published work, so an arXiv copy shows up as a location.
        urls = [u for loc in w.get("locations") or [] for u in (loc.get("landing_page_url"), loc.get("pdf_url")) if u]
        arxiv_id = next((a for u in urls if (a := parse_arxiv_id(u) or
                                              ((m := _ARXIV_DOI.search(u)) and m.group(1)))), None)
        primary = w.get("primary_location") or {}
        locations = [w.get("best_oa_location") or {}, *(w.get("locations") or [])]
        pdf_urls = [loc.get("pdf_url") for loc in locations] + [
            u for loc in locations if (u := loc.get("landing_page_url")) and u.lower().endswith(".pdf")]
        pdf_urls = list(dict.fromkeys(u for u in pdf_urls if u))
        out.append({
            "title": " ".join(w["display_name"].split()),
            "authors": [a["author"]["display_name"] for a in w.get("authorships") or []
                        if (a.get("author") or {}).get("display_name")],
            "year": str(w.get("publication_year") or ""),
            "venue": (primary.get("source") or {}).get("display_name") or "",
            "arxiv_id": arxiv_id,
            "url": (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else
                    w.get("doi") or primary.get("landing_page_url") or (urls[0] if urls else None)),
            "pdf_urls": pdf_urls,
        })
    return out


async def _first_pdf(client: httpx.AsyncClient, urls: list[str]) -> str | None:
    """The first URL that really serves a PDF; many "PDF" links are paywalls or HTML landing pages."""
    for url in urls[:3]:
        try:
            async with client.stream("GET", url) as r:
                if r.status_code == 200:
                    async for chunk in r.aiter_bytes():
                        if chunk.startswith(b"%PDF"):
                            return url
                        break
        except httpx.HTTPError:
            pass
    return None


async def search_papers(query: str, max_results: int = 8) -> list[dict]:
    """Find papers by name on arXiv and OpenAlex.

    Each result's "open" is what ingest() takes: the arXiv ID, else a PDF URL checked to download,
    else None (the paper can be shown but not opened in the reader).
    """
    arxiv, openalex = await asyncio.gather(
        search_arxiv(query, max_results), _openalex_search(query, max_results), return_exceptions=True
    )
    if isinstance(arxiv, Exception) and isinstance(openalex, Exception):
        raise arxiv
    from_arxiv = [] if isinstance(arxiv, Exception) else [{
        "title": r["title"], "authors": r["authors"], "year": r["published"][:4], "venue": "arXiv",
        "arxiv_id": r["arxiv_id"], "url": r["url"], "pdf_urls": [],
    } for r in arxiv]
    merged: list[dict] = []
    for r in ([] if isinstance(openalex, Exception) else openalex) + from_arxiv:
        # Same paper from both sources (or an OpenAlex duplicate): keep one, preferring an arXiv ID.
        dup = next((m for m in merged if (r["arxiv_id"] and m["arxiv_id"] == r["arxiv_id"])
                    or title_match(m["title"], r["title"]) >= 0.9), None)
        if dup is None:
            merged.append(r)
            continue
        dup["pdf_urls"] += [u for u in r["pdf_urls"] if u not in dup["pdf_urls"]]
        if r["arxiv_id"] and dup["arxiv_id"] in (None, r["arxiv_id"]):
            # OpenAlex sometimes dates a work by a later reprint; the preprint's year is the useful one.
            dup.update(arxiv_id=r["arxiv_id"], url=r["url"], year=min(filter(None, (dup["year"], r["year"])), default=""))
    # Closest title first; Python's sort is stable, so ties keep the sources' relevance order.
    merged.sort(key=lambda r: -round(title_match(query, r["title"]), 1))
    merged = merged[:max_results]

    no_arxiv = [r for r in merged if not r["arxiv_id"] and r["pdf_urls"]]
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=8, follow_redirects=True) as client:
        found = await asyncio.gather(*(asyncio.wait_for(_first_pdf(client, r["pdf_urls"]), 12) for r in no_arxiv),
                                     return_exceptions=True)
    pdfs = {id(r): f for r, f in zip(no_arxiv, found) if isinstance(f, str)}
    for r in merged:
        r["open"] = r["arxiv_id"] or pdfs.get(id(r))
        del r["pdf_urls"]
    return merged


def extract_text(pdf_bytes: bytes) -> tuple[str, int, dict]:
    """Extract text page by page with explicit page markers so the model can cite pages."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").replace("\x00", "").strip()  # NULs: extraction noise, and libsql cuts strings at one
            # Rejoin words hyphenated across line breaks.
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
            pages.append(f"[Page {i}]\n{text}")
        return "\n\n".join(pages), doc.page_count, doc.metadata or {}


async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=120, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise PaperError(f"{url} did not return a PDF")
    return r.content


async def ingest(source: str, known: dict | None = None) -> dict:
    """Download + extract a paper once; later calls return the stored copy.

    known: title/authors/published for a non-arXiv PDF (e.g. from search), used instead of
    guessing them from the PDF.

    The stored text is frozen at ingest time, so the paper message sent to the model is
    byte-identical across every conversation about this paper (keeps the prefix cache warm).
    """
    if stored := db.get_paper(source.strip()):  # by its id, e.g. an archived version ("2210.02747v2")
        return stored
    paper_id, arxiv_id = paper_id_for(source)
    async with _locks[paper_id]:
        existing = db.get_paper(paper_id)
        if existing:
            return existing
        paper, pdf_bytes = await _fetch(source, paper_id, arxiv_id, known)
        pdf_path(paper_id).write_bytes(pdf_bytes)
        db.insert_paper(paper)
        return db.get_paper(paper_id)


async def _fetch(source: str, paper_id: str, arxiv_id: str | None, known: dict | None = None) -> tuple[dict, bytes]:
    """The paper to store (not stored yet), and its PDF. For arXiv, the latest version, pinned:
    pdf_url names that version, so the same file can be downloaded again."""
    if arxiv_id:
        meta = await arxiv_metadata(arxiv_id)
        if meta is None:
            raise PaperError(f"arXiv paper {arxiv_id} not found")
        version = meta["version"]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" + (f"v{version}" if version else "")
        source_url = meta["url"]
    else:
        parsed = urlparse(source.strip())
        if parsed.scheme not in ("http", "https"):
            raise PaperError("Paste an arXiv link/ID or a direct PDF URL")
        meta, version, pdf_url, source_url = None, None, source.strip(), source.strip()

    pdf_bytes = await download_pdf(pdf_url)
    text, num_pages, pdf_meta = await asyncio.to_thread(extract_text, pdf_bytes)

    if meta is None and known and known.get("title"):
        meta = {"title": known["title"][:300], "authors": known.get("authors") or [],
                "abstract": "", "published": known.get("published") or ""}
    if meta is None:
        first_line = next((l for l in text.splitlines()[1:] if len(l.strip()) > 10), source_url)
        meta = {
            "title": (pdf_meta.get("title") or first_line).strip()[:300],
            "authors": [a.strip() for a in (pdf_meta.get("author") or "").split(",") if a.strip()],
            "abstract": "", "published": "",
        }
    return {
        "id": paper_id, "source_url": source_url, "title": meta["title"],
        "authors": meta["authors"], "abstract": meta["abstract"],
        "published": meta["published"], "num_pages": num_pages, "text": text,
        "pdf_url": pdf_url, "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(), "arxiv_version": version,
    }, pdf_bytes


ARCHIVE_FOLDER = "Archive"


async def update_to_latest(paper_id: str) -> dict:
    """Replaces the paper with arXiv's latest version, processed like a new paper. The version
    it was becomes a paper of its own, "<id>v<version>", with its conversations, in the Archive
    folder. The new version is downloaded first, so a failure leaves everything as it was."""
    from . import figures  # figures imports this module

    async with _locks[paper_id]:
        paper = db.get_paper(paper_id)
        if not paper or not paper.get("arxiv_version") or archived(paper_id):
            raise PaperError("Only the current version of an arXiv paper can be updated")
        new, pdf_bytes = await _fetch(paper_id, paper_id, paper_id)
        if new["arxiv_version"] <= paper["arxiv_version"]:
            raise PaperError(f"v{paper['arxiv_version']} is already the latest version")
        archived_id = f"{paper_id}v{paper['arxiv_version']}"
        if db.get_paper(archived_id):
            raise PaperError(f"{archived_id} is already in the library")
        # The files are caches of pinned downloads: one left behind is downloaded again.
        for old, kept in ((pdf_path(paper_id), pdf_path(archived_id)),
                          (figures.crops_dir(paper_id), figures.crops_dir(archived_id))):
            if old.exists():
                old.rename(kept)
        db.archive_and_replace(paper_id, archived_id, db.top_folder(ARCHIVE_FOLDER), new)
        pdf_path(paper_id).write_bytes(pdf_bytes)
        db.set_latest_version(paper_id, new["arxiv_version"])
        return db.get_paper(paper_id)
