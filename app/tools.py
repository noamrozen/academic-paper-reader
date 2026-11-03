"""Agent tools, defined once in a provider-neutral form (name, description, JSON schema)."""

import asyncio
import json
import re

import httpx
from bs4 import BeautifulSoup

from . import papers
from .config import HTTP_HEADERS, TAVILY_API_KEY

MAX_TOOL_CHARS = 40_000

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "1-10, default 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a web page or PDF and return its text content (truncated if long).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for papers by title, keywords or arXiv ID. Use it to resolve a citation "
            "from the paper's reference list to an arXiv ID. Returns IDs, titles, authors and abstracts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Paper title, keywords, or an arXiv ID"},
                "max_results": {"type": "integer", "description": "1-10, default 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_arxiv_paper",
        "description": (
            "Download an arXiv paper and return its full text with [Page N] markers. Use offset to "
            "continue reading a long paper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID, e.g. 1706.03762"},
                "offset": {"type": "integer", "description": "Character offset to start from, default 0"},
            },
            "required": ["arxiv_id"],
        },
    },
]


class ToolResult:
    def __init__(self, content: str, is_error: bool = False, papers: list[dict] | None = None):
        self.content = content
        self.is_error = is_error
        self.papers = papers or []  # surfaced in the UI as "open in reader" cards


def _clip(text: str, offset: int = 0) -> str:
    chunk = text[offset: offset + MAX_TOOL_CHARS]
    if offset + MAX_TOOL_CHARS < len(text):
        chunk += f"\n\n[truncated: showing chars {offset}-{offset + len(chunk)} of {len(text)}]"
    return chunk


async def web_search(query: str, max_results: int = 5) -> ToolResult:
    max_results = max(1, min(int(max_results or 5), 10))
    if TAVILY_API_KEY:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
            )
            r.raise_for_status()
        hits = [{"title": h["title"], "url": h["url"], "snippet": h.get("content", "")}
                for h in r.json().get("results", [])]
    else:
        from ddgs import DDGS

        raw = await asyncio.to_thread(lambda: DDGS().text(query, max_results=max_results))
        hits = [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in raw]
    if not hits:
        return ToolResult("No results.")
    found = [p for h in hits if (p := _arxiv_card(h["url"], h["title"]))]
    return ToolResult(json.dumps(hits, indent=1, ensure_ascii=False), papers=found)


def _arxiv_card(url: str | None, title: str | None) -> dict | None:
    arxiv_id = papers.parse_arxiv_id(url or "")
    return {"arxiv_id": arxiv_id, "title": title or arxiv_id} if arxiv_id else None


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return (f"# {title}\n\n" if title else "") + text.strip()


async def fetch_url(url: str) -> ToolResult:
    if papers.parse_arxiv_id(url) and "arxiv.org" in url:
        return await read_arxiv_paper(papers.parse_arxiv_id(url))
    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=30, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or r.content.startswith(b"%PDF"):
        text, _, _ = await asyncio.to_thread(papers.extract_text, r.content)
    elif "html" in ctype or "<html" in r.text[:2000].lower():
        text = _html_to_text(r.text)
    else:
        text = r.text
    return ToolResult(_clip(text))


async def search_arxiv(query: str, max_results: int = 5) -> ToolResult:
    results = await papers.search_arxiv(query, max(1, min(int(max_results or 5), 10)))
    if not results:
        return ToolResult("No arXiv results.")
    return ToolResult(
        json.dumps(results, indent=1, ensure_ascii=False),
        papers=[{"arxiv_id": r["arxiv_id"], "title": r["title"]} for r in results],
    )


async def read_arxiv_paper(arxiv_id: str, offset: int = 0) -> ToolResult:
    parsed = papers.parse_arxiv_id(arxiv_id)
    if not parsed:
        return ToolResult(f"'{arxiv_id}' is not a valid arXiv ID", is_error=True)
    paper = await papers.ingest(parsed)  # also stores it, so "open in reader" is instant
    full = papers.db.get_paper(paper["id"], with_text=True)
    header = f"Title: {full['title']}\nAuthors: {', '.join(full['authors'])}\n\n"
    return ToolResult(
        header + _clip(full["text"], int(offset or 0)),
        papers=[{"arxiv_id": full["id"], "title": full["title"]}],
    )


_IMPLS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "search_arxiv": search_arxiv,
    "read_arxiv_paper": read_arxiv_paper,
}


async def run_tool(name: str, args: dict) -> ToolResult:
    impl = _IMPLS.get(name)
    if impl is None:
        return ToolResult(f"Unknown tool: {name}", is_error=True)
    if not isinstance(args, dict):
        return ToolResult("Tool input must be a JSON object", is_error=True)
    try:
        return await impl(**args)
    except TypeError as e:
        return ToolResult(f"Invalid arguments for {name}: {e}", is_error=True)
    except Exception as e:  # tool failures go back to the model, not up the stack
        return ToolResult(f"{type(e).__name__}: {e}", is_error=True)
