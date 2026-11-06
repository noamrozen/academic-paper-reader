"""Downloads of URLs that come from users, models or search results, made safe for the server.

A URL the agent fetches, or a PDF link someone pastes, could point at the server's own network
(localhost, a cloud metadata address, a private range). Every hop of a redirect is checked: the
host must resolve only to public addresses. Responses are read up to a size limit, and the whole
download has a time limit.

What's left: a host that resolves to a public address when checked and to a private one a moment
later (DNS rebinding). Closing that needs connecting to the checked address itself.
"""

import asyncio
import ipaddress
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from .config import HTTP_HEADERS

MAX_REDIRECTS = 5


class FetchError(Exception):
    pass


@dataclass
class Fetched:
    url: str  # after redirects
    status_code: int
    headers: httpx.Headers
    content: bytes

    @property
    def text(self) -> str:
        return httpx.Response(self.status_code, headers=self.headers, content=self.content).text


def _public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


async def check_url(url: str):
    """Raises FetchError unless url is http(s) to a host with only public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError(f"Only http(s) URLs can be fetched: {url[:200]}")
    if parsed.username or parsed.password:
        raise FetchError("URLs with credentials can't be fetched")
    host = parsed.hostname
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                                             type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError(f"Unknown host: {host}") from None
    if not infos or not all(_public(info[4][0]) for info in infos):
        raise FetchError(f"{host} is not a public address")


@asynccontextmanager
async def open_url(url: str, *, timeout: float = 30, headers: dict | None = None):
    """A streaming GET of url, following redirects with each hop checked. Yields the final
    httpx.Response, its body not read yet."""
    async with httpx.AsyncClient(headers=headers or HTTP_HEADERS, timeout=timeout, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await check_url(url)
            request = client.build_request("GET", url)
            response = await client.send(request, stream=True)
            if response.is_redirect and "location" in response.headers:
                await response.aclose()
                url = urljoin(str(response.url), response.headers["location"])
                continue
            try:
                yield response
            finally:
                await response.aclose()
            return
    raise FetchError(f"Too many redirects from {url[:200]}")


async def read_limited(response: httpx.Response, max_bytes: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise FetchError(f"{response.url} is larger than {max_bytes // (1024 * 1024)} MB")
    chunks, size = [], 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise FetchError(f"{response.url} is larger than {max_bytes // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch(url: str, *, max_bytes: int, timeout: float = 30, headers: dict | None = None) -> Fetched:
    """GET url (checked, size-limited); raises for an HTTP error status. timeout bounds the whole download."""
    async def go():
        async with open_url(url, timeout=timeout, headers=headers) as r:
            content = await read_limited(r, max_bytes)
            fetched = Fetched(str(r.url), r.status_code, r.headers, content)
        if fetched.status_code >= 400:
            raise FetchError(f"HTTP {fetched.status_code} from {fetched.url[:200]}")
        return fetched
    try:
        return await asyncio.wait_for(go(), timeout)
    except TimeoutError:
        raise FetchError(f"{url[:200]} took longer than {timeout:.0f} s") from None
