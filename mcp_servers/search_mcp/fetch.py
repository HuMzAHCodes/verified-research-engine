"""Download a page within hard limits and turn it into readable text.

CONCEPT: building-search-mcp, decisions 4 to 6, and choice 2 (D-33). Every
step can fail on a real web page, so every failure becomes PageFetchError
with a reason the researcher can log before skipping the page:

1. the address is checked (url_safety.py), and checked again on every redirect;
2. the download is limited in time, size, and redirects;
3. only HTML and plain-text pages are read;
4. trafilatura keeps the main text and drops menus, footers, and ads;
5. the text is cut to max_chars, and truncated says whether it was.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from pydantic import BaseModel, Field

from mcp_servers.search_mcp.url_safety import (
    BlockedAddressError,
    HostResolver,
    ensure_url_is_safe,
    resolve_host,
)

READABLE_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
# A crawler-style User-Agent: the tool's name, its version, and a "+url" where
# a site owner can see what is reading their pages. The repo is public, so the
# url reveals nothing new; no email is included, so none lands in site logs.
USER_AGENT = "VerifiedResearchEngine/0.1 (+https://github.com/HuMzAHCodes/verified-research-engine)"
# When a page has no article, trafilatura falls back to whatever text it can
# find: a menu or a cookie banner ("Menu", "Accept all"), or on a video page the
# title, a line of description, and footer links (160 characters in the live
# run). Main text shorter than this is treated as no readable text. 300
# characters is about two sentences, too little to quote from anyway; the real
# articles in the live run were all over 2,500.
MIN_MAIN_TEXT_CHARS = 300


class FetchedPage(BaseModel):
    """The readable text of one web page."""

    url: str = Field(description="The address the text was read from, after any redirects.")
    title: str = Field(description="The page title, or empty if it has none.")
    text: str = Field(description="The page's main text, with menus, footers, and markup removed.")
    truncated: bool = Field(description="True when the text was cut at max_chars.")


class PageFetchError(RuntimeError):
    """The page could not be read; the message says why."""


@dataclass(frozen=True)
class FetchLimits:
    timeout_seconds: float = 15.0
    max_bytes: int = 2_000_000
    max_redirects: int = 3


FetchFunction = Callable[[str, int], Awaitable[FetchedPage]]
DEFAULT_FETCH_LIMITS = FetchLimits()


async def fetch_readable_page(
    url: str,
    max_chars: int,
    *,
    limits: FetchLimits = DEFAULT_FETCH_LIMITS,
    resolver: HostResolver = resolve_host,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FetchedPage:
    """Fetch one page and return its main text. Tests pass a fake resolver and transport."""
    try:
        async with asyncio.timeout(limits.timeout_seconds):
            final_url, content_type, body = await download_within_limits(
                url, limits, resolver, transport
            )
    except TimeoutError as error:
        raise PageFetchError(
            f"timed out after {limits.timeout_seconds:.0f} s: {host_of(url)}"
        ) from error
    except BlockedAddressError as error:
        raise PageFetchError(str(error)) from error
    except httpx.HTTPError as error:
        raise PageFetchError(f"could not fetch {host_of(url)}: {type(error).__name__}") from error

    page_text = body.decode("utf-8", errors="replace")
    if content_type == "text/plain":
        main_text, page_title = page_text.strip(), ""
    else:
        # trafilatura is synchronous and CPU-bound: run it in a worker thread
        # so the server keeps answering other calls meanwhile.
        main_text, page_title = await asyncio.to_thread(extract_main_text, page_text)
        if len(main_text) < MIN_MAIN_TEXT_CHARS:
            main_text = ""
    if not main_text:
        raise PageFetchError(f"no readable text on {host_of(final_url)}")

    return FetchedPage(
        url=final_url,
        title=page_title,
        text=main_text[:max_chars],
        truncated=len(main_text) > max_chars,
    )


async def download_within_limits(
    url: str,
    limits: FetchLimits,
    resolver: HostResolver,
    transport: httpx.AsyncBaseTransport | None,
) -> tuple[str, str, bytes]:
    """Follow redirects by hand, checking each address, and stop reading at max_bytes."""
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=False,  # every redirect must pass the address check first
        headers={"User-Agent": USER_AGENT},
    ) as http_client:
        current_url = url
        for _ in range(limits.max_redirects + 1):
            await ensure_url_is_safe(current_url, resolver)
            async with http_client.stream("GET", current_url) as response:
                if response.status_code in REDIRECT_STATUS_CODES:
                    next_location = response.headers.get("location")
                    if not next_location:
                        raise PageFetchError(
                            f"redirect without a location from {host_of(current_url)}"
                        )
                    current_url = urljoin(current_url, next_location)
                    continue
                if response.status_code >= 400:
                    raise PageFetchError(f"HTTP {response.status_code} from {host_of(current_url)}")

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if content_type not in READABLE_CONTENT_TYPES:
                    shown_type = content_type or "no content type"
                    raise PageFetchError(f"not a text page ({shown_type}): {host_of(current_url)}")
                body = await read_at_most(response, limits.max_bytes)
                return current_url, content_type, body
        raise PageFetchError(f"more than {limits.max_redirects} redirects from {host_of(url)}")


async def read_at_most(response: httpx.Response, max_bytes: int) -> bytes:
    """The body, but never more than max_bytes: a huge page is read in part, not in full."""
    received_chunks: list[bytes] = []
    received_size = 0
    async for chunk in response.aiter_bytes():
        received_chunks.append(chunk)
        received_size += len(chunk)
        if received_size >= max_bytes:
            break
    return b"".join(received_chunks)[:max_bytes]


def extract_main_text(html: str) -> tuple[str, str]:
    """The page's main text and title, using trafilatura (D-33)."""
    main_text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    page_metadata = trafilatura.extract_metadata(html)
    page_title = (page_metadata.title or "") if page_metadata else ""
    return main_text.strip(), page_title.strip()


def host_of(url: str) -> str:
    return urlsplit(url).hostname or url
