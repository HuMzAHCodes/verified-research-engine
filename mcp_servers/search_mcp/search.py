"""Web search through Tavily: a query in, a list of {title, url, snippet} out.

CONCEPT: building-search-mcp, decision 7. This is the only file in the
project's MCP servers that knows about Tavily. Replacing the search provider
means replacing this file; the tool's contract in server.py stays the same.

CONCEPT: building-search-mcp, decision 5, completed in tools-through-mcp
(D-40). Every search has a time limit of its own, 15 s, the same as a page
read and below the client's 30 s limit per call. Without it, Tavily's default
of 60 s meant an unreachable provider showed up in the graph as "search-mcp
stopped answering", blaming a server that was working. With it, the server
answers "search timed out after 15 s", and the researcher treats that as one
failed search.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

SEARCH_TIMEOUT_SECONDS = 15.0


class SearchResult(BaseModel):
    """One web search result."""

    title: str = Field(description="The page title.")
    url: str = Field(description="The page address. Pass it to fetch_page to read the page.")
    snippet: str = Field(description="A short extract from the page, chosen by the search engine.")


class SearchError(RuntimeError):
    """The search provider could not answer."""


SearchFunction = Callable[[str, int], Awaitable[list[SearchResult]]]


class SearchProvider(Protocol):
    """The part of AsyncTavilyClient this module uses; tests pass a fake."""

    async def search(self, query: str, **options: Any) -> dict[str, Any]: ...


def make_tavily_search(
    tavily_api_key: str,
    *,
    timeout_seconds: float = SEARCH_TIMEOUT_SECONDS,
    provider: SearchProvider | None = None,
) -> SearchFunction:
    """Build the search function used in production; tests pass a fake provider."""
    search_provider = provider or AsyncTavilyClient(api_key=tavily_api_key)

    async def search_with_tavily(query: str, max_results: int) -> list[SearchResult]:
        try:
            # Two limits: Tavily's own timeout option, and asyncio.timeout as a
            # backstop in case the library waits longer than it was asked to.
            async with asyncio.timeout(timeout_seconds):
                search_response = await search_provider.search(
                    query,
                    max_results=max_results,
                    search_depth="basic",
                    timeout=timeout_seconds,
                )
        except TimeoutError as error:
            raise SearchError(f"search timed out after {timeout_seconds:g} s") from error
        except Exception as error:  # any provider failure becomes one clear error
            raise SearchError(f"search failed: {type(error).__name__}: {error}") from error
        return [
            SearchResult(
                title=result.get("title", "").strip(),
                url=result["url"],
                snippet=result.get("content", "").strip(),
            )
            for result in search_response.get("results", [])
            if result.get("url")
        ]

    return search_with_tavily
