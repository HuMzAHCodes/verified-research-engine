"""Web search through Tavily: a query in, a list of {title, url, snippet} out.

CONCEPT: building-search-mcp, decision 7. This is the only file in the
project's MCP servers that knows about Tavily. Replacing the search provider
means replacing this file; the tool's contract in server.py stays the same.
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient


class SearchResult(BaseModel):
    """One web search result."""

    title: str = Field(description="The page title.")
    url: str = Field(description="The page address. Pass it to fetch_page to read the page.")
    snippet: str = Field(description="A short extract from the page, chosen by the search engine.")


class SearchError(RuntimeError):
    """The search provider could not answer."""


SearchFunction = Callable[[str, int], Awaitable[list[SearchResult]]]


def make_tavily_search(tavily_api_key: str) -> SearchFunction:
    """Build the search function used in production."""
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    async def search_with_tavily(query: str, max_results: int) -> list[SearchResult]:
        try:
            search_response = await tavily_client.search(
                query, max_results=max_results, search_depth="basic"
            )
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
