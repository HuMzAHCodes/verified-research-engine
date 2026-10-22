"""search-mcp: web search and page reading, offered as MCP tools.

CONCEPT: building-search-mcp. This is the server's main file (decision 7): it
defines the two tools and wires the parts together, and holds no logic of
its own. The work is done by search.py (Tavily), fetch.py (download and text
extraction), and url_safety.py (the address check).

create_server() takes the search and fetch functions as arguments (decision
3), so tests build the same server with fakes and call it through FastMCP's
in-memory client, with no key and no network.

Run it, and leave it running, in its own terminal (D-32):
    uv run python -m mcp_servers.search_mcp.server
It listens on http://127.0.0.1:8765/mcp, on this machine only.
"""

from functools import partial
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from mcp_servers.search_mcp.config import load_search_server_config
from mcp_servers.search_mcp.fetch import (
    FetchedPage,
    FetchFunction,
    FetchLimits,
    PageFetchError,
    fetch_readable_page,
)
from mcp_servers.search_mcp.search import (
    SearchError,
    SearchFunction,
    SearchResult,
    make_tavily_search,
)

SERVER_INSTRUCTIONS = (
    "Search the public web with web_search, then read a result's page with fetch_page. "
    "fetch_page returns the page's main text, cut to max_chars."
)


def create_server(search_function: SearchFunction, fetch_function: FetchFunction) -> FastMCP:
    """Build search-mcp around the given search and fetch functions."""
    search_server = FastMCP(name="search-mcp", instructions=SERVER_INSTRUCTIONS)

    # CONCEPT: building-search-mcp, decision 1. The type hints are the tool's
    # schema: max_results=11 is rejected before this function runs. The
    # docstring is the tool's description, read by the host.
    @search_server.tool
    async def web_search(
        query: Annotated[str, Field(min_length=1, description="What to search for.")],
        max_results: Annotated[
            int, Field(ge=1, le=10, description="How many results to return, 1 to 10.")
        ] = 5,
    ) -> list[SearchResult]:
        """Search the public web. Returns the title, url, and a snippet for each result."""
        try:
            return await search_function(query, max_results)
        except SearchError as error:
            raise ToolError(str(error)) from error

    @search_server.tool
    async def fetch_page(
        url: Annotated[str, Field(description="A url returned by web_search.")],
        max_chars: Annotated[
            int, Field(ge=200, le=20_000, description="Cut the text at this many characters.")
        ] = 6_000,
    ) -> FetchedPage:
        """Read a web page. Returns its main text, with menus, footers, and markup removed."""
        try:
            return await fetch_function(url, max_chars)
        except PageFetchError as error:
            # Decision 6: a page that cannot be read is a tool error with a
            # reason, never a crash; the researcher skips it.
            raise ToolError(str(error)) from error

    return search_server


def main() -> None:
    server_config = load_search_server_config()
    fetch_limits = FetchLimits(
        timeout_seconds=server_config.fetch_timeout_seconds,
        max_bytes=server_config.fetch_max_bytes,
        max_redirects=server_config.fetch_max_redirects,
    )
    search_server = create_server(
        search_function=make_tavily_search(server_config.tavily_api_key.get_secret_value()),
        fetch_function=partial(fetch_readable_page, limits=fetch_limits),
    )
    search_server.run(
        transport="http",
        host=server_config.search_mcp_host,
        port=server_config.search_mcp_port,
        path="/mcp",
    )


if __name__ == "__main__":
    main()
