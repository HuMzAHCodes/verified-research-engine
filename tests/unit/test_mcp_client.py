"""Tests of McpToolbox against the real server code, in memory.

CONCEPT: mcp-client, decision 9. Both sides of the protocol are real: concept
1's create_server with its tool schemas and error handling, and FastMCP's client
inside McpToolbox. Only the search and fetch functions given to the server are
fakes, and the connection is FastMCP's in-memory transport. No network, no key.
"""

import asyncio
import socket

import pytest
from fastmcp import FastMCP

from graph.mcp_client import (
    START_SERVER_COMMAND,
    McpToolbox,
    SearchHit,
    SearchServerUnavailable,
    ToolCallError,
    WebPage,
)
from mcp_servers.search_mcp.fetch import FetchedPage, PageFetchError
from mcp_servers.search_mcp.search import SearchResult
from mcp_servers.search_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[SearchResult]:
    return [
        SearchResult(title=f"{query} {number}", url=f"https://example.com/{number}", snippet="s")
        for number in range(1, max_results + 1)
    ]


async def fake_fetch(url: str, max_chars: int) -> FetchedPage:
    if url.endswith("/blocked"):
        raise PageFetchError("HTTP 403 from example.com")
    return FetchedPage(url=url, title="A page", text="Main text. " * 5, truncated=False)


def search_server() -> FastMCP:
    return create_server(fake_search, fake_fetch)


def server_with_tools(**tools) -> FastMCP:
    """A stand-in server with hand-written tools, to play a wrong or broken server."""
    stand_in_server = FastMCP(name="stand-in")
    for tool_function in tools.values():
        stand_in_server.tool(tool_function)
    return stand_in_server


async def test_search_returns_typed_hits_through_the_real_server():
    async with McpToolbox.connect(search_server()) as toolbox:
        hits = await toolbox.search("Symbotic customers", max_results=2)

    assert hits == [
        SearchHit(title="Symbotic customers 1", url="https://example.com/1", snippet="s"),
        SearchHit(title="Symbotic customers 2", url="https://example.com/2", snippet="s"),
    ]


async def test_fetch_returns_a_typed_page():
    async with McpToolbox.connect(search_server()) as toolbox:
        page = await toolbox.fetch("https://example.com/a")

    assert isinstance(page, WebPage)
    assert page.url == "https://example.com/a"
    assert page.text.startswith("Main text.")


async def test_a_page_the_server_cannot_read_is_a_tool_call_error_with_its_reason():
    async with McpToolbox.connect(search_server()) as toolbox:
        with pytest.raises(ToolCallError, match="HTTP 403 from example.com"):
            await toolbox.fetch("https://example.com/blocked")
        # The connection survives a tool error: the next call still works.
        assert await toolbox.search("next query", max_results=1)


async def test_a_server_that_is_not_running_gives_the_start_command():
    # Find a free local port, then close it, so nothing is listening there.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    with pytest.raises(SearchServerUnavailable, match="cannot connect") as error_info:
        async with McpToolbox.connect(f"http://127.0.0.1:{free_port}/mcp"):
            pass
    assert START_SERVER_COMMAND in str(error_info.value)


async def test_a_server_missing_a_tool_is_refused_before_any_call():
    async def web_search(query: str, max_results: int = 5) -> list[dict]:
        return []

    with pytest.raises(SearchServerUnavailable, match="missing fetch_page"):
        async with McpToolbox.connect(server_with_tools(web_search=web_search)):
            pass


async def test_a_result_with_the_wrong_shape_stops_the_run():
    async def web_search(query: str, max_results: int = 5) -> list[dict]:
        return [{"name": "no title, url, or snippet"}]

    async def fetch_page(url: str, max_chars: int = 6_000) -> dict:
        return {}

    stand_in_server = server_with_tools(web_search=web_search, fetch_page=fetch_page)
    async with McpToolbox.connect(stand_in_server) as toolbox:
        with pytest.raises(SearchServerUnavailable, match="unexpected result for web_search"):
            await toolbox.search("anything")


async def test_a_server_that_stops_answering_hits_the_time_limit():
    async def web_search(query: str, max_results: int = 5) -> list[dict]:
        await asyncio.sleep(5)
        return []

    async def fetch_page(url: str, max_chars: int = 6_000) -> dict:
        return {}

    stand_in_server = server_with_tools(web_search=web_search, fetch_page=fetch_page)
    async with McpToolbox.connect(stand_in_server, call_timeout_seconds=0.3) as toolbox:
        with pytest.raises(SearchServerUnavailable, match="stopped answering during web_search"):
            await toolbox.search("anything")
