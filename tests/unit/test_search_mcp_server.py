"""TC-MCP-01: search-mcp's two tools, called through the MCP protocol.

CONCEPT: building-search-mcp, decision 3. The real server is built with fake
search and fetch functions and called through FastMCP's in-memory client, so
schema validation, serialisation, and tool errors are all real, with no
Tavily key and no network.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_servers.search_mcp.fetch import FetchedPage, PageFetchError
from mcp_servers.search_mcp.search import SearchError, SearchResult
from mcp_servers.search_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[SearchResult]:
    if query == "provider down":
        raise SearchError("search failed: provider down")
    return [
        SearchResult(title=f"Result {number}", url=f"https://example.com/{number}", snippet="...")
        for number in range(1, max_results + 1)
    ]


async def fake_fetch(url: str, max_chars: int) -> FetchedPage:
    if "blocked" in url:
        raise PageFetchError("blocked address: 10.0.0.1 is not a public internet address")
    page_text = "Page text. " * 1_000
    return FetchedPage(
        url=url, title="A page", text=page_text[:max_chars], truncated=len(page_text) > max_chars
    )


@pytest.fixture
def search_server():
    return create_server(fake_search, fake_fetch)


async def test_tc_mcp_01_the_server_offers_both_tools(search_server):
    async with Client(search_server) as mcp_client:
        tool_names = {tool.name for tool in await mcp_client.list_tools()}

    assert tool_names == {"web_search", "fetch_page"}


async def test_tc_mcp_01_web_search_returns_title_url_and_snippet(search_server):
    async with Client(search_server) as mcp_client:
        search_result = await mcp_client.call_tool(
            "web_search", {"query": "Symbotic", "max_results": 2}
        )

    results = search_result.structured_content["result"]
    assert len(results) == 2
    assert set(results[0]) == {"title", "url", "snippet"}


async def test_tc_mcp_01_web_search_defaults_to_five_results(search_server):
    async with Client(search_server) as mcp_client:
        search_result = await mcp_client.call_tool("web_search", {"query": "Symbotic"})

    assert len(search_result.structured_content["result"]) == 5


async def test_tc_mcp_01_more_than_ten_results_is_rejected_by_the_schema(search_server):
    async with Client(search_server) as mcp_client:
        with pytest.raises(ToolError, match="less than or equal to 10"):
            await mcp_client.call_tool("web_search", {"query": "Symbotic", "max_results": 11})


async def test_tc_mcp_01_fetch_page_cuts_text_and_says_so(search_server):
    async with Client(search_server) as mcp_client:
        fetch_result = await mcp_client.call_tool(
            "fetch_page", {"url": "https://example.com/1", "max_chars": 500}
        )

    fetched_page = fetch_result.structured_content
    assert set(fetched_page) == {"url", "title", "text", "truncated"}
    assert len(fetched_page["text"]) == 500
    assert fetched_page["truncated"] is True


async def test_a_page_that_cannot_be_read_is_a_tool_error_with_the_reason(search_server):
    async with Client(search_server) as mcp_client:
        with pytest.raises(ToolError, match="blocked address"):
            await mcp_client.call_tool("fetch_page", {"url": "http://blocked.example.com/"})


async def test_a_search_failure_is_a_tool_error_with_the_reason(search_server):
    async with Client(search_server) as mcp_client:
        with pytest.raises(ToolError, match="provider down"):
            await mcp_client.call_tool("web_search", {"query": "provider down"})
