"""FakeToolbox must behave like McpToolbox wherever a node can tell.

CONCEPT: mcp-client, decision 9. A fake that drifts from the real thing makes
node tests pass for the wrong reason. The contract tests below run the same
checks against both: McpToolbox over the real server code in memory, and
FakeToolbox, given the same pages. Then two checks of what only the fake does.
"""

from contextlib import asynccontextmanager

import pytest

from graph.mcp_client import McpToolbox, SearchServerUnavailable, ToolCallError
from mcp_servers.search_mcp.fetch import FetchedPage, PageFetchError
from mcp_servers.search_mcp.search import SearchResult
from mcp_servers.search_mcp.server import create_server
from tests.conftest import FakeToolbox, search_hit, web_page

QUERY = "Symbotic customers"
HITS = [search_hit(1), search_hit(2), search_hit(3)]
PAGE = web_page(1)


@asynccontextmanager
async def real_toolbox():
    """McpToolbox over create_server, whose search and fetch serve HITS and PAGE."""

    async def search_function(query: str, max_results: int) -> list[SearchResult]:
        hits = HITS if query == QUERY else []
        return [SearchResult(**hit.model_dump()) for hit in hits[:max_results]]

    async def fetch_function(url: str, max_chars: int) -> FetchedPage:
        if url != PAGE.url:
            raise PageFetchError(f"no readable text on {url}")
        return FetchedPage(
            url=PAGE.url,
            title=PAGE.title,
            text=PAGE.text[:max_chars],
            truncated=len(PAGE.text) > max_chars,
        )

    async with McpToolbox.connect(create_server(search_function, fetch_function)) as toolbox:
        yield toolbox


@asynccontextmanager
async def fake_toolbox():
    yield FakeToolbox(hits_by_query={QUERY: HITS}, pages_by_url={PAGE.url: PAGE})


@pytest.fixture(params=[real_toolbox, fake_toolbox], ids=["McpToolbox", "FakeToolbox"])
def open_toolbox(request):
    return request.param


async def test_search_returns_the_same_typed_hits_cut_to_max_results(open_toolbox):
    async with open_toolbox() as toolbox:
        assert await toolbox.search(QUERY, max_results=2) == HITS[:2]


async def test_a_query_with_no_results_finds_nothing(open_toolbox):
    async with open_toolbox() as toolbox:
        assert await toolbox.search("nothing matches this") == []


async def test_fetch_cuts_text_at_max_chars_and_says_so(open_toolbox):
    async with open_toolbox() as toolbox:
        page = await toolbox.fetch(PAGE.url, max_chars=200)

    assert page.text == PAGE.text[:200]
    assert page.truncated is True


async def test_a_page_that_cannot_be_read_is_a_tool_call_error(open_toolbox):
    async with open_toolbox() as toolbox:
        with pytest.raises(ToolCallError, match="no readable text"):
            await toolbox.fetch("https://example.com/unknown")


async def test_the_fake_records_every_call():
    toolbox = FakeToolbox(hits_by_query={QUERY: HITS}, pages_by_url={PAGE.url: PAGE})

    await toolbox.search(QUERY)
    await toolbox.fetch(PAGE.url)

    assert toolbox.received_queries == [QUERY]
    assert toolbox.received_urls == [PAGE.url]


async def test_the_fake_raises_a_scripted_error():
    toolbox = FakeToolbox(hits_by_query={QUERY: SearchServerUnavailable("server gone")})

    with pytest.raises(SearchServerUnavailable, match="server gone"):
        await toolbox.search(QUERY)
