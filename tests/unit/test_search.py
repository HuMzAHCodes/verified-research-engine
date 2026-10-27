"""Tests of the server's web search, with a fake search provider.

CONCEPT: building-search-mcp, decisions 5 and 7, and D-40. No Tavily key and
no network: a fake provider answers, fails, or stops answering, and the tests
check what the search function makes of each.
"""

import asyncio

import pytest

from mcp_servers.search_mcp.search import SearchError, SearchResult, make_tavily_search


class FakeProvider:
    """Plays Tavily: returns a response, raises an error, or hangs."""

    def __init__(self, response=None, error=None, delay_seconds=0.0):
        self.response = response or {"results": []}
        self.error = error
        self.delay_seconds = delay_seconds
        self.received_options = {}

    async def search(self, query, **options):
        self.received_options = options
        await asyncio.sleep(self.delay_seconds)
        if self.error:
            raise self.error
        return self.response


def search_with(provider, timeout_seconds=15.0):
    return make_tavily_search("unused-key", timeout_seconds=timeout_seconds, provider=provider)


async def test_results_are_shaped_and_results_without_a_url_are_dropped():
    provider = FakeProvider(
        response={
            "results": [
                {"title": " Symbotic ", "url": "https://example.com/a", "content": " Robots. "},
                {"title": "No address", "content": "Dropped."},
            ]
        }
    )

    results = await search_with(provider)("Symbotic", 5)

    assert results == [
        SearchResult(title="Symbotic", url="https://example.com/a", snippet="Robots.")
    ]


async def test_the_provider_is_given_the_time_limit():
    provider = FakeProvider()

    await search_with(provider, timeout_seconds=15.0)("Symbotic", 5)

    assert provider.received_options["timeout"] == 15.0
    assert provider.received_options["max_results"] == 5


async def test_a_provider_that_stops_answering_is_a_timed_out_search():
    # D-40: the server reports the slow provider itself, well inside the
    # client's own 30 s limit, instead of leaving the client to blame the server.
    provider = FakeProvider(delay_seconds=5.0)

    with pytest.raises(SearchError, match="search timed out after 0.2 s"):
        await search_with(provider, timeout_seconds=0.2)("Symbotic", 5)


async def test_a_provider_error_is_a_search_error_with_the_reason():
    provider = FakeProvider(error=ConnectionError("name resolution failed"))

    with pytest.raises(SearchError, match="ConnectionError: name resolution failed"):
        await search_with(provider)("Symbotic", 5)
