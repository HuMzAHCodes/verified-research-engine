"""Live checks of search-mcp: a real Tavily search, real pages, and the real server.

CONCEPT: building-search-mcp. Runs only on request (`uv run pytest -m live`).

1. test_search_and_read_real_pages builds the server exactly as main() builds
   it, with the real search and fetch functions, and calls it through FastMCP's
   in-memory client. One search (one Tavily credit) and up to five page reads
   (no Tavily credits, D-33).
2. test_the_running_server_answers_over_http talks to the server you started
   by hand in a second terminal (D-32), over HTTP on localhost, exactly as the
   researcher will in the mcp-client concept. It only lists the tools, so it
   uses no Tavily credit. If the server is not running, the test is skipped
   and says how to start it.
"""

import socket
import time
from functools import partial

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_servers.search_mcp.config import load_search_server_config
from mcp_servers.search_mcp.fetch import FetchLimits, fetch_readable_page
from mcp_servers.search_mcp.search import make_tavily_search
from mcp_servers.search_mcp.server import create_server

pytestmark = pytest.mark.live

LIVE_QUERY = "Symbotic warehouse automation customers"
START_SERVER_COMMAND = "uv run python -m mcp_servers.search_mcp.server"


async def test_search_and_read_real_pages():
    server_config = load_search_server_config()
    search_server = create_server(
        make_tavily_search(server_config.tavily_api_key.get_secret_value()),
        partial(fetch_readable_page, limits=FetchLimits()),
    )

    async with Client(search_server) as mcp_client:
        started_at = time.perf_counter()
        search_result = await mcp_client.call_tool("web_search", {"query": LIVE_QUERY})
        search_seconds = time.perf_counter() - started_at
        results = search_result.structured_content["result"]
        print(f"\nweb_search: {len(results)} results in {search_seconds:.1f} s")

        pages_read, pages_skipped = 0, 0
        for result in results:
            started_at = time.perf_counter()
            try:
                fetch_result = await mcp_client.call_tool("fetch_page", {"url": result["url"]})
            except ToolError as error:
                pages_skipped += 1
                print(f"  skipped  {result['url'][:70]}  ({error})")
                continue
            fetched_page = fetch_result.structured_content
            pages_read += 1
            print(
                f"  read     {result['url'][:70]}  {len(fetched_page['text'])} chars, "
                f"truncated={fetched_page['truncated']}, {time.perf_counter() - started_at:.1f} s"
            )

    print(f"pages read: {pages_read}, skipped: {pages_skipped}")
    assert results, "the search returned nothing"
    assert pages_read >= 1, "no page could be read"


def server_is_listening(host: str, port: int) -> bool:
    """True if something accepts connections on host:port (the server you started)."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


async def test_the_running_server_answers_over_http():
    server_config = load_search_server_config()
    host, port = server_config.search_mcp_host, server_config.search_mcp_port
    if not server_is_listening(host, port):
        pytest.skip(
            f"search-mcp is not running on {host}:{port}; start it with: {START_SERVER_COMMAND}"
        )

    server_url = f"http://{host}:{port}/mcp"
    async with Client(server_url) as mcp_client:
        tool_names = sorted(tool.name for tool in await mcp_client.list_tools())

    print(f"\n{server_url} offers: {', '.join(tool_names)}")
    assert tool_names == ["fetch_page", "web_search"]
