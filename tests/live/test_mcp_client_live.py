"""Live check of McpToolbox against the search-mcp you started by hand.

CONCEPT: mcp-client. Runs only on request (`uv run pytest -m live`), with the
server running in its own terminal (D-32). It connects at VRE_SEARCH_MCP_URL,
exactly as the researcher will in concept 4, and measures what the concept
notes record: the time to connect and check the tools, one real search (one
Tavily credit), and up to three page reads. Skipped, with the start command,
if the server is not running.
"""

import socket
import time
from urllib.parse import urlsplit

import pytest

from graph.mcp_client import START_SERVER_COMMAND, McpToolbox, ToolCallError
from settings import settings

pytestmark = pytest.mark.live

LIVE_QUERY = "Symbotic warehouse automation customers"
PAGES_TO_READ = 3


def server_is_listening(server_url: str) -> bool:
    """True if something accepts connections at the address's host and port."""
    address = urlsplit(server_url)
    default_port = 443 if address.scheme == "https" else 80
    try:
        with socket.create_connection((address.hostname, address.port or default_port), 1.0):
            return True
    except OSError:
        return False


async def test_the_toolbox_searches_and_reads_through_the_running_server():
    server_url = settings.vre_search_mcp_url
    if not server_is_listening(server_url):
        pytest.skip(
            f"search-mcp is not running at {server_url}; start it with: {START_SERVER_COMMAND}"
        )

    started_at = time.perf_counter()
    async with McpToolbox.connect(server_url) as toolbox:
        print(f"\nconnected and checked the tools in {time.perf_counter() - started_at:.2f} s")

        started_at = time.perf_counter()
        hits = await toolbox.search(LIVE_QUERY)
        print(f"search: {len(hits)} hits in {time.perf_counter() - started_at:.2f} s")

        pages_read = 0
        for hit in hits[:PAGES_TO_READ]:
            started_at = time.perf_counter()
            try:
                page = await toolbox.fetch(hit.url)
            except ToolCallError as error:
                print(f"  skipped  {hit.url[:70]}  ({error})")
                continue
            pages_read += 1
            print(
                f"  read     {page.url[:70]}  {len(page.text)} chars, "
                f"truncated={page.truncated}, {time.perf_counter() - started_at:.2f} s"
            )

    assert hits, "the search returned nothing"
    assert pages_read >= 1, "no page could be read"
