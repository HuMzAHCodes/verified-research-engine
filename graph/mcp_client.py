"""The graph's side of MCP: one toolbox that reaches search-mcp.

CONCEPT: mcp-client. Nodes never see MCP, JSON, or result envelopes. They get a
Toolbox with two methods, search and fetch, that return typed objects and fail
in one of two ways:

- ToolCallError: the server answered, but this one search or page failed (a
  blocked address, HTTP 403, no readable text). Skip it and carry on.
- SearchServerUnavailable: the server cannot be used (not running, stopped
  answering, missing a tool, or answered with the wrong shape). Stop the run.

McpToolbox wraps FastMCP's own Client (D-34). One connection serves a whole
research run; whoever runs the graph opens it with `async with` (decision 4):

    async with McpToolbox.connect(settings.vre_search_mcp_url) as toolbox:
        graph = build_graph(gateway, toolbox)
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Protocol

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from langsmith import traceable
from pydantic import BaseModel, ValidationError

REQUIRED_TOOLS = frozenset({"web_search", "fetch_page"})
# Above the server's own 15 s limit per page, so a slow page comes back from the
# server as a tool error; only a server that has stopped answering reaches this.
CALL_TIMEOUT_SECONDS = 30.0
START_SERVER_COMMAND = "uv run python -m mcp_servers.search_mcp.server"


# CONCEPT: mcp-client, decision 3. The graph's own copies of the server's result
# shapes. The graph never imports from mcp_servers/, so the two stay separate
# programs; the protocol is the contract, checked here when results arrive.
class SearchHit(BaseModel):
    """One web search result."""

    title: str
    url: str
    snippet: str


class WebPage(BaseModel):
    """The readable text of one page."""

    url: str
    title: str
    text: str
    truncated: bool


class ToolCallError(RuntimeError):
    """The server could not do this one call; the message is the server's reason."""


class SearchServerUnavailable(RuntimeError):
    """search-mcp cannot be used for this run; the message says why and what to do."""


class Toolbox(Protocol):
    """What nodes see (decision 2). McpToolbox and the tests' FakeToolbox both fit."""

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...

    async def fetch(self, url: str, max_chars: int = 6_000) -> WebPage: ...


def without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the toolbox object itself out of the trace; show only the arguments."""
    return {name: value for name, value in inputs.items() if name != "self"}


class McpToolbox:
    """Toolbox over one open MCP connection to search-mcp."""

    def __init__(self, mcp_client: Client, server_label: str) -> None:
        self._mcp_client = mcp_client
        self._server_label = server_label

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        server: str | FastMCP,
        *,
        call_timeout_seconds: float = CALL_TIMEOUT_SECONDS,
    ) -> AsyncIterator["McpToolbox"]:
        """Open the connection, check the tools, and close it when the block ends.

        `server` is the server's address, or in tests the server object itself,
        which FastMCP's client then reaches in memory (decision 9).
        """
        server_label = server if isinstance(server, str) else f"in-memory server {server.name}"
        mcp_client = Client(server, timeout=call_timeout_seconds)
        # AsyncExitStack closes the connection when the block ends, while letting
        # this method tell a failure to connect apart from an error raised later
        # by the caller's own code inside the `async with` block.
        async with AsyncExitStack() as exit_stack:
            try:
                await exit_stack.enter_async_context(mcp_client)
            except Exception as error:
                raise SearchServerUnavailable(
                    f"cannot connect to search-mcp at {server_label}; "
                    f"start it with: {START_SERVER_COMMAND}"
                ) from error
            toolbox = cls(mcp_client, server_label)
            await toolbox.check_required_tools()
            yield toolbox

    async def check_required_tools(self) -> None:
        """Fail fast (decision 5): both tools must exist before any work is spent."""
        try:
            offered_tools = await self._mcp_client.list_tools()
        except Exception as error:
            raise SearchServerUnavailable(
                f"search-mcp at {self._server_label} did not list its tools: {type(error).__name__}"
            ) from error
        missing_tools = REQUIRED_TOOLS - {tool.name for tool in offered_tools}
        if missing_tools:
            raise SearchServerUnavailable(
                f"search-mcp at {self._server_label} is missing "
                f"{', '.join(sorted(missing_tools))}; is it the right server?"
            )

    # CONCEPT: mcp-client, decision 7. Each call is a tool run in the LangSmith
    # trace, next to the model calls, with its arguments and its result.
    @traceable(run_type="tool", name="web_search", process_inputs=without_self)
    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        structured_result = await self.call_tool(
            "web_search", {"query": query, "max_results": max_results}
        )
        try:
            return [SearchHit.model_validate(hit) for hit in structured_result["result"]]
        except (KeyError, TypeError, ValidationError) as error:
            raise self.unexpected_result("web_search", error) from error

    @traceable(run_type="tool", name="fetch_page", process_inputs=without_self)
    async def fetch(self, url: str, max_chars: int = 6_000) -> WebPage:
        structured_result = await self.call_tool("fetch_page", {"url": url, "max_chars": max_chars})
        try:
            return WebPage.model_validate(structured_result)
        except ValidationError as error:
            raise self.unexpected_result("fetch_page", error) from error

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one tool and sort its failure into one of the two kinds (decision 6)."""
        try:
            call_result = await self._mcp_client.call_tool(tool_name, arguments)
        except ToolError as error:
            # The server answered with a reason: this one input failed.
            raise ToolCallError(str(error)) from error
        except Exception as error:
            # No answer at all: a lost connection or the call's time limit.
            raise SearchServerUnavailable(
                f"search-mcp at {self._server_label} stopped answering during "
                f"{tool_name}: {type(error).__name__}"
            ) from error
        if not isinstance(call_result.structured_content, dict):
            raise self.unexpected_result(tool_name, "no structured content")
        return call_result.structured_content

    def unexpected_result(self, tool_name: str, reason: object) -> SearchServerUnavailable:
        return SearchServerUnavailable(
            f"search-mcp at {self._server_label} returned an unexpected result "
            f"for {tool_name}: {reason}"
        )
